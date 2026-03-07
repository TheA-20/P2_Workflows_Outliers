"""
benchmark_outliers.py
=====================
Compara todos los modelos de detección de anomalías del notebook pracica2.ipynb
para cada combinación de hiperparámetros, mide tiempos de ejecución y métricas
de evaluación, y genera benchmark_results.csv con los resultados.

Modelos evaluados
-----------------
  Offline (batch):
    1. IQR                - factor
    2. Z-Score            - k_threshold
    3. K-Means Batch      - n_clusters × dist_percentile
    4. Isolation Forest   - n_estimators × contamination

  Online (streaming, river):
    5. IQR Online         - window_size × factor
    6. Z-Score Online     - halflife (EMA) × k_threshold
    7. K-Means Online     - n_clusters × halflife × dist_threshold
    8. Half-Space Trees   - n_trees × window_size × height × score_threshold

Definiciones de anomalía evaluadas
-----------------------------------
  "BROKEN_only"        : solo BROKEN  → 1, el resto → 0
  "BROKEN+RECOVERING" : BROKEN o RECOVERING → 1, NORMAL → 0
"""

import itertools
import math
import time
import warnings
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend no interactivo: guarda a disco sin necesitar pantalla
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from river import anomaly, cluster, preprocessing as river_prep

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Almacén global de predicciones por muestra
# clave  → _pred_key(row_dict) ; valor → np.int8 array shape (n_samples,)
# ---------------------------------------------------------------------------
PRED_STORE: dict[str, np.ndarray] = {}

# Columnas de hiperparámetros (orden canónico compartido por todas las funciones)
_HP_COLS = [
    "factor", "k_threshold", "n_clusters", "dist_percentile",
    "n_estimators", "contamination", "halflife", "dist_threshold",
    "n_trees", "window_size", "height", "score_threshold",
]

# ---------------------------------------------------------------------------
# 0. Configuración de hiperparámetros
# ---------------------------------------------------------------------------

HYP = {
    "iqr": {
        "factor": [1.5, 2.0, 2.5, 3.0, 3.5],
    },
    "zscore": {
        "k_threshold": [2.0, 2.5, 3.0, 3.5, 4.0],
    },
    "kmeans_batch": {
        "n_clusters":     [2, 3, 4, 5, 8, 10],
        "dist_percentile": [90, 95, 99],
    },
    "iforest": {
        "n_estimators": [50, 100, 200],
        "contamination": [0.01, 0.05, 0.10, 0.15, 0.20],
    },
    "kmeans_online": {
        "n_clusters":     [2, 3, 4, 5],
        "halflife":       [0.1, 0.3, 0.5],
        "dist_threshold": [2.0, 2.5, 3.0, 3.5, 4.0],
    },
    "hst": {
        "n_trees":       [5, 10, 30, 50],
        "window_size":   [250, 1000, 5000],
        "height":        [8, 12, 15],
        "score_threshold": [0.80, 0.90, 0.95, 0.99],
    },
    # Online IQR: ventana deslizante por sensor
    "iqr_online": {
        "window_size": [50, 200, 500, 1000],
        "factor":     [1.5, 2.0, 2.5, 3.0, 3.5],
    },
    # Online Z-Score: media/varianza con EMA (halflife controla la memoria)
    "zscore_online": {
        "halflife":    [50, 200, 500, 1000],
        "k_threshold": [2.0, 2.5, 3.0, 3.5, 4.0],
    },
}

# ---------------------------------------------------------------------------
# Definiciones de etiqueta de anomalía a comparar
# ---------------------------------------------------------------------------
# BROKEN   → la máquina ha fallado (anomalía clara)
# RECOVERING → la máquina se está recuperando (estado transicional que
#              algunos consideran anómalo; lo evaluamos como opción)
# NORMAL   → operación normal
ANOMALY_LABELS: dict[str, list[str]] = {
    "BROKEN+RECOVERING":  ["BROKEN", "RECOVERING"],
}

# ---------------------------------------------------------------------------
# 1. Carga y preprocesado de datos  (igual que el notebook)
# ---------------------------------------------------------------------------

def load_data(
    csv_path: str = "sensor.csv",
    anomaly_labels: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Devuelve:
      df_sensors   – DataFrame únicamente con columnas sensor_*
      df_full      – DataFrame completo (con is_anomaly_real)
      y_true       – array con las etiquetas reales (0/1)

    anomaly_labels: lista de valores de machine_status que se consideran
                    anomalía (p.ej. ["BROKEN"] o ["BROKEN", "RECOVERING"]).
                    Por defecto ["BROKEN"].
    """
    if anomaly_labels is None:
        anomaly_labels = ["BROKEN"]

    df = pd.read_csv(csv_path)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    # Etiqueta real — configurable según los estados que se consideren anomalía
    if "machine_status" in df.columns:
        df["is_anomaly_real"] = df["machine_status"].apply(
            lambda x: 1 if x in anomaly_labels else 0
        )

    # Limpieza
    df = df.drop_duplicates()
    df.drop(columns=["sensor_15"], inplace=True, errors="ignore")

    numeric_columns = df.filter(regex="sensor").columns
    df[numeric_columns] = (
        df[numeric_columns].ffill().fillna(0)
    )

    # Eliminamos columnas no-sensor para el modelado
    df_sensors = df[numeric_columns].copy()
    y_true = df["is_anomaly_real"].values

    return df_sensors, df, y_true


# ---------------------------------------------------------------------------
# 2. Utilidades
# ---------------------------------------------------------------------------

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "n_total":        int(len(y_pred)),
        "n_outliers":     int(y_pred.sum()),
        "outlier_ratio_pct": round(100.0 * y_pred.sum() / len(y_pred), 4),
        "precision":      round(precision_score(y_true, y_pred, zero_division=0), 6),
        "recall":         round(recall_score(y_true, y_pred, zero_division=0), 6),
        "f1_score":       round(f1_score(y_true, y_pred, zero_division=0), 6),
    }


def _pred_key(row: dict) -> str:
    """Clave única para PRED_STORE construida a partir de model_type, anomaly_def y
    los hiperparámetros no-NaN del resultado."""
    parts = [row["model_type"], row.get("anomaly_def", "")]
    for col in _HP_COLS:
        v = row.get(col, float("nan"))
        if isinstance(v, float) and np.isnan(v):
            continue
        parts.append(f"{col}={v}")
    return "|".join(parts)


def _base_row(model_type: str, anomaly_def: str = "BROKEN_only") -> dict:
    """Fila vacía con todas las columnas de hiperparámetros a NaN."""
    return {
        "model_type":       model_type,
        "anomaly_def":      anomaly_def,
        # IQR / IQR_Online comparten 'factor' y 'window_size'
        "factor":           np.nan,
        # Z-Score / ZScore_Online comparten 'k_threshold' y 'halflife'
        "k_threshold":      np.nan,
        # K-Means batch / online compartidos
        "n_clusters":       np.nan,
        # K-Means batch
        "dist_percentile":  np.nan,
        # Isolation Forest
        "n_estimators":     np.nan,
        "contamination":    np.nan,
        # K-Means online / ZScore online (EMA halflife)
        "halflife":         np.nan,
        "dist_threshold":   np.nan,
        # HST / IQR_Online comparten 'window_size'
        "n_trees":          np.nan,
        "window_size":      np.nan,
        "height":           np.nan,
        "score_threshold":  np.nan,
        # Tiempos
        "total_time_s":           np.nan,
        "mean_time_per_sample_ms": np.nan,
    }


# ---------------------------------------------------------------------------
# 3. Modelos offline
# ---------------------------------------------------------------------------

def run_iqr(df_sensors: pd.DataFrame, y_true: np.ndarray, anomaly_def: str = "BROKEN_only") -> list[dict]:
    rows = []
    for factor in HYP["iqr"]["factor"]:
        t0 = time.perf_counter()
        anomalies = pd.DataFrame(index=df_sensors.index)
        for col in df_sensors.columns:
            Q1 = df_sensors[col].quantile(0.25)
            Q3 = df_sensors[col].quantile(0.75)
            IQR = Q3 - Q1
            lb, ub = Q1 - factor * IQR, Q3 + factor * IQR
            anomalies[col] = (df_sensors[col] < lb) | (df_sensors[col] > ub)
        y_pred = anomalies.any(axis=1).astype(int).values
        elapsed = time.perf_counter() - t0

        row = _base_row("IQR", anomaly_def)
        row["factor"] = factor
        row["total_time_s"] = round(elapsed, 6)
        row["mean_time_per_sample_ms"] = round(elapsed / len(y_pred) * 1000, 6)
        row.update(_metrics(y_true, y_pred))
        rows.append(row)
        PRED_STORE[_pred_key(row)] = y_pred.astype(np.int8).copy()
        print(f"  IQR factor={factor:4.1f} | outliers={row['n_outliers']:5d} | F1={row['f1_score']:.4f} | t={elapsed:.3f}s")
    return rows


def run_iqr_online(
    df_sensors: pd.DataFrame, y_true: np.ndarray, anomaly_def: str = "BROKEN_only"
) -> list[dict]:
    """
    IQR Online vectorizado con pandas rolling.
    rolling(window, min_periods) calcula Q1/Q3 sobre las últimas `window` obs.
    sin bucles Python — varios órdenes de magnitud más rápido que iterrows.
    """
    rows = []
    for w_size, factor in itertools.product(
        HYP["iqr_online"]["window_size"], HYP["iqr_online"]["factor"]
    ):
        t0 = time.perf_counter()
        min_p = max(4, w_size // 4)
        roll = df_sensors.rolling(window=w_size, min_periods=min_p)
        Q1 = roll.quantile(0.25)
        Q3 = roll.quantile(0.75)
        IQR = Q3 - Q1
        out_of_bounds = (df_sensors < Q1 - factor * IQR) | (df_sensors > Q3 + factor * IQR)
        y_pred = out_of_bounds.any(axis=1).fillna(False).astype(int).values
        elapsed = time.perf_counter() - t0

        row = _base_row("IQR_Online", anomaly_def)
        row["window_size"] = w_size
        row["factor"] = factor
        row["total_time_s"] = round(elapsed, 6)
        row["mean_time_per_sample_ms"] = round(elapsed / len(y_pred) * 1000, 6)
        row.update(_metrics(y_true, y_pred))
        rows.append(row)
        PRED_STORE[_pred_key(row)] = y_pred.astype(np.int8).copy()
        print(
            f"  IQR_Online win={w_size:5d} factor={factor:.1f} "
            f"| outliers={row['n_outliers']:5d} | F1={row['f1_score']:.4f} "
            f"| t={elapsed:.3f}s"
        )
    return rows


def run_zscore(df_sensors: pd.DataFrame, y_true: np.ndarray, anomaly_def: str = "BROKEN_only") -> list[dict]:
    rows = []
    for k_thr in HYP["zscore"]["k_threshold"]:
        t0 = time.perf_counter()
        anomalies = pd.DataFrame(index=df_sensors.index)
        for col in df_sensors.columns:
            std = df_sensors[col].std()
            if std == 0:
                anomalies[col] = False
            else:
                z = np.abs((df_sensors[col] - df_sensors[col].mean()) / std)
                anomalies[col] = z > k_thr
        y_pred = anomalies.any(axis=1).astype(int).values
        elapsed = time.perf_counter() - t0

        row = _base_row("ZScore", anomaly_def)
        row["k_threshold"] = k_thr
        row["total_time_s"] = round(elapsed, 6)
        row["mean_time_per_sample_ms"] = round(elapsed / len(y_pred) * 1000, 6)
        row.update(_metrics(y_true, y_pred))
        rows.append(row)
        PRED_STORE[_pred_key(row)] = y_pred.astype(np.int8).copy()
        print(f"  ZScore k={k_thr:.1f} | outliers={row['n_outliers']:5d} | F1={row['f1_score']:.4f} | t={elapsed:.3f}s")
    return rows


def run_zscore_online(
    df_sensors: pd.DataFrame, y_true: np.ndarray, anomaly_def: str = "BROKEN_only"
) -> list[dict]:
    """
    Z-Score Online vectorizado con pandas ewm (media/std exponencialmente ponderadas).
    pandas ewm(halflife=h) usa la misma convención: alpha = 1 - exp(-1/h).
    Mucho más rápido que la implementación escalar con iterrows.
    """
    rows = []
    for halflife, k_thr in itertools.product(
        HYP["zscore_online"]["halflife"], HYP["zscore_online"]["k_threshold"]
    ):
        t0 = time.perf_counter()
        ewm_obj = df_sensors.ewm(halflife=halflife, min_periods=2)
        ema_mean = ewm_obj.mean()
        ema_std  = ewm_obj.std()
        z = ((df_sensors - ema_mean) / ema_std).abs()
        y_pred = (z > k_thr).any(axis=1).fillna(False).astype(int).values
        elapsed = time.perf_counter() - t0

        row = _base_row("ZScore_Online", anomaly_def)
        row["halflife"]    = halflife
        row["k_threshold"] = k_thr
        row["total_time_s"] = round(elapsed, 6)
        row["mean_time_per_sample_ms"] = round(elapsed / len(y_pred) * 1000, 6)
        row.update(_metrics(y_true, y_pred))
        rows.append(row)
        PRED_STORE[_pred_key(row)] = y_pred.astype(np.int8).copy()
        print(
            f"  ZScore_Online hl={halflife:5d} k={k_thr:.1f} "
            f"| outliers={row['n_outliers']:5d} | F1={row['f1_score']:.4f} "
            f"| t={elapsed:.3f}s"
        )
    return rows


def run_kmeans_batch(
    data_scaled: np.ndarray, y_true: np.ndarray, anomaly_def: str = "BROKEN_only"
) -> list[dict]:
    rows = []
    for n_clusters, dist_pct in itertools.product(
        HYP["kmeans_batch"]["n_clusters"], HYP["kmeans_batch"]["dist_percentile"]
    ):
        t0 = time.perf_counter()
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_ids = km.fit_predict(data_scaled)
        distances = np.linalg.norm(
            data_scaled - km.cluster_centers_[cluster_ids], axis=1
        )
        threshold = np.percentile(distances, dist_pct)
        y_pred = (distances > threshold).astype(int)
        elapsed = time.perf_counter() - t0

        row = _base_row("KMeans_Batch", anomaly_def)
        row["n_clusters"] = n_clusters
        row["dist_percentile"] = dist_pct
        row["total_time_s"] = round(elapsed, 6)
        row["mean_time_per_sample_ms"] = round(elapsed / len(y_pred) * 1000, 6)
        row.update(_metrics(y_true, y_pred))
        rows.append(row)
        PRED_STORE[_pred_key(row)] = y_pred.astype(np.int8).copy()
        print(
            f"  KMeans_Batch k={n_clusters} pct={dist_pct} | outliers={row['n_outliers']:5d} "
            f"| F1={row['f1_score']:.4f} | t={elapsed:.3f}s"
        )
    return rows


def run_iforest(data_scaled: np.ndarray, y_true: np.ndarray, anomaly_def: str = "BROKEN_only") -> list[dict]:
    rows = []
    for n_est, contam in itertools.product(
        HYP["iforest"]["n_estimators"], HYP["iforest"]["contamination"]
    ):
        t0 = time.perf_counter()
        iso = IsolationForest(
            n_estimators=n_est, contamination=contam, random_state=42, n_jobs=-1
        )
        preds = iso.fit_predict(data_scaled)
        y_pred = np.where(preds == -1, 1, 0)
        elapsed = time.perf_counter() - t0

        row = _base_row("IsolationForest", anomaly_def)
        row["n_estimators"] = n_est
        row["contamination"] = contam
        row["total_time_s"] = round(elapsed, 6)
        row["mean_time_per_sample_ms"] = round(elapsed / len(y_pred) * 1000, 6)
        row.update(_metrics(y_true, y_pred))
        rows.append(row)
        PRED_STORE[_pred_key(row)] = y_pred.astype(np.int8).copy()
        print(
            f"  IForest n={n_est} contam={contam:.2f} | outliers={row['n_outliers']:5d} "
            f"| F1={row['f1_score']:.4f} | t={elapsed:.3f}s"
        )
    return rows


# ---------------------------------------------------------------------------
# 4. Modelos online (river)
# ---------------------------------------------------------------------------

def run_kmeans_online(df_sensors: pd.DataFrame, y_true: np.ndarray, anomaly_def: str = "BROKEN_only") -> list[dict]:
    rows = []
    # Pre-convertir a lista de dicts una sola vez (evita el overhead de iterrows en cada combo)
    records = df_sensors.to_dict("records")
    combos = list(itertools.product(
        HYP["kmeans_online"]["n_clusters"],
        HYP["kmeans_online"]["halflife"],
        HYP["kmeans_online"]["dist_threshold"],
    ))
    for n_clusters, halflife, dist_thr in combos:
        scaler = river_prep.StandardScaler()
        km_online = cluster.KMeans(n_clusters=n_clusters, halflife=halflife, seed=42)

        anomalies = []
        sample_times = []

        for x in records:
            t_sample = time.perf_counter()
            scaler.learn_one(x)
            x_scaled = scaler.transform_one(x)

            cluster_id = km_online.predict_one(x_scaled)
            centers = km_online.centers
            is_anom = 0
            if centers and cluster_id in centers:
                centroid = centers[cluster_id]
                dist = math.sqrt(
                    sum((x_scaled[f] - centroid.get(f, 0.0)) ** 2 for f in x_scaled)
                )
                if dist > dist_thr:
                    is_anom = 1
            km_online.learn_one(x_scaled)
            sample_times.append(time.perf_counter() - t_sample)
            anomalies.append(is_anom)

        y_pred = np.array(anomalies)
        total_time = sum(sample_times)
        mean_ms = float(np.mean(sample_times)) * 1000

        row = _base_row("KMeans_Online", anomaly_def)
        row["n_clusters"] = n_clusters
        row["halflife"] = halflife
        row["dist_threshold"] = dist_thr
        row["total_time_s"] = round(total_time, 6)
        row["mean_time_per_sample_ms"] = round(mean_ms, 6)
        row.update(_metrics(y_true, y_pred))
        rows.append(row)
        PRED_STORE[_pred_key(row)] = y_pred.astype(np.int8).copy()
        print(
            f"  KMeans_Online k={n_clusters} hl={halflife} dth={dist_thr} "
            f"| outliers={row['n_outliers']:5d} | F1={row['f1_score']:.4f} "
            f"| t={total_time:.2f}s  mean={mean_ms*1000:.3f}µs/sample"
        )
    return rows


def run_hst(df_sensors: pd.DataFrame, y_true: np.ndarray, anomaly_def: str = "BROKEN_only") -> list[dict]:
    rows = []
    # Pre-convertir a lista de dicts una sola vez
    records = df_sensors.to_dict("records")
    arch_combos = list(itertools.product(
        HYP["hst"]["n_trees"],
        HYP["hst"]["window_size"],
        HYP["hst"]["height"],
    ))
    for n_trees, w_size, height in arch_combos:
        scaler_hst = river_prep.StandardScaler()
        hst = anomaly.HalfSpaceTrees(
            n_trees=n_trees, height=height, window_size=w_size, seed=42
        )

        raw_scores = []
        sample_times = []

        for x in records:
            t_sample = time.perf_counter()
            scaler_hst.learn_one(x)
            x_scaled = scaler_hst.transform_one(x)
            score = hst.score_one(x_scaled)
            hst.learn_one(x_scaled)
            sample_times.append(time.perf_counter() - t_sample)
            raw_scores.append(score)

        total_time = sum(sample_times)
        mean_ms = float(np.mean(sample_times)) * 1000
        raw_scores_arr = np.array(raw_scores)

        for score_thr in HYP["hst"]["score_threshold"]:
            y_pred = (raw_scores_arr > score_thr).astype(int)
            row = _base_row("HalfSpaceTrees", anomaly_def)
            row["n_trees"] = n_trees
            row["window_size"] = w_size
            row["height"] = height
            row["score_threshold"] = score_thr
            row["total_time_s"] = round(total_time, 6)
            row["mean_time_per_sample_ms"] = round(mean_ms, 6)
            row.update(_metrics(y_true, y_pred))
            rows.append(row)
            PRED_STORE[_pred_key(row)] = y_pred.astype(np.int8).copy()

        # Resumen del bloque de arquitectura
        best_f1 = max(r["f1_score"] for r in rows[-len(HYP["hst"]["score_threshold"]):])
        print(
            f"  HST n_trees={n_trees:3d} win={w_size:5d} h={height:2d} "
            f"| best_F1={best_f1:.4f} | t={total_time:.2f}s  mean={mean_ms*1000:.3f}µs/sample"
        )
    return rows


# ---------------------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------------------

def plot_results(
    df_results: pd.DataFrame,
    df_sensors: pd.DataFrame,
    y_true_map: dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    """
    Para cada anomaly_def genera:
      · heatmap_{anomaly_def}.png  — matriz binaria (mejor config/modelo × tiempo)
      · timeline_{anomaly_def}.png — señal de referencia + GT shading + marcadores
                                     de predicción (mejor config por modelo)
    Y guarda predictions_best.csv con las predicciones de la mejor config.
    """
    out_dir.mkdir(exist_ok=True)
    timestamps = df_sensors.index
    ref_col = "sensor_23" if "sensor_23" in df_sensors.columns else df_sensors.columns[0]
    ref_signal = df_sensors[ref_col].values

    # Decimación para display: máximo 3 000 puntos en el eje X
    MAX_PTS = 3_000
    step = max(1, len(timestamps) // MAX_PTS)
    t_idx = np.arange(0, len(timestamps), step)

    pred_df = pd.DataFrame(index=timestamps)  # acumula columnas para el CSV

    for anomaly_def, y_true in y_true_map.items():
        pred_df[f"gt_{anomaly_def}"] = y_true

        subset = df_results[df_results["anomaly_def"] == anomaly_def]
        best_configs = (
            subset.sort_values("f1_score", ascending=False)
            .drop_duplicates(subset="model_type")
        )
        model_types = best_configs["model_type"].tolist()

        # ------------------------------------------------------------------
        # Plot 1: Heatmap — Ground Truth + mejor config por modelo
        # ------------------------------------------------------------------
        n_rows_h = len(model_types) + 1
        matrix = np.zeros((n_rows_h, len(t_idx)), dtype=np.int8)
        matrix[0] = y_true[t_idx]
        row_labels = [f"Ground Truth ({anomaly_def})"]

        for i, mtype in enumerate(model_types, start=1):
            row_meta = best_configs[best_configs["model_type"] == mtype].iloc[0]
            key = _pred_key(row_meta.to_dict())
            if key in PRED_STORE:
                matrix[i] = PRED_STORE[key][t_idx]
            row_labels.append(mtype)

        cmap_bin = matplotlib.colors.ListedColormap(["#d6eaf8", "#c0392b"])
        fig, ax = plt.subplots(figsize=(22, max(3, 0.55 * n_rows_h)))
        ax.imshow(matrix, aspect="auto", cmap=cmap_bin,
                  interpolation="nearest", vmin=0, vmax=1)
        ax.set_yticks(range(n_rows_h))
        ax.set_yticklabels(row_labels, fontsize=9)
        ax.set_xlabel("Tiempo (muestra decimada)", fontsize=9)
        ax.set_title(
            f"Anomalías detectadas — {anomaly_def}  "
            f"(azul = normal · rojo = anomalía, mejor config por modelo)",
            fontsize=10,
        )
        legend_patches = [
            mpatches.Patch(color="#d6eaf8", label="Normal"),
            mpatches.Patch(color="#c0392b", label="Anomalía"),
        ]
        ax.legend(handles=legend_patches, loc="upper right", fontsize=8)
        plt.tight_layout()
        hmap_path = out_dir / f"heatmap_{anomaly_def}.png"
        fig.savefig(hmap_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Plot] {hmap_path.name}")

        # ------------------------------------------------------------------
        # Plot 2: Timelines — señal + GT shading + marcadores de predicción
        # ------------------------------------------------------------------
        ncols = 2
        nrows_grid = math.ceil(len(model_types) / ncols)
        fig2, axes = plt.subplots(
            nrows_grid, ncols,
            figsize=(22, 4 * nrows_grid),
            sharex=True,
        )
        axes_flat = np.array(axes).flatten()

        for i, mtype in enumerate(model_types):
            ax2 = axes_flat[i]
            row_meta = best_configs[best_configs["model_type"] == mtype].iloc[0]
            key = _pred_key(row_meta.to_dict())
            y_pred = PRED_STORE.get(key, np.zeros(len(timestamps), dtype=np.int8))

            # Fondo rojo donde la GT es anomalía
            ax2.fill_between(
                range(len(timestamps)),
                float(ref_signal.min()), float(ref_signal.max()),
                where=y_true.astype(bool),
                color="#fadbd8", alpha=0.55, zorder=1, label="GT anomaly",
            )
            # Señal de referencia
            ax2.plot(ref_signal, color="#2980b9", linewidth=0.35,
                     zorder=2, label=ref_col)
            # Marcadores de predicción del modelo
            anom_idx = np.where(y_pred == 1)[0]
            if len(anom_idx):
                ax2.scatter(
                    anom_idx, ref_signal[anom_idx],
                    color="#e67e22", s=3, zorder=3, label="Pred anomaly",
                )

            # Subtítulo: hiperparámetros de la mejor config + F1
            hp_str = ", ".join(
                f"{c}={row_meta[c]}"
                for c in _HP_COLS
                if not (isinstance(row_meta.get(c, float("nan")), float)
                        and np.isnan(row_meta.get(c, float("nan"))))
            )
            ax2.set_title(
                f"{mtype}\n{hp_str}\n"
                f"F1={row_meta['f1_score']:.4f}  "
                f"outliers={int(row_meta['n_outliers'])}",
                fontsize=7,
            )
            ax2.set_ylabel(ref_col, fontsize=7)
            ax2.tick_params(labelsize=6)
            ax2.legend(fontsize=6, loc="upper right", markerscale=2)

            # Acumular en el CSV de predicciones
            pred_df[f"{mtype}_{anomaly_def}"] = y_pred.astype(np.int8)

        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig2.suptitle(
            f"Señal `{ref_col}` + Anomalías — {anomaly_def}  "
            f"(mejor config por modelo)",
            fontsize=11, y=1.005,
        )
        plt.tight_layout()
        tl_path = out_dir / f"timeline_{anomaly_def}.png"
        fig2.savefig(tl_path, dpi=130, bbox_inches="tight")
        plt.close(fig2)
        print(f"  [Plot] {tl_path.name}")

    # ------------------------------------------------------------------
    # predictions_best.csv: timestamps × (ground truths + best predictions)
    # ------------------------------------------------------------------
    pred_csv = out_dir.parent / "predictions_best.csv"
    pred_df.to_csv(pred_csv)
    print(f"  [CSV]  predictions_best.csv  "
          f"({len(pred_df)} filas × {len(pred_df.columns)} columnas)")


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    csv_path = Path(__file__).parent / "sensor.csv"
    plots_dir = Path(__file__).parent / "plots"
    all_rows: list[dict] = []
    y_true_map: dict[str, np.ndarray] = {}  # acumula ground truth por anomaly_def
    df_sensors_ref: pd.DataFrame | None = None   # mismo para todos los label_defs

    for label_name, label_list in ANOMALY_LABELS.items():
        print(f"\n{'#' * 65}")
        print(f"  Definición de anomalía: {label_name}  {label_list}")
        print(f"{'#' * 65}")

        df_sensors, df_full, y_true = load_data(str(csv_path), anomaly_labels=label_list)
        y_true_map[label_name] = y_true
        if df_sensors_ref is None:
            df_sensors_ref = df_sensors  # guardar referencia para los plots
        print(
            f"  {len(df_sensors)} muestras · {df_sensors.shape[1]} sensores · "
            f"{y_true.sum()} anomalías reales ({100 * y_true.mean():.2f}%)"
        )

        # Escalado offline (compartido por K-Means batch e IForest)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df_sensors.values)

        print("\n" + "=" * 65)
        print("1/8  IQR (batch)")
        print("=" * 65)
        all_rows.extend(run_iqr(df_sensors, y_true, label_name))

        print("\n" + "=" * 65)
        print("2/8  IQR Online (ventana deslizante)")
        print("=" * 65)
        all_rows.extend(run_iqr_online(df_sensors, y_true, label_name))

        print("\n" + "=" * 65)
        print("3/8  Z-Score (batch)")
        print("=" * 65)
        all_rows.extend(run_zscore(df_sensors, y_true, label_name))

        print("\n" + "=" * 65)
        print("4/8  Z-Score Online (EMA)")
        print("=" * 65)
        all_rows.extend(run_zscore_online(df_sensors, y_true, label_name))

        print("\n" + "=" * 65)
        print("5/8  K-Means Batch")
        print("=" * 65)
        all_rows.extend(run_kmeans_batch(data_scaled, y_true, label_name))

        print("\n" + "=" * 65)
        print("6/8  Isolation Forest")
        print("=" * 65)
        all_rows.extend(run_iforest(data_scaled, y_true, label_name))

        print("\n" + "=" * 65)
        print("7/8  K-Means Online (river)")
        print("=" * 65)
        all_rows.extend(run_kmeans_online(df_sensors, y_true, label_name))

        print("\n" + "=" * 65)
        print("8/8  Half-Space Trees (river)")
        print("=" * 65)
        all_rows.extend(run_hst(df_sensors, y_true, label_name))

    # -------------------------------------------------------------------
    # 6. Guardar CSV con orden coherente de columnas
    # -------------------------------------------------------------------
    col_order = [
        "model_type",
        "anomaly_def",
        # Hiperparámetros
        "factor",
        "k_threshold",
        "n_clusters",
        "dist_percentile",
        "n_estimators",
        "contamination",
        "halflife",
        "dist_threshold",
        "n_trees",
        "window_size",
        "height",
        "score_threshold",
        # Estadísticas de detección
        "n_total",
        "n_outliers",
        "outlier_ratio_pct",
        # Métricas de evaluación
        "precision",
        "recall",
        "f1_score",
        # Tiempos
        "total_time_s",
        "mean_time_per_sample_ms",
    ]

    df_results = pd.DataFrame(all_rows)[col_order]
    out_path = Path(__file__).parent / "benchmark_results.csv"
    df_results.to_csv(out_path, index=False)

    print(f"\n{'='*65}")
    print(f"Resultados guardados en: {out_path}")
    print(f"Total de experimentos: {len(df_results)}")

    # -------------------------------------------------------------------
    # Plots y CSV de predicciones por muestra
    # -------------------------------------------------------------------
    print(f"\n{'='*65}")
    print("Generando plots …")
    print(f"{'='*65}")
    plot_results(df_results, df_sensors_ref, y_true_map, plots_dir)

    print("\nTop-10 por F1-Score:")
    top10 = df_results.nlargest(10, "f1_score")[
        ["model_type", "anomaly_def", "n_clusters", "contamination", "n_trees",
         "score_threshold", "n_outliers", "precision", "recall", "f1_score",
         "total_time_s"]
    ]
    print(top10.to_string(index=False))


if __name__ == "__main__":
    main()
