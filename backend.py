"""
DefectIntel backend — AI engine, data validation, clustering, similarity search,
batch analytics, alert management, and model evaluation.

All public functions are used by app.py. Preserve this module's API when modifying.
"""

from functools import lru_cache
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "defectintel_sample_defects_100.csv"
OUTPUT_PATH = ROOT / "data" / "clustered_defects.csv"
ACKNOWLEDGEMENT_PATH = ROOT / "data" / "alert_acknowledgements.sqlite3"
ALERT_STATUS_PATH = ROOT / "data" / "alert_statuses.sqlite3"
MODEL_NAME = "all-MiniLM-L6-v2"
REQUIRED_COLUMNS = ("defect_description", "batch_id")

# Valid 4-state alert lifecycle
ALERT_STATUSES = ("new", "acknowledged", "under_investigation", "resolved")

PRESET_DATASETS = {
    "Mobile Electronics (100 reports)": DATA_PATH,
    "Automotive Assembly (90 reports)": ROOT / "defectintel_automotive_90.csv",
    "Food Packaging (90 reports)": ROOT / "defectintel_food_packaging_90.csv",
    "Textile & Garments (90 reports)": ROOT / "defectintel_textile_90.csv",
}


# ---------------------------------------------------------------------------
# Model Loading (cached — model is downloaded once and reused)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_model():
    """Load and cache the SentenceTransformer model. Called once per process."""
    return SentenceTransformer(MODEL_NAME)


# ---------------------------------------------------------------------------
# Dataset Validation & Analysis
# ---------------------------------------------------------------------------

def validate_dataset(df):
    """
    Validate uploaded DataFrame. Returns cleaned copy with non-empty
    defect_description and batch_id. Raises ValueError on problems.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The uploaded data must be a CSV table.")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    cleaned = df.copy()
    cleaned["defect_description"] = (
        cleaned["defect_description"].fillna("").astype(str).str.strip()
    )
    cleaned["batch_id"] = cleaned["batch_id"].fillna("").astype(str).str.strip()
    cleaned = cleaned[
        (cleaned["defect_description"] != "") & (cleaned["batch_id"] != "")
    ].copy()

    if cleaned.empty:
        raise ValueError(
            "No usable reports found. Each row needs a defect description and batch ID."
        )

    return cleaned.reset_index(drop=True)


def analyze_dataset(source_df, model=None, distance_threshold=0.5):
    """
    Validate + embed + cluster a DataFrame. Returns (df_with_cluster, embeddings).
    distance_threshold controls grouping sensitivity:
      lower = tighter, more specific groups; higher = broader groups.
    """
    df = validate_dataset(source_df)
    model = model or load_model()
    reports = df["defect_description"].tolist()
    embeddings = np.asarray(
        model.encode(reports, normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float32,
    )

    if len(df) == 1:
        df["ai_cluster"] = 0
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="average",
        )
        df["ai_cluster"] = clustering.fit_predict(embeddings)

    return df, embeddings


def analyze_defects(distance_threshold=0.5):
    """Analyze the bundled default dataset and save clustered output."""
    source_df = pd.read_csv(DATA_PATH)
    df, embeddings = analyze_dataset(source_df, distance_threshold=distance_threshold)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    return df, embeddings


# ---------------------------------------------------------------------------
# Group Profile Building
# ---------------------------------------------------------------------------

def group_profiles(df, embeddings=None):
    """
    Return a list of dicts, one per cluster, with representative description,
    report count, affected batches, and optional category/severity info.
    Representative is chosen as the report closest to the cluster centroid.
    """
    profiles = []
    for cluster, group in df.groupby("ai_cluster", sort=True):
        positions = group.index.to_numpy()

        if embeddings is not None and len(embeddings) == len(df):
            vectors = embeddings[positions]
            centroid = vectors.mean(axis=0, keepdims=True)
            centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-9)
            rep_pos = positions[int(cosine_similarity(centroid_norm, vectors)[0].argmax())]
            representative = df.loc[rep_pos, "defect_description"]
        else:
            representative = group["defect_description"].iloc[0]

        batches = (
            sorted(group["batch_id"].dropna().astype(str).unique().tolist())
            if "batch_id" in group.columns else []
        )
        categories = (
            sorted(group["category"].dropna().astype(str).unique().tolist())
            if "category" in group.columns else []
        )
        severities = (
            sorted(group["severity"].dropna().astype(str).unique().tolist())
            if "severity" in group.columns else []
        )

        profiles.append({
            "cluster": int(cluster),
            "label": str(representative),
            "representative_description": str(representative),
            "report_count": int(len(group)),
            "affected_batches": batches,
            "categories": categories,
            "severities": severities,
        })
    return profiles


# ---------------------------------------------------------------------------
# Cluster Assignment for New Reports (centroid-based)
# ---------------------------------------------------------------------------

def assign_to_cluster(query_embedding, df, embeddings, min_similarity=0.40):
    """
    Assign a new-report embedding to the nearest cluster centroid using cosine
    similarity. If the best similarity is below min_similarity, the report is
    flagged as a 'potential_new_group' rather than forced into an existing cluster.

    Returns (cluster_id_or_None, best_similarity_float, assignment_type_str).
    assignment_type is 'suggested' or 'potential_new_group'.

    Note: Agglomerative Clustering has no native predict(). This centroid-distance
    approach is a heuristic — not guaranteed to produce the same grouping as a full
    re-cluster, but gives a reasonable, transparent approximation.
    """
    clusters = sorted(df["ai_cluster"].unique())
    if not clusters:
        return None, 0.0, "no_clusters"

    qe = np.asarray(query_embedding, dtype=np.float32).flatten()
    qe_norm = qe / (np.linalg.norm(qe) + 1e-9)

    best_cluster = None
    best_sim = -1.0
    for cid in clusters:
        mask = (df["ai_cluster"] == cid).values
        cluster_embs = embeddings[mask]
        centroid = cluster_embs.mean(axis=0)
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-9)
        sim = float(np.dot(qe_norm, centroid_norm))
        if sim > best_sim:
            best_sim = sim
            best_cluster = cid

    if best_sim >= min_similarity:
        return int(best_cluster), float(best_sim), "suggested"
    return None, float(best_sim), "potential_new_group"


# ---------------------------------------------------------------------------
# Similarity Search
# ---------------------------------------------------------------------------

def search_defect(query, df, embeddings, limit=5, model=None):
    """
    Embed a query description and rank existing reports by cosine similarity.
    Returns a list of result dicts. Similarity is a ranking signal only —
    not a classification probability or model accuracy.
    """
    query = str(query).strip()
    if not query:
        raise ValueError("Enter a defect description.")
    if limit < 1:
        raise ValueError("The result limit must be at least 1.")
    if len(df) == 0 or len(embeddings) != len(df):
        raise ValueError("Analyze a valid dataset before searching.")

    model = model or load_model()
    query_embedding = model.encode([query], normalize_embeddings=True)
    scores = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = scores.argsort()[::-1][: min(limit, len(df))]
    results = []

    for index in top_indices:
        row = df.iloc[index]
        results.append({
            "description": row["defect_description"],
            "cluster": int(row["ai_cluster"]),
            "similarity": round(float(scores[index]), 3),
            "batch_id": str(row["batch_id"]),
            "report_id": str(row["report_id"]) if "report_id" in df.columns else str(index + 1),
            "category": str(row["category"]) if "category" in df.columns else "Not provided",
            "severity": str(row["severity"]) if "severity" in df.columns else "Not assessed",
            "data_origin": str(row["data_origin"]) if "data_origin" in df.columns else "unknown",
        })
    return results


# ---------------------------------------------------------------------------
# Batch Analytics & Alert Generation
# ---------------------------------------------------------------------------

def batch_analytics(
    df,
    previous_batch=None,
    current_batch=None,
    min_current_count=5,
    increase_threshold=50,
    high_increase_threshold=100,
):
    """
    Cross-tab defect groups by batch and generate rule-based count-spike alerts.
    Alerts trigger when a group exceeds min_current_count reports in current_batch
    and its count has risen by >= increase_threshold % over previous_batch.

    IMPORTANT: These are report-count warnings, not defect-rate statistics.
    Production volume denominators are not supplied, so rates cannot be computed.
    Batch IDs are treated as labels only; chronological order is not assumed.
    """
    if "batch_id" not in df.columns or "ai_cluster" not in df.columns:
        raise ValueError("Batch analysis requires batch_id and ai_cluster columns.")
    if min_current_count < 1:
        raise ValueError("The minimum report count must be at least 1.")
    if increase_threshold < 0 or high_increase_threshold < 0:
        raise ValueError("Increase thresholds cannot be negative.")

    summary = pd.crosstab(df["batch_id"], df["ai_cluster"])
    alerts = []

    if previous_batch is None or current_batch is None:
        return summary, alerts
    if previous_batch == current_batch:
        raise ValueError("Choose two different batches for comparison.")
    if previous_batch not in summary.index or current_batch not in summary.index:
        raise ValueError("The selected batch is not present in the analyzed dataset.")

    for cluster in summary.columns:
        previous = int(summary.loc[previous_batch, cluster])
        current = int(summary.loc[current_batch, cluster])
        if current < min_current_count:
            continue

        if previous == 0:
            increase = None
            triggered = True
        else:
            increase = round(((current - previous) / previous) * 100, 1)
            triggered = increase >= increase_threshold
        if not triggered:
            continue

        severity = (
            "High"
            if increase is not None and increase >= high_increase_threshold
            else "Medium"
        )

        cluster_reports = df[df["ai_cluster"] == cluster]
        category_sample = None
        desc_sample = ""
        if not cluster_reports.empty:
            if "category" in cluster_reports.columns and not cluster_reports["category"].dropna().empty:
                category_sample = str(cluster_reports["category"].mode().iloc[0])
            if "defect_description" in cluster_reports.columns and not cluster_reports["defect_description"].dropna().empty:
                desc_sample = str(cluster_reports["defect_description"].iloc[0])

        recommendation = generate_investigation_recommendation(
            category=category_sample, text_sample=desc_sample, severity=severity
        )

        alerts.append({
            "cluster": int(cluster),
            "batch": str(current_batch),
            "previous_batch": str(previous_batch),
            "previous_count": previous,
            "current_count": current,
            "increase_percent": increase,
            "severity": severity,
            "alert_id": f"{previous_batch}|{current_batch}|{int(cluster)}",
            "recommendation": recommendation,
        })
    return summary, alerts


def generate_investigation_recommendation(category=None, text_sample="", severity="Medium"):
    """
    Return a domain-aware investigation suggestion based on category/description text.
    These are heuristic starting points, not root-cause diagnoses.
    """
    text = (str(category or "") + " " + str(text_sample or "")).lower()
    if any(k in text for k in ("display", "screen", "pixel", "flicker", "blink")):
        rec = "Inspect display flex cable seating and SMT solder joints on line driver IC. Verify ESD shielding continuity."
    elif any(k in text for k in ("charg", "usb", "port", "power", "battery", "cable")):
        rec = "Audit USB-C connector pin crimping torque and wave soldering thermal profiles. Inspect incoming connector lot."
    elif any(k in text for k in ("camera", "lens", "photo", "image", "blur", "purple")):
        rec = "Check cleanroom air particulate count (ISO Class 5) at optical mounting station. Recalibrate camera alignment fixture."
    elif any(k in text for k in ("audio", "speaker", "sound", "mic", "ear")):
        rec = "Inspect speaker membrane ultrasonic weld pressure. Verify audio codec trace impedance on motherboard sub-assembly."
    elif any(k in text for k in ("paint", "coat", "scratch", "bubble", "finish")):
        rec = "Check paint booth relative humidity and bake oven temperature ramp-up. Inspect spray nozzle pressure and atomization."
    elif any(k in text for k in ("fitment", "gap", "hole", "align", "door", "trim")):
        rec = "Audit CNC fixture zero-point datum alignment and robotic end-effector wear. Verify tolerance on mating stampings."
    elif any(k in text for k in ("seal", "pouch", "leak", "tear", "tray", "fill")):
        rec = "Inspect heat-seal bar temperature calibration and dwell pressure timing. Perform underwater vacuum decay leak test."
    elif any(k in text for k in ("yarn", "fabric", "dye", "seam", "knit", "textile")):
        rec = "Inspect loom tension sensors and yarn guide ceramic eyelets. Verify dye bath pH balance and fixation temperature."
    else:
        rec = "Conduct 100% gate inspection on current batch inventory. Quarantine sample for engineering cross-section analysis."
    if severity == "High":
        rec += " Immediate Quality Alert: Notify line supervisor and initiate 8D root-cause report."
    return rec


# ---------------------------------------------------------------------------
# Model Evaluation
# ---------------------------------------------------------------------------

def clustering_evaluation(df, reference_column="category"):
    """
    Compare AI clusters against reference labels using ARI and NMI.
    These measure clustering agreement — they are NOT classification accuracy.
    Returns None if reference labels are absent.
    """
    if reference_column not in df.columns or "ai_cluster" not in df.columns:
        return None
    evaluation_rows = df[[reference_column, "ai_cluster"]].dropna()
    if len(evaluation_rows) < 2:
        return None
    reference = evaluation_rows[reference_column].astype(str)
    clusters = evaluation_rows["ai_cluster"]
    return {
        "adjusted_rand_index": float(adjusted_rand_score(reference, clusters)),
        "normalized_mutual_info": float(normalized_mutual_info_score(reference, clusters)),
        "reference_column": reference_column,
        "reports_evaluated": len(evaluation_rows),
    }


def compute_silhouette(df, embeddings):
    """
    Compute cosine-distance silhouette score for the discovered clusters.
    Returns None if fewer than 2 clusters or not enough samples.
    Range: -1.0 (poor) to +1.0 (dense, well-separated clusters).
    """
    if "ai_cluster" not in df.columns:
        return None
    from sklearn.metrics import silhouette_score
    labels = df["ai_cluster"].values
    n_clusters = len(set(labels))
    n_samples = len(labels)
    if n_clusters < 2 or n_clusters >= n_samples:
        return None
    try:
        return float(silhouette_score(embeddings, labels, metric="cosine"))
    except Exception:
        return None


def compute_full_evaluation(df, embeddings):
    """
    Aggregate all available evaluation metrics into one dict.
    Used by the Model Evaluation page.
    """
    result = {}

    # ARI + NMI against first available reference label column
    for ref_col in ("category", "product", "data_origin"):
        eval_res = clustering_evaluation(df, reference_column=ref_col)
        if eval_res:
            result["ari_nmi"] = eval_res
            break

    # Silhouette (always computed when mathematically valid)
    result["silhouette"] = compute_silhouette(df, embeddings)

    # Cluster shape stats
    sizes = df["ai_cluster"].value_counts()
    result["n_clusters"] = int(df["ai_cluster"].nunique())
    result["n_samples"] = int(len(df))
    result["min_cluster_size"] = int(sizes.min())
    result["max_cluster_size"] = int(sizes.max())
    result["mean_cluster_size"] = float(sizes.mean())
    return result


# ---------------------------------------------------------------------------
# Legacy Alert Acknowledgement (preserved for backward compatibility)
# ---------------------------------------------------------------------------

def get_acknowledged_alerts():
    """Return set of acknowledged alert IDs (legacy 2-state system)."""
    ACKNOWLEDGEMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(ACKNOWLEDGEMENT_PATH)
    try:
        with connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS alert_acknowledgements (
                alert_id TEXT PRIMARY KEY,
                acknowledged_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""")
            rows = connection.execute("SELECT alert_id FROM alert_acknowledgements").fetchall()
    finally:
        connection.close()
    return {row[0] for row in rows}


def acknowledge_alert(alert_id):
    if not alert_id:
        raise ValueError("An alert ID is required.")
    ACKNOWLEDGEMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(ACKNOWLEDGEMENT_PATH)
    try:
        with connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS alert_acknowledgements (
                alert_id TEXT PRIMARY KEY,
                acknowledged_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""")
            connection.execute(
                "INSERT OR IGNORE INTO alert_acknowledgements (alert_id) VALUES (?)", (alert_id,)
            )
    finally:
        connection.close()


def unacknowledge_alert(alert_id):
    if not alert_id:
        raise ValueError("An alert ID is required.")
    ACKNOWLEDGEMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(ACKNOWLEDGEMENT_PATH)
    try:
        with connection:
            connection.execute(
                "DELETE FROM alert_acknowledgements WHERE alert_id = ?", (alert_id,)
            )
    finally:
        connection.close()


def reset_acknowledgements():
    ACKNOWLEDGEMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(ACKNOWLEDGEMENT_PATH)
    try:
        with connection:
            connection.execute("DELETE FROM alert_acknowledgements")
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# 4-State Alert Status System (New / Acknowledged / Under Investigation / Resolved)
# ---------------------------------------------------------------------------

def _init_alert_status_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alert_statuses (
            alert_id TEXT PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'new',
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            notes TEXT DEFAULT ''
        )
    """)


def get_alert_statuses():
    """
    Return {alert_id: {'status': str, 'notes': str}}.
    Status is one of: new, acknowledged, under_investigation, resolved.
    'acknowledged' = engineer has seen the alert; defect not yet resolved.
    'resolved' = investigation complete and corrective action confirmed.
    """
    ALERT_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(ALERT_STATUS_PATH)
    try:
        with conn:
            _init_alert_status_table(conn)
            rows = conn.execute(
                "SELECT alert_id, status, notes FROM alert_statuses"
            ).fetchall()
    finally:
        conn.close()
    return {row[0]: {"status": row[1], "notes": row[2] or ""} for row in rows}


def set_alert_status(alert_id, status, notes=""):
    """
    Persist an alert status transition. Valid statuses:
    new → acknowledged → under_investigation → resolved
    (any transition is allowed; the UI enforces the workflow).
    """
    if not alert_id:
        raise ValueError("alert_id is required.")
    if status not in ALERT_STATUSES:
        raise ValueError(f"Invalid status '{status}'. Choose from: {ALERT_STATUSES}")
    ALERT_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(ALERT_STATUS_PATH)
    try:
        with conn:
            _init_alert_status_table(conn)
            conn.execute("""
                INSERT INTO alert_statuses (alert_id, status, notes)
                VALUES (?, ?, ?)
                ON CONFLICT(alert_id) DO UPDATE SET
                    status = excluded.status,
                    notes = excluded.notes,
                    updated_at = CURRENT_TIMESTAMP
            """, (alert_id, status, notes or ""))
    finally:
        conn.close()


def reset_alert_statuses():
    """Clear all stored alert statuses."""
    ALERT_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(ALERT_STATUS_PATH)
    try:
        with conn:
            conn.execute("DELETE FROM alert_statuses")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Entry point for quick CLI verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("DefectIntel backend self-check…")
    df, embeddings = analyze_defects()
    summary, alerts = batch_analytics(df)
    print(f"Total reports: {len(df)}")
    print(f"AI defect groups: {df['ai_cluster'].nunique()}")
    eval_res = compute_full_evaluation(df, embeddings)
    print(f"Silhouette score (cosine): {eval_res.get('silhouette')}")
    print("Batch report counts:")
    print(summary.sum(axis=1))
    print("Backend checks passed.")
