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
MODEL_NAME = "all-MiniLM-L6-v2"
REQUIRED_COLUMNS = ("defect_description", "batch_id")

PRESET_DATASETS = {
    "Mobile Electronics (100 reports)": DATA_PATH,
    "Automotive Assembly (90 reports)": ROOT / "defectintel_automotive_90.csv",
    "Food Packaging (90 reports)": ROOT / "defectintel_food_packaging_90.csv",
    "Textile & Garments (90 reports)": ROOT / "defectintel_textile_90.csv",
}


@lru_cache(maxsize=1)
def load_model():
    return SentenceTransformer(MODEL_NAME)


def validate_dataset(df):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The uploaded data must be a CSV table.")

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
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
    df = validate_dataset(source_df)
    model = model or load_model()
    reports = df["defect_description"].tolist()
    embeddings = np.asarray(
        model.encode(reports, normalize_embeddings=True),
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
    source_df = pd.read_csv(DATA_PATH)
    df, embeddings = analyze_dataset(source_df, distance_threshold=distance_threshold)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    return df, embeddings


def group_profiles(df, embeddings=None):
    profiles = []
    for cluster, group in df.groupby("ai_cluster", sort=True):
        positions = group.index.to_numpy()
        if embeddings is not None and len(embeddings) == len(df):
            vectors = embeddings[positions]
            centroid = vectors.mean(axis=0, keepdims=True)
            representative_position = positions[
                int(cosine_similarity(centroid, vectors)[0].argmax())
            ]
            representative = df.loc[representative_position, "defect_description"]
        else:
            representative = group["defect_description"].iloc[0]

        profiles.append({
            "cluster": int(cluster),
            "label": str(representative),
            "representative_description": str(representative),
            "report_count": int(len(group)),
        })
    return profiles


def clustering_evaluation(df, reference_column="category"):
    if reference_column not in df.columns or "ai_cluster" not in df.columns:
        return None
    evaluation_rows = df[[reference_column, "ai_cluster"]].dropna()
    if len(evaluation_rows) < 2:
        return None
    reference = evaluation_rows[reference_column].astype(str)
    clusters = evaluation_rows["ai_cluster"]
    return {
        "adjusted_rand_index": float(adjusted_rand_score(reference, clusters)),
        "normalized_mutual_info": float(
            normalized_mutual_info_score(reference, clusters)
        ),
        "reference_column": reference_column,
        "reports_evaluated": len(evaluation_rows),
    }


def search_defect(query, df, embeddings, limit=5, model=None):
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
            "severity": str(row["severity"]) if "severity" in df.columns else "Not provided",
        })
    return results


def batch_analytics(
    df,
    previous_batch=None,
    current_batch=None,
    min_current_count=5,
    increase_threshold=50,
    high_increase_threshold=100,
):
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
            category=category_sample,
            text_sample=desc_sample,
            severity=severity,
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


def get_acknowledged_alerts():
    ACKNOWLEDGEMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(ACKNOWLEDGEMENT_PATH)
    try:
        with connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS alert_acknowledgements (
                    alert_id TEXT PRIMARY KEY,
                    acknowledged_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )"""
            )
            rows = connection.execute(
                "SELECT alert_id FROM alert_acknowledgements"
            ).fetchall()
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
            connection.execute(
                """CREATE TABLE IF NOT EXISTS alert_acknowledgements (
                    alert_id TEXT PRIMARY KEY,
                    acknowledged_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )"""
            )
            connection.execute(
                "INSERT OR IGNORE INTO alert_acknowledgements (alert_id) VALUES (?)",
                (alert_id,),
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
                "DELETE FROM alert_acknowledgements WHERE alert_id = ?",
                (alert_id,),
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


if __name__ == "__main__":
    print("Starting DefectIntel backend...")
    df, embeddings = analyze_defects()
    summary, alerts = batch_analytics(df)
    print(f"\nTotal reports: {len(df)}")
    print(f"AI defect groups: {df['ai_cluster'].nunique()}")
    print("\nBatch report counts:")
    print(summary.sum(axis=1))
    print("\nSelect an explicit batch comparison in the dashboard to calculate alerts.")
    print("\nExample similarity search:")
    for result in search_defect("Phone screen keeps flashing", df, embeddings):
        print(result)
