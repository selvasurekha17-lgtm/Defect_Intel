import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

df = pd.read_csv("data/defectintel_sample_defects_100.csv")

print("Available columns:", df.columns.tolist())

possible_columns = [
    "defect_description",
    "description",
    "defect_details",
    "complaint",
    "issue_description"
]

text_column = None

for column in possible_columns:
    if column in df.columns:
        text_column = column
        break

if text_column is None:
    print("ERROR: No defect description column found.")
    print("Please share the available column names.")
    exit()

df = df.dropna(subset=[text_column]).copy()
df[text_column] = df[text_column].astype(str)

reports = df[text_column].tolist()

if len(reports) < 2:
    print("ERROR: At least 2 defect reports are required.")
    exit()

print("Total reports:", len(reports))

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(reports, normalize_embeddings=True)

clustering = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=0.5,
    metric="cosine",
    linkage="average"
)

df["ai_cluster"] = clustering.fit_predict(embeddings)

print("\nDefect groups:")
print(df["ai_cluster"].value_counts().sort_index())

df.to_csv("data/clustered_defects.csv", index=False)

print("\nSaved: data/clustered_defects.csv")