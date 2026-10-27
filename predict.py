import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data/clustered_defects.csv")

df = df.dropna(subset=["defect_description"]).copy()

model = SentenceTransformer("all-MiniLM-L6-v2")

reports = df["defect_description"].astype(str).tolist()
embeddings = model.encode(reports, normalize_embeddings=True)

print("\nDEFECTINTEL — AI DEFECT SEARCH")

while True:
    new_defect = input("\nEnter defect (or type exit): ").strip()

    if new_defect.lower() == "exit":
        break

    if not new_defect:
        print("Please enter a defect description.")
        continue

    new_embedding = model.encode(
        [new_defect],
        normalize_embeddings=True
    )

    scores = cosine_similarity(new_embedding, embeddings)[0]

    top_indices = scores.argsort()[-3:][::-1]

    print("\nMost similar existing defects:")

    for index in top_indices:
        row = df.iloc[index]

        print("\nDescription:", row["defect_description"])
        print("AI Cluster:", row["ai_cluster"])
        print("Similarity:", round(float(scores[index]), 3))