import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

df = pd.read_csv("data/clustered_defects.csv")

print("Total reports:", len(df))
print("AI clusters:", df["ai_cluster"].nunique())

print("\nCluster distribution:")
print(pd.crosstab(df["ai_cluster"], df["category"]))

ari = adjusted_rand_score(df["category"], df["ai_cluster"])
nmi = normalized_mutual_info_score(df["category"], df["ai_cluster"])

print("\nAdjusted Rand Index:", round(ari, 3))
print("Normalized Mutual Information:", round(nmi, 3))