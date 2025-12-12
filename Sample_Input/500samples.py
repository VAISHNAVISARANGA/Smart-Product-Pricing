from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
df = pd.read_csv("train.csv")

# create simple metadata features (fast to compute)
df['text_len'] = df['Catalog Content'].fillna("").apply(len)
df['has_image'] = df['Image Link'].notna().astype(int)

meta = df[['price']].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(meta)

kmeans = KMeans(n_clusters=500, random_state=42, n_init='auto')
labels = kmeans.fit_predict(X_scaled)

# pick one sample from each cluster
sampled_idx = []
for c in range(500):
    idx = np.where(labels == c)[0]
    if len(idx) > 0:
        sampled_idx.append(np.random.choice(idx))

sampled_df = df.iloc[sampled_idx]
sampled_df.to_csv("strategic_kmeans_500.csv", index=False)

print("Saved strategic_kmeans_500.csv with shape:", sampled_df.shape)
