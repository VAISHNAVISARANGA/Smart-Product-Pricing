from sklearn.decomposition import PCA
import importlib.util
from pathlib import Path
import pandas as pd

# Load `merged_values` and `sample_ids` from `04_merged_data.py` (handles digit-starting filename)
_base = Path(__file__).resolve().parent
_mod_path = _base / "04_merged_data.py"
_spec = importlib.util.spec_from_file_location("merged_data_module", str(_mod_path))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
merged_values = getattr(_mod, "merged_values", getattr(_mod, "mergedvalues", None))
sample_ids = getattr(_mod, "sample_ids", getattr(_mod, "ids", None))

pca_dim = 256
pca = PCA(n_components=pca_dim, random_state=42)
pca.fit(merged_values)

#  pca transform the data

final_features = pca.transform(merged_values)

# Reattach sample_id
pca_df = pd.DataFrame(final_features, columns=[f"feat_{i}" for i in range(pca_dim)])
pca_df.insert(0, "sample_id", sample_ids)

print(pca_df.head())
print("Final shape:", pca_df.shape) 

import pandas as pd

# pca_df  -> Dataset with PCA features (with sample_id)
# price_df -> original df containing price column

# Loading the original sampled dataframe again to get the price column
price_df = pd.read_csv('train.csv')

# Keeping only sample_id and price from the original data
price_only = price_df[['sample_id', 'price']]

# Merging based on sample_id
final_df = pca_df.merge(price_only, on='sample_id', how='inner')

print("Final shape:", final_df.shape)
print(final_df.head())
