import importlib.util
from pathlib import Path
import numpy as np

# Load `final_df` from `04_PCA.py` (handles digit-starting filename)
_base = Path(__file__).resolve().parent
_mod_path = _base / "04_PCA.py"
_spec = importlib.util.spec_from_file_location("pca_module", str(_mod_path))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
final_df = getattr(_mod, "final_df", getattr(_mod, "pca_df", None))

#  Dropping rows where price missing
final_df = final_df.dropna(subset=['price'])

# Filling PCA missing values 
final_df = final_df.fillna(final_df.median())

#  Log transforming the price column
final_df['price_log'] = np.log1p(final_df['price'])

final_df.to_csv("final_df.csv", index=False)
print("final_df saved as final_df.csv")