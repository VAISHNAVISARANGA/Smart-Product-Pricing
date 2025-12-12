import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import importlib.util
from pathlib import Path

# Load modules by path (works even if filenames start with digits)
_base = Path(__file__).resolve().parent
_text_path = _base / "01_text_embeddings.py"
_spec = importlib.util.spec_from_file_location("text_embeddings_module", str(_text_path))
_text_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_text_mod)
text_embeddings = _text_mod.text_embeddings
ids = getattr(_text_mod, "ids", None)

_img_path = _base / "02_image_embeddings.py"
_spec2 = importlib.util.spec_from_file_location("image_embeddings_module", str(_img_path))
_img_mod = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_img_mod)
image_embeddings = _img_mod.image_embeddings

# Assume you have:
# ids                -> list of sample_ids (in df order)
# text_embeddings    -> (N, 512)
# image_embeddings   -> (N, 1024)

# 1. Building two dataframes with sample_id as key
text_df = pd.DataFrame(text_embeddings)
text_df.insert(0, "sample_id", ids)

image_df = pd.DataFrame(image_embeddings)
image_df.insert(0, "sample_id", ids)

# 2. Merging them using sample_id (strict join)

merged = text_df.merge(image_df, on="sample_id", suffixes=("_text", "_img"))

# Droping sample_id temporarily for PCA
sample_ids = merged["sample_id"]
merged_values = merged.drop(columns=["sample_id"]).values

print("Merged shape:", merged_values.shape)
