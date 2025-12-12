import pandas as pd
import torch
import open_clip
import numpy as np

# ----------------------------
# 1. Load data
# ----------------------------
df = pd.read_csv("train.csv", engine='python', on_bad_lines='skip')   # adjust filename
df = df[['sample_id', 'catalog_content']].fillna("")

texts = df['catalog_content'].tolist()
ids = df['sample_id'].tolist()

# ----------------------------
# 2. Load CLIP model on GPU
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess, *_ = open_clip.create_model_and_transforms("RN50", pretrained='openai', device=device)
model.eval()

# ----------------------------
# 3. GPU-optimized text embedding extractor (BATCh)
# ----------------------------
def extract_clip_text_embeddings(text_list, batch_size=64):
    all_embeddings = []
    n = len(text_list)

    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch_texts = text_list[i:i+batch_size]

            tokens = open_clip.tokenize(batch_texts).to(device)
            feats = model.encode_text(tokens)

            # Normalize for better model stability
            feats = feats / feats.norm(dim=-1, keepdim=True)

            all_embeddings.append(feats.cpu().numpy())

    return np.vstack(all_embeddings)

# ----------------------------
# 4. Run embedding extraction
# ----------------------------
text_embeddings = extract_clip_text_embeddings(texts)

print("Text Embeddings Shape:", text_embeddings.shape)

