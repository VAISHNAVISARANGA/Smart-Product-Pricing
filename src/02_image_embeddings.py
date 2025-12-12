import pandas as pd
import torch
import open_clip # Changed from 'import clip'
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# ----------------------------
# 1. Load dataset
# ----------------------------
df = pd.read_csv("train.csv")  # adjust file
df = df[['sample_id', 'image_link']].fillna("")

image_urls = df['image_link'].tolist()
ids = df['sample_id'].tolist()

# ----------------------------
# 2. Load CLIP model on GPU
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess, *_ = open_clip.create_model_and_transforms("RN50", pretrained='openai', device=device) # Changed to use open_clip
model.eval()

# ----------------------------
# 3. Parallel image downloader
# ----------------------------
def load_image_safe(url):
    try:
        response = requests.get(url, timeout=5)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img
    except:
        return Image.new("RGB", (224, 224), (0,0,0))

def load_images(urls, max_workers=32):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        images = list(executor.map(load_image_safe, urls))
    return images

# ----------------------------
# 4. Batched embedding extraction
# ----------------------------
def extract_clip_image_embeddings(urls, batch_size=64):
    all_embeddings = []
    n = len(urls)

    # Step 1: download all images in parallel
    print("Downloading images...")
    images = load_images(urls)

    print("Extracting embeddings on GPU...")
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch_images = images[i:i+batch_size]
            batch_tensor = torch.stack([preprocess(img) for img in batch_images]).to(device)
            feats = model.encode_image(batch_tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)  # normalize
            all_embeddings.append(feats.cpu().numpy())

    return np.vstack(all_embeddings)

# ----------------------------
# 5. Run extraction
# ----------------------------
image_embeddings = extract_clip_image_embeddings(image_urls)

print("Image Embeddings Shape:", image_embeddings.shape)

