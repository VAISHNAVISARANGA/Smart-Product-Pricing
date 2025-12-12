# Amazon Price Predictor

This repository contains a multimodal pricing pipeline built for the Amazon ML Hackathon 2025. The project extracts text and image embeddings, fuses them, reduces dimensionality with PCA, and trains an ensemble of tree-based regressors using cross-validation to predict product prices.

## Project highlights
- Built for: **Amazon ML Hackathon 2025**
- Multimodal: text + image embeddings
- Dimensionality reduction: PCA
- Models: LightGBM, XGBoost, RandomForest (5‑fold CV + weighted ensemble)
- Evaluation: SMAPE / MAE / RMSE (log-space & price-space)

## Quick start

1. Create and activate a Python environment:

```bash
python -m venv .venv
# mac / linux
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the pipeline (recommended order):

```bash
python src/01_text_embeddings.py
python src/02_image_embeddings.py
python src/03_merged_data.py
python src/04_PCA.py
python src/05_Preprocessing_final.py
python src/06_Model.py
python src/07_Evaluation.py
```

Artifacts produced:
- `final_df.csv` — preprocessed dataset with PCA features and price
- `predicted_prices.csv` — predicted prices (price-space)
- `evaluation_metrics.csv` — MAE / RMSE / SMAPE
- plots: `predicted_actual_distribution.png`, `actual_vs_predicted_scatter.png`

## File overview
- `src/01_text_embeddings.py` — extract text embeddings
- `src/02_image_embeddings.py` — download images (parallel) and extract image embeddings
- `src/03_merged_data.py` — merge embeddings
- `src/04_PCA.py` — apply PCA to features
- `src/05_Preprocessing_final.py` — attach price and finalize dataset
- `src/06_Model.py` — train models (5-fold CV) and save predictions
- `src/07_Evaluation.py` — compute metrics and generate plots

## Notes
- Models are trained in `log1p(price)` space; predictions are inverse-transformed with `expm1` for final price outputs.
- Use OOF (out-of-fold) predictions for reliable CV-based evaluation and stacking.


