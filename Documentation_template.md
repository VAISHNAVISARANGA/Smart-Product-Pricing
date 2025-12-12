# ML Challenge 2025: Smart Product Pricing — Project Documentation

**Team Name:** 3-layer Perceptron
**Team Members:** Saranga Vaishnavi, Tanisha Ramgopal Saini, Sheetal Singh Chauhan
**Submission Date:** 13-Oct-2025

---

## 1. Executive Summary
We build a multimodal, pipelined pricing solution that combines text and image embeddings, compresses features with PCA, and trains an ensemble of tree-based regressors (LightGBM, XGBoost, RandomForest) with K‑Fold cross-validation. Key outcomes: robust out-of-fold predictions, an ensemble that reduces variance, and an evaluation suite that reports SMAPE / MAE / RMSE in both log- and price-space.

---

## 2. Methodology Overview

### 2.1 Problem Analysis
The task is price regression for product listings using product text and images. EDA showed price distribution is heavy-tailed, so we model in log-space (log1p) to stabilize variance and train models that predict log(price+1). We merge text and image embeddings per sample and use PCA to reduce dimensionality before modeling.

**Key Observations:**
- Price distribution is skewed — log transform improves stability.
- Multimodal embeddings (text + image) improve signal compared to single modality.
- Tree ensembles perform well on engineered numeric features.

### 2.2 Solution Strategy
Approach Type: Hybrid / Ensemble (multimodal feature extraction + PCA + tree ensembles).  
Core Innovation: A practical pipeline that extracts text and image embeddings, aligns samples, reduces dimensionality with PCA, and uses a weighted ensemble of LGB/XGB/RF trained with 5-fold CV to produce out-of-fold (OOF) predictions used for robust evaluation and stacking.

---

## 3. Pipeline & Model Architecture

### 3.1 High-level Flow
1. Extract text embeddings: `src/01_text_embeddings.py`.
2. Extract image embeddings: `src/02_image_embeddings.py`.
3. Merge embeddings by `sample_id`: `src/03_merged_data.py`.
4. Apply PCA to merged features: `src/04_PCA.py` (default n_components=256).
5. Final preprocessing & attach prices: `src/05_Preprocessing_final.py` (produces `final_df.csv`).
6. Train models with 5‑fold CV and create OOF ensemble predictions: `src/06_Model.py` (saves `predicted_prices.csv`).
7. Evaluate predictions & produce metrics/plots: `src/07_Evaluation.py` (saves `evaluation_metrics.csv` and plots).

### 3.2 Model Components

Text Processing Pipeline:
- Preprocessing: tokenization/normalization inside `01_text_embeddings.py` (outputs `text_embeddings`).  
- Model type: pre-trained openai RN50.

Image Processing Pipeline:
- Preprocessing: parallel image downloading, resizing and normalization inside `02_image_embeddings.py` (outputs `image_embeddings`). Images are fetched using a parallel downloader to speed up IO and preprocessing for large image sets.
- Model type: pre-trained openai RN50.

Feature Fusion & Dimensionality Reduction:
- Concatenate text + image embeddings per `sample_id` (`03_merged_data.py`).  
- Use PCA (`04_PCA.py`) to reduce to 256 features by default.

Modeling (training):
- Base learners: LightGBM, XGBoost, RandomForest (trained in log(price+1) space).  
- Training strategy: 5‑fold cross-validation producing OOF predictions for robust validation.  
- Ensemble weighting implemented: 0.5 (LGB) / 0.3 (XGB) / 0.2 (RF).

## 4. Model Performance

### 4.1 Validation Results
SMAPE Score: 42.74(on validation split)
MAE: 14.9 (approx.)
RMSE: 34.39 (approx.)


## 5. Conclusion
We implemented a robust multimodal pricing pipeline that combines text and image embeddings, reduces dimensionality with PCA, and trains a weighted ensemble (LightGBM, XGBoost, RandomForest) using 5‑fold cross-validation. The pipeline produces reliable out-of-fold predictions for model selection and final predicted prices; using a parallel image downloader substantially reduced preprocessing time when working with large image datasets. Future improvements include automated hyperparameter tuning, learned multimodal fusion, and stacking a meta-learner on OOF predictions to further improve final accuracy.

