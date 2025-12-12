                          ┌──────────────────────────────┐
                          │     Raw Dataset              │
                          │  sample_id | catalog | image │
                          └──────────────────────────────┘
                                         │
                                         ▼
                   ┌──────────────────────────────────────────┐
                   │      01_text_embeddings.py               │
                   │  → Extract CLIP text embeddings          │
                   │  → Output: text_embeddings               │
                   └──────────────────────────────────────────┘
                                         │
                                         ▼
                   ┌──────────────────────────────────────────┐
                   │     02_image_embeddings.py               │
                   │  → Extract CLIP image embeddings         │
                   │  → Output: image_embeddings              │
                   └──────────────────────────────────────────┘
                                         │
                                         ▼
                   ┌──────────────────────────────────────────┐
                   │        03_merged_data.py                 │
                   │  → Merge text + image embeds using       │
                   │    sample_id as key                      │
                   │  → Output: merged_values                 │
                   └──────────────────────────────────────────┘
                                         │
                                         ▼
                   ┌──────────────────────────────────────────┐
                   │            04_PCA.py                     │
                   │  → Reduce dimension to 256 comps         │
                   │  → Output: pca_df                        │
                   └──────────────────────────────────────────┘
                                         │
                                         ▼
             ┌────────────────────────────────────────────────────────┐
             │                 05_Preprocessing_final.py              │
             │  → Join PCA output with price using sample_id          │
             │  → Apply log transform on price                        │
             │  → Handle missing values, scaling, cleaning            │
             │  → Output: final_df.csv                                │
             └────────────────────────────────────────────────────────┘
                                         │
                                         ▼
             ┌────────────────────────────────────────────────────────┐
             │                    06_Model.py                         │
             │  → 5-Fold Cross-Validation                             │
             │  → Train Ensemble (LightGBM + XGBoost + RF)            │
             │  → Generate OOF predictions                            │
             │  → Output: predicted_prices.csv                        │
             └────────────────────────────────────────────────────────┘
                                         │
                                         ▼
             ┌──────────────────────────────────────────────────────────┐
             │                     07_Evaluation.py                     │
             │  → Compute SMAPE, MAE, RMSE                              │
             │  → Plot error curves                                     │
             │  → Output: evaluation_metrics.csv + evaluation_plots/    │
             └──────────────────────────────────────────────────────────┘
                                         │
                                         ▼
                          ┌────────────────────────────────┐
                          │        Final Results           │
                          │  → SMAPE Score                 │
                          │  → Predicted CSV               │
                          │  → Evaluation Plots            │
                          └────────────────────────────────┘
