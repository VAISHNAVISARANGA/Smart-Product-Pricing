import numpy as np
import pandas as pd
from pathlib import Path
import importlib.util
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False


def smape(y_true, y_pred):
    # Safe SMAPE implementation
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    # avoid division by zero
    denom = np.where(denom == 0, 1e-8, denom)
    return np.mean(np.abs(y_true - y_pred) / denom) * 100.0


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def load_predictions():
    """Try to load predicted prices (price-space) and/or oof predictions.
    Priority: module variables in src/06_Model.py, then predicted_prices.csv
    Returns a DataFrame with columns ['sample_id','predicted_price'] when possible.
    """
    base = Path(__file__).resolve().parent
    mod_path = base / "06_Model.py"
    df = None
    if mod_path.exists():
        try:
            spec = importlib.util.spec_from_file_location("model_module", str(mod_path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "predicted_prices"):
                # if module exposes array, try align by index
                preds = np.asarray(mod.predicted_prices)
                df = pd.DataFrame({"predicted_price": preds})
            elif (base.parent / "predicted_prices.csv").exists():
                df = pd.read_csv(base.parent / "predicted_prices.csv")
        except Exception:
            pass

    # fallback to CSV in repo root
    if df is None:
        csv_path = base.parent / "predicted_prices.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)

    return df


def load_truth():
    """Load ground-truth `final_df` from preprocessing module or CSV.
    Returns a DataFrame containing at least ['sample_id','price','price_log']
    """
    base = Path(__file__).resolve().parent
    mod_path = base / "05_Preprocessing_final.py"
    df = None
    if mod_path.exists():
        try:
            spec = importlib.util.spec_from_file_location("preproc_module", str(mod_path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "final_df"):
                df = mod.final_df.copy()
        except Exception:
            pass

    # fallback to CSV
    if df is None:
        csv_path = base.parent / "final_df.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)

    return df


def evaluate_and_plot(save_dir=None):
    save_dir = Path(save_dir) if save_dir is not None else Path(__file__).resolve().parent.parent
    save_dir.mkdir(parents=True, exist_ok=True)

    preds_df = load_predictions()
    truth_df = load_truth()

    if truth_df is None:
        raise FileNotFoundError("Could not locate ground-truth `final_df`. Run preprocessing first.")

    if preds_df is None:
        raise FileNotFoundError("Could not locate predictions. Run model script to produce predictions.")

    # If predictions include sample_id, merge; otherwise align by index
    if "sample_id" in preds_df.columns:
        merged = truth_df.merge(preds_df, on="sample_id", how="inner")
        if "predicted_price" not in merged.columns:
            # maybe CSV used 'predicted' column name
            preds_col = [c for c in merged.columns if c.lower().startswith("predicted")]
            if preds_col:
                merged = merged.rename(columns={preds_col[0]: "predicted_price"})
    else:
        # align by index
        merged = truth_df.copy().reset_index(drop=True)
        merged["predicted_price"] = preds_df["predicted_price"].values[: len(merged)]

    # Ensure price columns present
    if "price" not in merged.columns and "price_log" in merged.columns:
        merged["price"] = np.expm1(merged["price_log"].values)

    y_true = merged["price"].values
    y_pred = merged["predicted_price"].values

    # Metrics in price space
    mae_val = mean_absolute_error(y_true, y_pred)
    rmse_val = rmse(y_true, y_pred)
    smape_val = smape(y_true, y_pred)


    metrics = {
        "MAE_price": mae_val,
        "RMSE_price": rmse_val,
        "SMAPE_price": smape_val,
    
        "n_samples": len(y_true)
    }

    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(save_dir / "evaluation_metrics.csv", index=False)
    print("Saved evaluation metrics to:", save_dir / "evaluation_metrics.csv")
    print(metrics)

    # Distribution plot
    plt.figure(figsize=(8, 5))
    if _HAS_SEABORN:
        sns.kdeplot(np.log1p(y_true), label="Actual (log)", fill=True)
        sns.kdeplot(np.log1p(y_pred), label="Predicted (log)", fill=True)
        plt.xlabel("log(price+1)")
    else:
        plt.hist(np.log1p(y_true), bins=50, alpha=0.6, label="Actual (log)")
        plt.hist(np.log1p(y_pred), bins=50, alpha=0.4, label="Predicted (log)")
        plt.xlabel("log(price+1)")
    plt.legend()
    plt.title("Predicted vs Actual Distribution (log price)")
    plt.tight_layout()
    plt.savefig(save_dir / "predicted_actual_distribution.png")
    plt.close()

    # Scatter plot (true vs predicted)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, s=10)
    mx = max(y_true.max(), y_pred.max())
    plt.plot([0, mx], [0, mx], color="red", linestyle="--")
    plt.xlabel("Actual price")
    plt.ylabel("Predicted price")
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(save_dir / "actual_vs_predicted_scatter.png")
    plt.close()

    print("Saved plots to:", save_dir)

    return metrics, merged
