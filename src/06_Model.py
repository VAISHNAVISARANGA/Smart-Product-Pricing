import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import importlib.util
from pathlib import Path
from sklearn.metrics import mean_squared_error

# Loading `final_df` from `05_Preprocessing_final.py`
_base = Path(__file__).resolve().parent
_mod_path = _base / "05_Preprocessing_final.py"
_spec = importlib.util.spec_from_file_location("preproc_module", str(_mod_path))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
final_df = getattr(_mod, "final_df", None)

X = final_df.drop(['sample_id', 'price', 'price_log'], axis=1)
y = final_df['price_log'].values

# Preparing out-of-fold prediction arrays
n_samples = X.shape[0]
oof_lgb = np.zeros(n_samples)
oof_xgb = np.zeros(n_samples)
oof_rf = np.zeros(n_samples)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold = 0
for train_idx, val_idx in kf.split(X):
    fold += 1
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    lgb_model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=-1
    )

    xgb_model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=10,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method='hist',
        n_jobs=-1
    )

    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        n_jobs=-1
    )

    lgb_model.fit(X_tr, y_tr)
    xgb_model.fit(X_tr, y_tr)
    rf_model.fit(X_tr, y_tr)

    oof_lgb[val_idx] = lgb_model.predict(X_val)
    oof_xgb[val_idx] = xgb_model.predict(X_val)
    oof_rf[val_idx] = rf_model.predict(X_val)

    #  quick fold log
    fold_ens = 0.5 * oof_lgb[val_idx] + 0.3 * oof_xgb[val_idx] + 0.2 * oof_rf[val_idx]
    print(f"Fold {fold} RMSE (log space):", np.sqrt(mean_squared_error(y_val, fold_ens)))

# Ensemble OOF predictions (log space)
oof_ensemble = 0.5 * oof_lgb + 0.3 * oof_xgb + 0.2 * oof_rf

# Converting back to price space
predicted_price = np.expm1(oof_ensemble)

# Saving predicted prices with sample_id
out_df = pd.DataFrame({
    'sample_id': final_df['sample_id'],
    'predicted_price': predicted_price
})
out_df.to_csv('predicted_prices.csv', index=False)
print('Saved predicted prices to predicted_prices.csv')

# Expose predictions in the module namespace
predicted_prices = predicted_price
oof_predictions = oof_ensemble
