# qe_expanded_daily_to_quarterly.py
# End-to-end QE prediction using daily FRED data -> quarterly features -> RF vs XGBoost

import os
from datetime import date
import numpy as np
import pandas as pd
from fredapi import Fred

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

import matplotlib.pyplot as plt

# =========================
# Config
# =========================
FRED_API_KEY = "4ff94616949b02a01853e8effd6b5999"
START_DATE = pd.Timestamp("2002-12-18")
DATA_DIR = os.path.join("..", "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

SERIES = [
    # Policy and liquidity
    "WSHOSHO",   # Fed securities
    "M2SL",      # M2
    "FEDFUNDS",  # Effective Fed Funds Rate
    # Rates and spreads
    "GS10",      # 10Y Tsy
    "BAA10Y",    # BAA minus 10Y spread
    # Market stress
    "VIXCLS",    # VIX
    # Macro
    "UNRATE",    # Unemployment
    "INDPRO",    # Industrial production
    "CPIAUCSL",  # CPI
    "T5YIFR",    # 5y5y inflation expectation
]

# =========================
# Helpers
# =========================
def fetch_series(fred: Fred, series_id: str, start: pd.Timestamp) -> pd.DataFrame:
    """Fetch a series from FRED and return a DataFrame with daily index and column named as series_id.
    We forward fill onto a daily calendar later; here we just return the native frequency Series."""
    s = fred.get_series(series_id, observation_start=start)
    if s is None or len(s) == 0:
        raise ValueError(f"Empty series: {series_id}")
    df = s.to_frame(name=series_id)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df

def forward_fill_to_daily(dfs: list[pd.DataFrame], start: pd.Timestamp) -> pd.DataFrame:
    """Merge on date, build daily calendar, forward fill then back fill."""
    # Determine max end date from all series
    max_end = max(df.index.max() for df in dfs)
    daily_index = pd.date_range(start=start, end=max_end, freq="D")
    merged = pd.DataFrame(index=daily_index)
    for df in dfs:
        merged = merged.join(df, how="left")
    merged.sort_index(inplace=True)
    merged = merged.ffill().bfill()
    return merged

def within_quarter_growth(x: pd.Series) -> float:
    """Pct change from first to last within the group, in percent."""
    first = x.iloc[0]
    last = x.iloc[-1]
    if pd.isna(first) or pd.isna(last) or first == 0:
        return np.nan
    return 100.0 * (last / first - 1.0)

def realized_vol(x: pd.Series) -> float:
    """Standard deviation of a daily return series within the group."""
    return float(np.nanstd(x, ddof=1))

def add_quarter_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["quarter"] = out.index.to_period("Q").to_timestamp(how="end")
    return out

# =========================
# 1) Fetch and regularize to daily
# =========================
fred = Fred(api_key=FRED_API_KEY)
frames = [fetch_series(fred, sid, START_DATE) for sid in SERIES]
daily_all = forward_fill_to_daily(frames, START_DATE)

# Save daily merged
daily_path = os.path.join(DATA_DIR, "expanded_daily_all.csv")
daily_all.reset_index().rename(columns={"index": "date"}).to_csv(daily_path, index=False)
print(f"Saved daily merged to: {daily_path}")

# =========================
# 2) Build quarterly features from daily
# =========================
df = daily_all.copy()

# Daily returns for realized volatility (percent returns)
df["VIX_ret"]    = 100.0 * (df["VIXCLS"] / df["VIXCLS"].shift(1) - 1.0)
df["CPI_ret"]    = 100.0 * (df["CPIAUCSL"] / df["CPIAUCSL"].shift(1) - 1.0)
df["INDPRO_ret"] = 100.0 * (df["INDPRO"] / df["INDPRO"].shift(1) - 1.0)
df["M2_ret"]     = 100.0 * (df["M2SL"] / df["M2SL"].shift(1) - 1.0)

# Add quarter label
df_q = add_quarter_column(df)

# Aggregate
agg_dict_levels_last = {
    "WSHOSHO": "last",
    "M2SL": "last",
    "FEDFUNDS": "last",
    "GS10": "last",
    "BAA10Y": "last",
    "VIXCLS": "last",
    "UNRATE": "last",
    "INDPRO": "last",
    "CPIAUCSL": "last",
    "T5YIFR": "last",
}
# We will compute growth and realized vol separately to preserve logic
levels = (
    df_q
    .groupby("quarter")
    .agg(agg_dict_levels_last)
    .rename(columns={
        "WSHOSHO": "Fed_Securities",
        "M2SL": "M2",
        "FEDFUNDS": "FedFunds",
    })
)

# Growth within quarter from first to last, percent
growth_df = (
    df_q
    .groupby("quarter")
    .agg(
        Fed_Growth=("WSHOSHO", within_quarter_growth),
        M2_Growth=("M2SL", within_quarter_growth),
        INDPRO_Growth=("INDPRO", within_quarter_growth),
        CPI_Growth=("CPIAUCSL", within_quarter_growth),
    )
)

# Realized vol within quarter
rv_df = (
    df_q
    .groupby("quarter")
    .agg(
        VIX_realized_vol=("VIX_ret", realized_vol),
        M2_realized_vol=("M2_ret", realized_vol),
        INDPRO_realized_vol=("INDPRO_ret", realized_vol),
    )
)

# Extremes and rate slope
ext_df = (
    df_q
    .groupby("quarter")
    .agg(
        VIX_max=("VIXCLS", "max"),
        VIX_min=("VIXCLS", "min"),
        Rate_Slope=("GS10", "last"),
        FedFunds_last=("FEDFUNDS", "last"),
    )
)
ext_df["Rate_Slope"] = ext_df["Rate_Slope"] - ext_df["FedFunds_last"]
ext_df.drop(columns=["FedFunds_last"], inplace=True)

# Merge all quarterly pieces
quarterly = levels.join(growth_df, how="inner").join(rv_df, how="inner").join(ext_df, how="inner")
quarterly = quarterly.sort_index()

# Lags
for col in ["Fed_Growth", "M2_Growth", "INDPRO_Growth", "Rate_Slope"]:
    quarterly[f"{col}_L1"] = quarterly[col].shift(1)

# Flags
quarterly["VIX_High"] = (quarterly["VIXCLS"] > 30).astype("int64")
quarterly["High_Spread"] = (quarterly["BAA10Y"] > 2).astype("int64")
quarterly["High_InflExp"] = (quarterly["T5YIFR"] > 2.5).astype("int64")

# Target: QE in next quarter if Fed assets increase by more than 100
quarterly["QE_Next_Quarter"] = (
    (quarterly["Fed_Securities"].shift(-1) - quarterly["Fed_Securities"]) > 100
).astype("int64")

# Drop last row with NaN target if present
quarterly = quarterly.dropna(subset=["QE_Next_Quarter"])

# Save quarterly features
q_csv = os.path.join(DATA_DIR, "expanded_quarterly_features_py.csv")
q_rds_like = os.path.join(DATA_DIR, "expanded_qe_dataset_daily_to_quarterly_py.parquet")
quarterly.reset_index().rename(columns={"quarter": "quarter_end"}).to_csv(q_csv, index=False)
quarterly.to_parquet(q_rds_like)
print(f"Saved quarterly features to: {q_csv}")
print(f"Saved dataset parquet to: {q_rds_like}")

# =========================
# 3) Train RF and XGB with time split
# =========================
# Prepare ML dataset
data_ml = quarterly.copy()
data_ml = data_ml.dropna()  # safety

feature_cols = [c for c in data_ml.columns if c not in ["QE_Next_Quarter"]]
target_col = "QE_Next_Quarter"

# Time split: first 70 percent train, last 30 percent test
n = len(data_ml)
split_idx = int(np.floor(0.7 * n))
train_df = data_ml.iloc[:split_idx].copy()
test_df = data_ml.iloc[split_idx:].copy()

X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].astype(int).values
X_test = test_df[feature_cols].copy()
y_test = test_df[target_col].astype(int).values

# Optional: class weighting for RF
class_weight = "balanced"

# Random Forest
rf = RandomForestClassifier(
    n_estimators=500,
    max_features=max(2, int(np.floor(np.sqrt(len(feature_cols))))),
    class_weight=class_weight,
    random_state=123,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
rf_prob = rf.predict_proba(X_test)[:, 1]
rf_pred = (rf_prob > 0.5).astype(int)

rf_acc = accuracy_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_prob)
print("\nRandom Forest")
print("Accuracy:", round(rf_acc, 4))
print("AUC:", round(rf_auc, 4))
print("Confusion matrix:\n", confusion_matrix(y_test, rf_pred))
print("Report:\n", classification_report(y_test, rf_pred, digits=4))

# XGBoost
# Compute scale_pos_weight = negative / positive
pos = max(1, int((y_train == 1).sum()))
neg = max(1, int((y_train == 0).sum()))
scale_pos_weight = neg / pos

xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    n_estimators=400,
    reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight,
    random_state=123,
    n_jobs=-1,
)
xgb.fit(X_train, y_train)
xgb_prob = xgb.predict_proba(X_test)[:, 1]
xgb_pred = (xgb_prob > 0.5).astype(int)

xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_auc = roc_auc_score(y_test, xgb_prob)
print("\nXGBoost")
print("Accuracy:", round(xgb_acc, 4))
print("AUC:", round(xgb_auc, 4))
print("Confusion matrix:\n", confusion_matrix(y_test, xgb_pred))
print("Report:\n", classification_report(y_test, xgb_pred, digits=4))

# =========================
# 4) Compare and save artifacts
# =========================
compare = pd.DataFrame(
    {
        "Model": ["Random Forest", "XGBoost"],
        "Accuracy": [rf_acc, xgb_acc],
        "AUC": [rf_auc, xgb_auc],
    }
).sort_values("AUC", ascending=False)
print("\nModel comparison:")
print(compare.to_string(index=False))

compare_path = os.path.join(DATA_DIR, "model_comparison_py.csv")
compare.to_csv(compare_path, index=False)
print(f"Saved comparison to: {compare_path}")

# Feature importances
rf_imp = pd.DataFrame(
    {"feature": feature_cols, "importance": rf.feature_importances_}
).sort_values("importance", ascending=False)
xgb_imp = pd.DataFrame(
    {"feature": feature_cols, "importance": xgb.feature_importances_}
).sort_values("importance", ascending=False)

rf_imp_path = os.path.join(DATA_DIR, "rf_feature_importance_py.csv")
xgb_imp_path = os.path.join(DATA_DIR, "xgb_feature_importance_py.csv")
rf_imp.to_csv(rf_imp_path, index=False)
xgb_imp.to_csv(xgb_imp_path, index=False)
print(f"Saved RF importance to: {rf_imp_path}")
print(f"Saved XGB importance to: {xgb_imp_path}")

# ROC plot
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_prob)

plt.figure(figsize=(7, 5))
plt.plot(fpr_rf, tpr_rf, label=f"RF AUC={rf_auc:.3f}")
plt.plot(fpr_xgb, tpr_xgb, label=f"XGB AUC={xgb_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Expanded Model")
plt.legend(loc="lower right")
roc_path = os.path.join(DATA_DIR, "roc_expanded_model_py.png")
plt.tight_layout()
plt.savefig(roc_path, dpi=150)
plt.close()
print(f"Saved ROC plot to: {roc_path}")

print("\nDone.")

