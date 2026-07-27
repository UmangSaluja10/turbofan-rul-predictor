"""
Train a Remaining-Useful-Life (RUL) regression model on the NASA CMAPSS
FD001 turbofan dataset, from scratch (no pretrained models).

Run from the project root:
    python train_model.py

Outputs:
    models/rul_model.pkl        <- trained model
    models/scaler.pkl           <- fitted StandardScaler
    models/feature_columns.pkl  <- list of feature column names used
Prints:
    Evaluation metrics (MAE, RMSE, R2) on the held-out NASA test set.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "train_FD001.txt")
TEST_PATH = os.path.join(DATA_DIR, "test_FD001.txt")
RUL_PATH = os.path.join(DATA_DIR, "RUL_FD001.txt")

COLS = (
    ["unit_number", "time_in_cycles", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Sensors identified as constant/near-constant in EDA for FD001.
# (If your EDA output differs, adjust this list.)
DROP_SENSORS = ["sensor_1", "sensor_5", "sensor_6", "sensor_10", "sensor_16", "sensor_18", "sensor_19"]

RUL_CAP = 125  # standard trick: cap RUL so the model isn't penalized for
               # early-life cycles where degradation isn't visible yet


# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------
def load_raw(path):
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLS)


def add_rul_train(df):
    max_cycles = df.groupby("unit_number")["time_in_cycles"].transform("max")
    df["RUL"] = max_cycles - df["time_in_cycles"]
    df["RUL"] = df["RUL"].clip(upper=RUL_CAP)
    return df


def get_feature_columns(df):
    op_cols = ["op_setting_1", "op_setting_2", "op_setting_3"]
    sensor_cols = [c for c in df.columns if c.startswith("sensor_") and c not in DROP_SENSORS]
    return op_cols + sensor_cols


def build_test_set(feature_cols, scaler):
    """
    For the test set, NASA gives partial run histories and a single true RUL
    per engine (the RUL at the LAST recorded cycle). We take the last row of
    each engine's history as the feature vector.
    """
    test_df = load_raw(TEST_PATH)
    true_rul = pd.read_csv(RUL_PATH, header=None, names=["RUL"])
    true_rul["RUL"] = true_rul["RUL"].clip(upper=RUL_CAP)

    last_cycle = test_df.groupby("unit_number").last().reset_index()
    X_test = last_cycle[feature_cols].values
    X_test = scaler.transform(X_test)
    y_test = true_rul["RUL"].values
    return X_test, y_test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading training data...")
    train_df = load_raw(TRAIN_PATH)
    train_df = add_rul_train(train_df)

    feature_cols = get_feature_columns(train_df)
    print(f"\nUsing {len(feature_cols)} features:")
    print(feature_cols)

    X_train = train_df[feature_cols].values
    y_train = train_df["RUL"].values

    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print("Training RandomForestRegressor (from scratch)...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    # -----------------------------------------------------------------
    # Evaluate on NASA's official held-out test set
    # -----------------------------------------------------------------
    print("\nEvaluating on official test set...")
    X_test, y_test = build_test_set(feature_cols, scaler)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("\n=== Evaluation Metrics (Test Set) ===")
    print(f"MAE  : {mae:.2f} cycles")
    print(f"RMSE : {rmse:.2f} cycles")
    print(f"R2   : {r2:.3f}")

    # -----------------------------------------------------------------
    # Feature importance (for your technical report / justification)
    # -----------------------------------------------------------------
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False)
    print("\n=== Top 10 Feature Importances ===")
    print(importances.head(10))

    # -----------------------------------------------------------------
    # Save artifacts for the backend to load
    # -----------------------------------------------------------------
    joblib.dump(model, os.path.join(MODEL_DIR, "rul_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_columns.pkl"))

    print(f"\nSaved model + scaler + feature list to: {MODEL_DIR}")
    print("You're ready to build the FastAPI backend next.")


if __name__ == "__main__":
    main()
