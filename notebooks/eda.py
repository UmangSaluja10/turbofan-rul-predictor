"""
EDA script for NASA CMAPSS Turbofan Degradation Dataset (FD001 subset).

Run this from the project root:
    python notebooks/eda.py

It will:
  1. Load train_FD001.txt
  2. Show basic structure/stats
  3. Compute RUL (Remaining Useful Life) for the training set
  4. Plot sensor trends for a sample engine
  5. Identify constant / low-variance sensors (candidates to drop)

All plots are saved to notebooks/plots/ so you don't need a GUI.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# 1. Column names (dataset has no header row)
# ---------------------------------------------------------------------------
COLS = (
    ["unit_number", "time_in_cycles", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "train_FD001.txt")

# ---------------------------------------------------------------------------
# 2. Load data
# ---------------------------------------------------------------------------
def load_train_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLS)
    return df


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """RUL = max cycle for that engine - current cycle."""
    max_cycles = df.groupby("unit_number")["time_in_cycles"].transform("max")
    df["RUL"] = max_cycles - df["time_in_cycles"]
    return df


def main():
    print("Loading data from:", TRAIN_PATH)
    df = load_train_data(TRAIN_PATH)
    df = add_rul(df)

    print("\n--- Shape ---")
    print(df.shape)

    print("\n--- First 5 rows ---")
    print(df.head())

    print("\n--- Number of engines (units) ---")
    print(df["unit_number"].nunique())

    print("\n--- Cycles per engine (describe) ---")
    print(df.groupby("unit_number")["time_in_cycles"].max().describe())

    print("\n--- Missing values ---")
    print(df.isnull().sum().sum(), "missing values total")

    # -----------------------------------------------------------------
    # Identify low-variance / constant sensors (safe to drop later)
    # -----------------------------------------------------------------
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    variances = df[sensor_cols].var().sort_values()
    print("\n--- Sensor variances (lowest first) ---")
    print(variances)

    low_var_sensors = variances[variances < 1e-5].index.tolist()
    print("\nLikely constant / near-constant sensors (drop candidates):")
    print(low_var_sensors)

    # -----------------------------------------------------------------
    # Plot: sensor trends for one sample engine
    # -----------------------------------------------------------------
    sample_unit = df["unit_number"].unique()[0]
    sample = df[df["unit_number"] == sample_unit]

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    for ax, sensor in zip(axes, ["sensor_2", "sensor_3", "sensor_4", "sensor_11"]):
        ax.plot(sample["time_in_cycles"], sample[sensor])
        ax.set_title(f"{sensor} over time — engine {sample_unit}")
        ax.set_ylabel(sensor)
    axes[-1].set_xlabel("time_in_cycles")
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "sample_engine_sensor_trends.png")
    plt.savefig(out_path)
    print(f"\nSaved plot: {out_path}")

    # -----------------------------------------------------------------
    # Plot: correlation heatmap of sensors vs RUL
    # -----------------------------------------------------------------
    plt.figure(figsize=(12, 10))
    corr = df[sensor_cols + ["RUL"]].corr()
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Sensor correlation heatmap (incl. RUL)")
    plt.tight_layout()
    out_path2 = os.path.join(PLOTS_DIR, "correlation_heatmap.png")
    plt.savefig(out_path2)
    print(f"Saved plot: {out_path2}")

    print("\nEDA complete. Review the printed variances and plots/ folder")
    print("before moving on to train_model.py.")


if __name__ == "__main__":
    main()
