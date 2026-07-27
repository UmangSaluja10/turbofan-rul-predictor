# Turbofan Engine RUL Predictor

End-to-end AI application that predicts the **Remaining Useful Life (RUL)**
of a jet engine from sensor readings, using NASA's CMAPSS dataset.
Model trained from scratch (RandomForestRegressor) — no pretrained models used.

## Project Structure
```
turbofan-rul-project/
├── data/                  <- put train_FD001.txt, test_FD001.txt, RUL_FD001.txt here
├── notebooks/
│   └── eda.py             <- exploratory data analysis
├── models/                <- saved model.pkl, scaler.pkl (auto-generated)
├── train_model.py         <- preprocessing + training + evaluation
├── backend/                (added in next step)
├── frontend/                (added in next step)
├── requirements.txt
└── README.md
```

## Setup (do this once)

1. **Install VS Code** (if not already) and the **Python extension** (Microsoft).
2. Open this folder in VS Code: `File > Open Folder`.
3. Open a terminal in VS Code: `` Terminal > New Terminal `` (or Ctrl+`).
4. Create and activate a virtual environment:

   **Windows:**
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

   **Mac/Linux:**
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```
   You should now see `(venv)` at the start of your terminal prompt.

5. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

6. Download the dataset — follow `data/README_DATA.md`. You need
   `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` inside the `data/` folder.

## Step 1: Run EDA

```
python notebooks/eda.py
```

This prints dataset stats to the terminal and saves plots to `notebooks/plots/`.
Open those PNGs in VS Code's file explorer to view them. Skim the printed
sensor variances — this tells you which sensors are "dead" (constant) and
safe to drop, which the training script already handles.

## Step 2: Train and evaluate the model

```
python train_model.py
```

This will:
- Load and preprocess the data
- Compute RUL labels (capped at 125 cycles — standard practice for this dataset)
- Scale features
- Train a RandomForestRegressor **from scratch**
- Evaluate on NASA's official test set and print MAE / RMSE / R²
- Save `models/rul_model.pkl`, `models/scaler.pkl`, `models/feature_columns.pkl`

**What to expect:** MAE typically lands somewhere around 15–25 cycles on
FD001 with this setup — good enough for a project demo, and you can quote
the exact number from your run in your technical report.

## Git / GitHub

Initialize the repo once you're happy with this stage:
```
git init
git add .
git commit -m "EDA + trained RUL model"
```

Add a `.gitignore` (create this file in the project root) with:
```
venv/
__pycache__/
*.pyc
data/*.txt
models/*.pkl
```
(We exclude the raw data files and trained model from git since they're
large/regeneratable — mention in your README how to regenerate them.
If your instructor wants the model file included, remove that line.)

Then push to GitHub:
```
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main
```

## Next steps
Once `train_model.py` runs successfully and you're happy with the metrics,
come back and I'll give you the **FastAPI backend** (`backend/`) that loads
`models/rul_model.pkl` and serves a `/predict` endpoint, followed by the
**frontend** (`frontend/`) that calls it.
