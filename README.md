# ✈️ Predictive Maintenance using NASA CMAPSS Dataset

## 📌 Overview
This project focuses on **Remaining Useful Life (RUL) prediction** for jet engines using sensor data from NASA’s CMAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset. The goal is to build a machine learning model that can estimate how many operational cycles an engine has left before failure, enabling **proactive maintenance**.

---

## 🚀 Problem Statement
Traditional maintenance strategies are:
- **Reactive** (fix after failure)
- **Preventive** (scheduled, but inefficient)

This project implements a **predictive maintenance system** using sensor data to:
- Predict **RUL (in cycles)**
- Classify engine health into:
  - ✅ Healthy
  - ⚠️ Warning
  - ❌ Critical

---

## 📂 Dataset
- **Source**: NASA CMAPSS Turbofan Engine Degradation Dataset  
- **Type**: Multivariate time-series (converted to tabular format)
- **Characteristics**:
  - 100 engines (FD001 subset)
  - 20+ sensor readings per cycle
  - No missing values
  - Clean and ready-to-use

---

## 🧠 Methodology

### Step 1: Exploratory Data Analysis (EDA)
- Checked dataset shape and structure
- Verified missing values (none found)
- Analyzed sensor variance
- Dropped constant / near-constant sensors:
sensor_1, sensor_19, sensor_18, sensor_10,
sensor_16, sensor_5, sensor_6
- Generated:
- Sensor trend plots
- Correlation heatmap

---

### Step 2: Feature Engineering
- Selected 17 meaningful features:
op_setting_1, op_setting_2, op_setting_3,
sensor_2, sensor_3, sensor_4, sensor_7,
sensor_8, sensor_9, sensor_11, sensor_12,
sensor_13, sensor_14, sensor_15,
sensor_17, sensor_20, sensor_21

- Applied **StandardScaler** for normalization

---

### Step 3: Model Training
- Model: **RandomForestRegressor**
- Training: From scratch (no pretrained models)
- Advantages:
- Handles nonlinear relationships
- Robust to noise
- Fast training (CPU-friendly)

---

### Step 4: Evaluation

| Metric | Value |
|------|------|
| MAE  | 12.22 cycles |
| RMSE | 17.27 cycles |
| R²   | 0.814 |

---

### Step 5: Feature Importance (Top 10)

| Feature     | Importance |
|------------|-----------|
| sensor_11  | 0.6388 |
| sensor_9   | 0.1345 |
| sensor_4   | 0.0693 |
| sensor_12  | 0.0411 |
| sensor_7   | 0.0208 |
| sensor_14  | 0.0199 |
| sensor_15  | 0.0147 |
| sensor_21  | 0.0100 |
| sensor_13  | 0.0097 |
| sensor_2   | 0.0093 |

---

## 🏗️ Project Structure
major_project/
│
├── backend/
│ └── main.py # FastAPI backend (prediction API)
│
├── frontend/
│ └── index.html # Simple UI for user input
│
├── data/
│ └── train_FD001.txt # Dataset
│
├── models/
│ ├── model.pkl # Trained model
│ ├── scaler.pkl # Feature scaler
│ └── features.json # Feature list
│
├── notebooks/
│ ├── eda.py # Data analysis script
│ └── plots/ # Generated plots
│
├── train_model.py # Model training script
├── requirements.txt
└── README.md

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/predictive-maintenance-rul.git
cd predictive-maintenance-rul

2. Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

3. Install dependencies
pip install -r requirements.txt
▶️ Usage
Run EDA
python notebooks/eda.py
Train Model
python train_model.py
Start Backend (FastAPI)
uvicorn backend.main:app --reload
Open Frontend

Open frontend/index.html in your browser.

🔌 API Endpoint
POST /predict

Request Body:

{
  "sensor_2": 641.82,
  "sensor_3": 1589.70,
  ...
}

Response:

{
  "predicted_rul": 145.3,
  "status": "Warning",
  "message": "Engine nearing maintenance window"
}
🌍 Real-World Impact
Reduces unexpected engine failures
Optimizes maintenance scheduling
Saves operational costs
Improves safety in aviation and industry
🔮 Future Improvements
Use XGBoost or LightGBM for better accuracy
Add time-series models (LSTM)
Deploy on cloud (AWS / Azure)
Build interactive dashboard (React)
👨‍💻 Author

Umang Saluja

📜 License

This project is open-source and available under the MIT License.


If you want, I can also:
1. :contentReference[oaicite:0]{index=0}
2. :contentReference[oaicite:1]{index=1}
3. :contentReference[oaicite:2]{index=2}

Just tell me.