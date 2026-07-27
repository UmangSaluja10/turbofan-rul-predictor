# Downloading the Dataset

This project uses NASA's **C-MAPSS Turbofan Engine Degradation Simulation Dataset**
(FD001 subset — simplest one, single operating condition, single fault mode).

### Option A (recommended, fastest): Kaggle
1. Go to: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
2. Download the zip and extract it.
3. Copy these 3 files into this `data/` folder:
   - `train_FD001.txt`
   - `test_FD001.txt`
   - `RUL_FD001.txt`

### Option B: Official NASA source
1. Go to: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
2. Download the "Turbofan Engine Degradation Simulation Data Set" zip.
3. Extract and copy the same 3 files (`train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`)
   into this `data/` folder.

### About the data
- Space-separated text files, **no header row**.
- 26 columns: `unit_number, time_in_cycles, op_setting_1, op_setting_2, op_setting_3, sensor_1 ... sensor_21`
- `train_FD001.txt`: engines run until failure (we compute RUL ourselves).
- `test_FD001.txt`: engines cut off before failure.
- `RUL_FD001.txt`: true remaining useful life for each engine in the test set (one number per line).

Once the 3 files are in this folder, you're ready to run `train_model.py`.
