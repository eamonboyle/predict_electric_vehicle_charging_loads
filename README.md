# Predicting Residential EV Charging Loads

Predict electric vehicle charging loads (kWh) from Norwegian residential charging data using traffic features and a PyTorch neural network.

## Overview

This project trains a regression model to estimate how much energy (`El_kWh`) is consumed per EV charging session. The model uses:

- **EV charging reports** — plug-in/plug-out times, duration, user type, month, weekday
- **Traffic data** — hourly vehicle counts at 5 nearby locations (hypothesis: traffic density may correlate with charging behavior)

Pipeline: load and merge datasets → clean and encode → 80/20 train/test split → linear regression baseline → PyTorch MLP.

## Project Structure

```
predict_electric_vehicle_charging_loads/
├── datasets/
│   ├── EV charging reports.csv
│   └── Local traffic distribution.csv
├── models/
│   └── model.pth              # Saved PyTorch model (created on run)
├── predict_electric_vehicle_charging_loads.ipynb
├── requirements.txt
└── README.md
```

## Setup

1. Create a virtual environment (recommended):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   # source .venv/bin/activate   # macOS/Linux
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Open and run the Jupyter notebook:

```bash
jupyter notebook predict_electric_vehicle_charging_loads.ipynb
```

Or use JupyterLab, VS Code, or Cursor. The notebook will:

1. Load EV charging and traffic data
2. Merge on plug-in hour
3. Clean and encode features
4. Train a linear regression baseline
5. Train a PyTorch MLP (56 → 26 → 1)
6. Save the model to `models/model.pth`

## Results

| Model              | Test MSE | √MSE (≈ avg error, kWh) |
|--------------------|----------|--------------------------|
| Linear regression  | ~121     | ~11                      |
| Neural network     | ~118     | ~10.9                    |

The neural network slightly outperforms the baseline. Possible extensions: different feature sets, architectures, learning rates, or training duration.

## Requirements

- Python 3.8+
- numpy, pandas, matplotlib, torch, scikit-learn
