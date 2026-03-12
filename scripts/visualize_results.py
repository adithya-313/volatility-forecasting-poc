import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
  sys.path.append(str(root_dir))
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.evaluation.walk_forward import WalkForwardValidator
from src.models.sarima_model import SARIMAModel
from src.models.garch_model import GARCHModel
from src.models.prophet_model import ProphetModel
from src.models.lstm_model import LSTMModel
from src.models.lstm_advanced import AdvancedLSTMModel
def main():
  print("---")
  print("Phase 3 Visualization Engine")
  print("Generating Comparative Forecast Plots")
  print("---")
  docs_dir = root_dir / 'docs'
  docs_dir.mkdir(exist_ok=True)
  print("Preparing data")
  loader = DataLoader()
  data = loader.load_yahoo_finance(
    ticker='AAPL',
    start_date='2021-01-01',
    end_date='2024-01-01',
    include_vix=True
  )
  prep = DataPreprocessor()
  data = data.dropna()
  y = data['volatility'].values
  exog_vix = data['vix'].values.reshape(-1, 1)
  exog_scaled, _, _ = prep.scale_features(X_train=exog_vix)
  split_idx = len(y) - 40
  y_train, y_test = y[:split_idx], y[split_idx:]
  exog_train, exog_test = exog_scaled[:split_idx], exog_scaled[split_idx:]
  print(f"Fitting and Forecasting on {len(y_test)} test days")
  models = {
    'SARIMA': SARIMAModel(order=(1, 1, 1)),
    'Prophet': ProphetModel(),
    'GARCH(1,1)': GARCHModel(1, 1),
    'LSTM (Basic)': LSTMModel(epochs=5),
    'Advanced LSTM (VIX)': AdvancedLSTMModel(epochs=5)
  }
  predictions = {}
  horizon = len(y_test)
  for name, model in models.items():
    print(f"{name} fitting")
    try:
      if name == 'Advanced LSTM (VIX)':
        model.fit(y_train, exog=exog_train)
        print(f"{name} fitted. Forecasting")
        preds = model.forecast(horizon=horizon, exog=exog_test)
      elif name in ['SARIMA', 'Prophet']:
        model.fit(y_train, exog=exog_train)
        print(f"{name} fitted. Forecasting")
        preds = model.forecast(horizon=horizon, exog=exog_test)
      else:
        model.fit(y_train)
        print(f"{name} fitted. Forecasting")
        preds = model.forecast(horizon=horizon)
      predictions[name] = preds
      print(f"{name} results stored")
    except Exception as e:
      print(f"{name}: {e}")
  print("Generating final plot")
  sns.set_theme(style="darkgrid")
  plt.figure(figsize=(15, 8))
  time_pts = np.arange(len(y_test))
  plt.plot(time_pts, y_test, label='Actual AAPL Volatility', color='black', linewidth=3, alpha=0.9)
  colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
  for (name, preds), color in zip(predictions.items(), colors):
    plt.plot(time_pts, preds, label=f'{name} Forecast', linewidth=2, linestyle='--', color=color)
  plt.title('Final Model Comparison: Predicted vs Actual Volatility (AAPL)', fontsize=18, fontweight='bold')
  plt.xlabel('Trading Days (Test Set)', fontsize=14)
  plt.ylabel('Realized Volatility', fontsize=14)
  plt.legend(prop={'size': 12}, loc='upper left')
  docs_dir = root_dir / 'docs'
  docs_dir.mkdir(exist_ok=True)
  save_path = docs_dir / 'final_forecast_comparison.png'
  plt.savefig(save_path, dpi=300, bbox_inches='tight')
  print(f"DONE! Visualization saved to:   {save_path}")
  print("---")
  plt.tight_layout()
  save_path = docs_dir / 'phase3_forecast_comparison.png'
  plt.savefig(save_path, dpi=300)
  print(f"Success! Visualization saved to:   {save_path}")
  print("---")
if __name__ == "__main__":
  main()