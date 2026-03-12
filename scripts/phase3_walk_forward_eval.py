import sys
import numpy as np
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
  print("Phase 3 Production Evaluation")
  print("Walk-Forward Validation with Multivariate Data")
  print("---")
  print("Loading real data (AAPL Volatility + VIX Index)")
  try:
    data = DataLoader.load_yahoo_finance(
      ticker='AAPL',
      start_date='2021-01-01',
      end_date='2024-01-01',
      include_vix=True,  
      include_volume=False 
    )
    print(f"Loaded {len(data)} trading days")
  except Exception as e:
    print(f"Could not load Yahoo Finance data: {e}")
    print("Please check your internet connection or run pip install yfinance")
    return
  print("Preprocessing data")
  prep = DataPreprocessor()
  data = prep.handle_missing_values(data, 'volatility')
  data = prep.handle_outliers(data, 'volatility')
  data = prep.handle_missing_values(data, 'vix')
  data = prep.handle_outliers(data, 'vix')
  volatility = data['volatility'].values
  vix = data['vix'].values.reshape(-1, 1)
  vix_scaled, _, _ = prep.scale_features(X_train=vix)
  print("Initializing Models")
  models_to_test = [
    GARCHModel(p=1, q=1),
    SARIMAModel(order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)),
    ProphetModel(changepoint_prior_scale=0.1),
    LSTMModel(
      sequence_length=15, 
      lstm_units=(64, 32), 
      epochs=20, 
      batch_size=16
    ),
    AdvancedLSTMModel(
      sequence_length=15, 
      lstm_units=(128, 64), 
      epochs=30, 
      batch_size=16,
      dropout=0.3
    )
  ]
  print("Running simulated production evaluation")
  validator = WalkForwardValidator(
    initial_train_size=600, 
    step_size=10, 
    window_type='expanding'
  )
  final_results = {}
  for model in models_to_test:
    try:
      if model.name == 'AdvancedLSTM':
        result = validator.evaluate(model, volatility, exog=vix_scaled)
      else:
        result = validator.evaluate(model, volatility, exog=None)
      final_results[model.name] = result['mae']
    except Exception as e:
      print(f"Testing {model.name} crashed: {e}")
      final_results[model.name] = float('inf')
  print("---")
  print("Final Rankings (Mean Absolute Error - Lower is Better)")
  print("---")
  sorted_results = sorted(final_results.items(), key=lambda x: x[1])
  for rank, (model_name, mae) in enumerate(sorted_results):
    medal = " (1st)" if rank == 0 else " (2nd)" if rank == 1 else " (3rd)" if rank == 2 else "   "
    if mae == float('inf'):
      print(f"{rank+1}. {model_name:20}: FAILED")
    else:
      print(f"{rank+1}. {model_name:20}: {mae:.6f} {medal}")
  print("---")
  print("Why scores look different: Walk-forward validation is stricter and reflects real market shifts.")
  print("---")
if __name__ == "__main__":
  main()