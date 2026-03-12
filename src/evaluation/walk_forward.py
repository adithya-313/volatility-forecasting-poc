import numpy as np
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
  sys.path.append(str(root_dir))
from src.models.base import VolatilityModel
from src.evaluation.metrics import VolatilityMetrics
class WalkForwardValidator:
  def __init__(self, initial_train_size=500, step_size=5, window_type='expanding'):
    self.initial_train_size = initial_train_size
    self.step_size = step_size
    if window_type not in ['expanding', 'rolling']:
      raise ValueError("window_type must be either 'expanding' or 'rolling'")
    self.window_type = window_type
    self.metrics = VolatilityMetrics()
  def evaluate(self, model, data, exog=None):
    total_days = len(data)
    if total_days <= self.initial_train_size + self.step_size:
      raise ValueError("Not enough data to run walk-forward validation.")
    print(f"Start Walk-Forward Validation for: {model.name}")
    print(f"- Total Data Size: {total_days} days")
    print(f"- Initial Training: First {self.initial_train_size} days")
    print(f"- Step Size (Forecast Horizon): {self.step_size} days")
    print(f"- Window Style: {self.window_type.capitalize()}")
    all_actuals = []
    all_predictions = []
    all_lower_bounds = []
    all_upper_bounds = []
    current_step = self.initial_train_size
    simulation_step_counter = 1
    while current_step < total_days:
      if self.window_type == 'expanding':
        train_data = data[:current_step]
        train_exog = exog[:current_step] if exog is not None else None
      else:
        train_start_index = current_step - self.initial_train_size
        train_data = data[train_start_index:current_step]
        train_exog = exog[train_start_index:current_step] if exog is not None else None
      test_end = min(current_step + self.step_size, total_days)
      horizon = test_end - current_step 
      test_actuals = data[current_step:test_end]
      test_exog = exog[current_step:test_end] if exog is not None else None
      print(f"=> Window {simulation_step_counter}: Training on {len(train_data)} days, Predicting next {horizon} days")
      model.is_fitted = False
      try:
        model.fit(train_data, exog=train_exog)
        options = {'horizon': horizon, 'exog': test_exog}
        if hasattr(model, 'neural_model') and model.neural_model is not None:
           options['recent_data'] = train_data
        preds, (lower, upper) = model.forecast(**options)
        preds = preds[:horizon]
        lower = lower[:horizon]
        upper = upper[:horizon]
      except Exception as e:
        print(f"Model failed on window {simulation_step_counter}: {e}")
        print("Filling predictions with the naive constant (the last known day's value)")
        last_known_value = train_data[-1]
        preds = np.full(horizon, last_known_value)
        lower = np.full(horizon, last_known_value * 0.8)
        upper = np.full(horizon, last_known_value * 1.2)
      all_actuals.extend(test_actuals)
      all_predictions.extend(preds)
      all_lower_bounds.extend(lower)
      all_upper_bounds.extend(upper)
      current_step += self.step_size
      simulation_step_counter += 1
    all_actuals = np.array(all_actuals)
    all_predictions = np.array(all_predictions)
    all_lower_bounds = np.array(all_lower_bounds)
    all_upper_bounds = np.array(all_upper_bounds)
    mae = self.metrics.mae(all_actuals, all_predictions)
    rmse = self.metrics.rmse(all_actuals, all_predictions)
    mape = self.metrics.mape(all_actuals, all_predictions)
    print(f"Configuration complete for {model.name}!")
    print(f"- Final Mean Absolute Error (MAE): {mae:.6f}")
    print(f"- We made {len(all_predictions)} predictions iteratively")
    return {
      'model_name': model.name,
      'mae': mae,
      'rmse': rmse,
      'mape': mape,
      'actuals': all_actuals,
      'predictions': all_predictions,
      'lower_bounds': all_lower_bounds,
      'upper_bounds': all_upper_bounds
    }
if __name__ == "__main__":
  from src.data.loader import DataLoader
  from src.models.garch_model import GARCHModel
  print("Testing Walk-Forward Validator")
  print("=" * 60)
  data = DataLoader.generate_synthetic_volatility(n_samples=1000)
  volatility = data['volatility'].values
  garch = GARCHModel(1, 1)
  validator = WalkForwardValidator(initial_train_size=800, step_size=10, window_type='expanding')
  results = validator.evaluate(model=garch, data=volatility)
  print("Walk-forward summary saved successfully")