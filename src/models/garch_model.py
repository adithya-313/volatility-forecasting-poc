import numpy as np
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
  sys.path.append(str(root_dir))
from src.models.base import ClassicalModel
class GARCHModel(ClassicalModel):
  def __init__(self, p=1, q=1):
    super().__init__(name=f'GARCH({p},{q})')
    self.p = p
    self.q = q
    self.model = None
    self.fitted_model = None
  def fit(self, train_data, exog=None):
    super().fit(train_data, exog=exog)
    try:
      from arch import arch_model
    except ImportError:
      print("ERROR: 'arch' library not installed. Run: pip install arch")
      self.is_fitted = False
      return
    print(f"Training {self.name}")
    try:
      self.model = arch_model(
        train_data,
        x=exog,    
        mean='Constant',
        vol='Garch',
        p=self.p,
        q=self.q
      )
      self.fitted_model = self.model.fit(disp='off')
      self.residuals = self.fitted_model.resid
      print(f"{self.name} trained successfully")
      print(f"- AIC: {self.fitted_model.aic:.2f}")
      print(f"- BIC: {self.fitted_model.bic:.2f}")
    except Exception as e:
      print(f"Error training {self.name}: {str(e)}")
      self.is_fitted = False
  def forecast(self, horizon=1, confidence_level=0.95, exog=None):
    super().forecast(horizon=horizon, confidence_level=confidence_level, exog=exog)
    forecast_result = self.fitted_model.forecast(horizon=horizon, x=exog)
    predicted_variance = forecast_result.variance.values[-1, :]
    predictions = np.sqrt(predicted_variance)
    residual_std = np.std(self.residuals)
    margin = 1.96 * residual_std
    lower_bound = np.maximum(predictions - margin, 0.001)
    upper_bound = predictions + margin
    return predictions, (lower_bound, upper_bound)
  def get_params(self):
    return {
      'name': self.name,
      'p_lags': self.p,
      'q_lags': self.q,
      'is_fitted': self.is_fitted,
      'aic': self.fitted_model.aic if self.is_fitted else None,
    }
if __name__ == "__main__":
  from src.data.loader import DataLoader
  from src.data.preprocessor import DataSplitter
  print(f"Testing {GARCHModel(1,1).name} Model")
  print("=" * 60)
  data = DataLoader.generate_synthetic_volatility(n_samples=1000)
  volatility = data['volatility'].values
  train, val, test = DataSplitter.train_val_test_split(volatility)
  garch = GARCHModel(p=1, q=1)
  garch.fit(train)
  if garch.is_fitted:
    print("Predicting tomorrow and the next 4 days")
    predictions, (lower, upper) = garch.forecast(horizon=5)
    print(f"Predictions: {predictions}")
    print(f"Lower Bounds: {lower}")
    print(f"Upper Bounds: {upper}")
    print("GARCH model test complete!")