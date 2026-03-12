import numpy as np
import sys
import os
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
  sys.path.append(str(root_dir))
from src.models.base import ClassicalModel
from src.config import Config
class SARIMAModel(ClassicalModel):
  def __init__(self, order=(2, 1, 2), seasonal_order=(1, 1, 1, 252)):
    super().__init__(name='SARIMA')
    self.order = order
    self.seasonal_order = seasonal_order
    self.model = None
    self.fitted_model = None
  def fit(self, train_data, exog=None):
    super().fit(train_data, exog=exog) 
    try:
      from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
      print("ERROR: statsmodels not installed. Run: pip install statsmodels")
      return
    print(f"Training {self.name}")
    print(f"- Order: {self.order}")
    print(f"- Seasonal order: {self.seasonal_order}")
    try:
      self.model = SARIMAX(
        train_data,
        exog=exog,
        order=self.order,
        seasonal_order=self.seasonal_order,
        enforce_stationarity=True,
        enforce_invertibility=True
      )
      self.fitted_model = self.model.fit(disp=False)
      resid = self.fitted_model.resid
      if hasattr(resid, 'values'):
        self.residuals = resid.values
      else:
        self.residuals = resid
      print(f"{self.name} trained successfully")
      print(f"- AIC: {self.fitted_model.aic:.2f}")
      print(f"- BIC: {self.fitted_model.bic:.2f}")
    except Exception as e:
      print(f"Error training {self.name}: {str(e)}")
      self.is_fitted = False
  def forecast(self, horizon=1, confidence_level=0.95, exog=None):
    super().forecast(horizon=horizon, confidence_level=confidence_level, exog=exog) 
    forecast_result = self.fitted_model.get_forecast(steps=horizon, exog=exog)
    predictions = forecast_result.predicted_mean
    if hasattr(predictions, 'values'):
      predictions = predictions.values
    conf_int = forecast_result.conf_int(alpha=1-confidence_level)
    if hasattr(conf_int, 'iloc'):
      lower_bound = conf_int.iloc[:, 0].values
      upper_bound = conf_int.iloc[:, 1].values
    else:
      lower_bound = conf_int[:, 0]
      upper_bound = conf_int[:, 1]
    return predictions, (lower_bound, upper_bound)
  def get_params(self):
    return {
      'name': self.name,
      'order': self.order,
      'seasonal_order': self.seasonal_order,
      'is_fitted': self.is_fitted,
      'aic': self.fitted_model.aic if self.is_fitted else None,
      'bic': self.fitted_model.bic if self.is_fitted else None,
    }
if __name__ == "__main__":
  from src.data.loader import DataLoader
  from src.data.preprocessor import DataPreprocessor, DataSplitter
  print("Testing SARIMA Model")
  print("=" * 60)
  print("1. Generating data")
  data = DataLoader.generate_synthetic_volatility(n_samples=1500)
  print("2. Preprocessing")
  prep = DataPreprocessor()
  data = prep.handle_missing_values(data, 'volatility')
  data = prep.handle_outliers(data, 'volatility')
  print("3. Splitting data")
  vol = data['volatility'].values
  train, val, test = DataSplitter.train_val_test_split(vol)
  print("4. Training SARIMA")
  sarima = SARIMAModel(
    order=(1, 1, 1),     
    seasonal_order=(0, 0, 0, 0) 
  )
  sarima.fit(train)
  print("5. Forecasting")
  predictions, (lower, upper) = sarima.forecast(horizon=10)
  print("Forecast results:")
  print(f"- Predictions: {predictions}")
  print(f"- Lower bounds: {lower}")
  print(f"- Upper bounds: {upper}")
  print("Model parameters:")
  params = sarima.get_params()
  for key, val in params.items():
    print(f"- {key}: {val}")
  print("SARIMA test complete!")