import numpy as np
import sys
import os
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
  sys.path.append(str(root_dir))
from src.models.base import ClassicalModel
from src.config import Config
class ExponentialSmoothingModel(ClassicalModel):
  def __init__(self, trend='add', seasonal=None, seasonal_periods=252):
    super().__init__(name='ExponentialSmoothing')
    self.trend = trend
    self.seasonal = seasonal
    self.seasonal_periods = seasonal_periods
    self.fitted_model = None
  def fit(self, train_data):
    super().fit(train_data)
    try:
      from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ImportError:
      print("ERROR: statsmodels not installed")
      return
    print(f"Training {self.name}")
    print(f"- Trend: {self.trend}")
    print(f"- Seasonal: {self.seasonal}")
    try:
      self.model = ExponentialSmoothing(
        train_data,
        trend=self.trend,
        seasonal=self.seasonal,
        seasonal_periods=self.seasonal_periods if self.seasonal else None
      )
      self.fitted_model = self.model.fit(optimized=True)
      resid = self.fitted_model.resid
      if hasattr(resid, 'values'):
        self.residuals = resid.values
      else:
        self.residuals = resid
      print(f"{self.name} trained successfully")
      print(f"- SSE: {self.fitted_model.sse:.4f}")
    except Exception as e:
      print(f"Error training {self.name}: {str(e)}")
      self.is_fitted = False
  def forecast(self, horizon=1, confidence_level=0.95):
    super().forecast(horizon=horizon, confidence_level=confidence_level)
    predictions = self.fitted_model.forecast(steps=horizon)
    residual_std = np.std(self.residuals)
    z_score = 1.96 
    lower_bound = predictions - z_score * residual_std
    upper_bound = predictions + z_score * residual_std
    res_predictions = predictions.values if hasattr(predictions, 'values') else predictions
    res_lower = lower_bound.values if hasattr(lower_bound, 'values') else lower_bound
    res_upper = upper_bound.values if hasattr(upper_bound, 'values') else upper_bound
    return res_predictions, (res_lower, res_upper)
  def get_params(self):
    return {
      'name': self.name,
      'trend': self.trend,
      'seasonal': self.seasonal,
      'seasonal_periods': self.seasonal_periods,
      'is_fitted': self.is_fitted,
      'sse': self.fitted_model.sse if self.is_fitted else None,
    }
if __name__ == "__main__":
  from src.data.loader import DataLoader
  from src.data.preprocessor import DataPreprocessor, DataSplitter
  print("Testing Exponential Smoothing Model")
  print("=" * 60)
  data = DataLoader.generate_synthetic_volatility(n_samples=1500)
  prep = DataPreprocessor()
  data = prep.handle_missing_values(data, 'volatility')
  data = prep.handle_outliers(data, 'volatility')
  vol = data['volatility'].values
  train, val, test = DataSplitter.train_val_test_split(vol)
  print("Training")
  es = ExponentialSmoothingModel(
    trend=Config.EXP_SMOOTHING_CONFIG['trend'],
    seasonal=Config.EXP_SMOOTHING_CONFIG['seasonal']
  )
  es.fit(train)
  print("Forecasting")
  predictions, (lower, upper) = es.forecast(horizon=10)
  print(f"Predictions (first 5): {predictions}")
  print(f"Lower bounds (first 5): {lower}")
  print(f"Upper bounds (first 5): {upper}")
  print(f"Model parameters: {es.get_params()}")
  print("Exponential Smoothing test complete!")