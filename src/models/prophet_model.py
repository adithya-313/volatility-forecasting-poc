import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
  sys.path.append(str(root_dir))
from src.models.base import ClassicalModel
from src.config import Config
class ProphetModel(ClassicalModel):
  def __init__(self, 
         yearly_seasonality=False,
         weekly_seasonality=False,
         daily_seasonality=False,
         changepoint_prior_scale=0.05):
    super().__init__(name='Prophet')
    self.yearly_seasonality = yearly_seasonality
    self.weekly_seasonality = weekly_seasonality
    self.daily_seasonality = daily_seasonality
    self.changepoint_prior_scale = changepoint_prior_scale
    self.fitted_model = None
  def fit(self, train_data, exog=None):
    super().fit(train_data, exog=exog)
    self.train_exog = exog
    try:
      from prophet import Prophet
    except ImportError:
      print("Error: prophet not installed. Run: pip install prophet")
      return
    print(f"Training {self.name}")
    print(f"- Yearly seasonality: {self.yearly_seasonality}")
    print(f"- Weekly seasonality: {self.weekly_seasonality}")
    print(f"- Daily seasonality: {self.daily_seasonality}")
    try:
      df = pd.DataFrame({
        'ds': pd.date_range(start='2015-01-01', periods=len(train_data), freq='B'),
        'y': train_data
      })
      if exog is not None:
        for i in range(exog.shape[1]):
          df[f'regressor_{i}'] = exog[:, i]
      self.model = Prophet(
        yearly_seasonality=self.yearly_seasonality,
        weekly_seasonality=self.weekly_seasonality,
        daily_seasonality=self.daily_seasonality,
        changepoint_prior_scale=self.changepoint_prior_scale,
        interval_width=0.95
      )
      if exog is not None:
        for i in range(exog.shape[1]):
          self.model.add_regressor(f'regressor_{i}')
      null_device = 'NUL' if os.name == 'nt' else '/dev/null'
      with open(null_device, 'w') as devnull:
        import sys
        old_stdout = sys.stdout
        sys.stdout = devnull
        self.fitted_model = self.model.fit(df)
        sys.stdout = old_stdout
      print(f"{self.name} trained successfully")
    except Exception as e:
      print(f"Error training {self.name}: {str(e)}")
      self.is_fitted = False
  def forecast(self, horizon=1, confidence_level=0.95, exog=None):
    super().forecast(horizon=horizon, confidence_level=confidence_level, exog=exog)
    future = self.model.make_future_dataframe(periods=horizon, freq='B')
    if self.train_exog is not None:
      if exog is None or len(exog) < horizon:
        raise ValueError("exog matching horizon length is required for Prophet when trained with exog")
      for i in range(self.train_exog.shape[1]):
        full_exog = np.concatenate([self.train_exog[:, i], exog[:horizon, i]])
        future[f'regressor_{i}'] = full_exog
    forecast = self.model.predict(future)
    forecast = forecast.iloc[-horizon:]
    predictions = forecast['yhat'].values
    lower_bound = forecast['yhat_lower'].values
    upper_bound = forecast['yhat_upper'].values
    return predictions, (lower_bound, upper_bound)
  def get_params(self):
    return {
      'name': self.name,
      'yearly_seasonality': self.yearly_seasonality,
      'weekly_seasonality': self.weekly_seasonality,
      'daily_seasonality': self.daily_seasonality,
      'changepoint_prior_scale': self.changepoint_prior_scale,
      'is_fitted': self.is_fitted,
    }
if __name__ == "__main__":
  from src.data.loader import DataLoader
  from src.data.preprocessor import DataPreprocessor, DataSplitter
  print("Testing Prophet Model")
  print("=" * 60)
  data = DataLoader.generate_synthetic_volatility(n_samples=1500)
  prep = DataPreprocessor()
  data = prep.handle_missing_values(data, 'volatility')
  data = prep.handle_outliers(data, 'volatility')
  vol = data['volatility'].values
  train, val, test = DataSplitter.train_val_test_split(vol)
  print("Training")
  prophet = ProphetModel(
    yearly_seasonality=Config.PROPHET_CONFIG['yearly_seasonality'],
    weekly_seasonality=Config.PROPHET_CONFIG['weekly_seasonality'],
    daily_seasonality=Config.PROPHET_CONFIG['daily_seasonality']
  )
  prophet.fit(train)
  print("Forecasting")
  predictions, (lower, upper) = prophet.forecast(horizon=10)
  print(f"Predictions (first 5): {predictions}")
  print(f"Lower bounds (first 5): {lower}")
  print(f"Upper bounds (first 5): {upper}")
  print(f"Model parameters: {prophet.get_params()}")
  print("Prophet test complete!")