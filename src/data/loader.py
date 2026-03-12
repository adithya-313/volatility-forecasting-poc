import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
class DataLoader:
  @staticmethod
  def generate_synthetic_volatility(
    n_samples = 1500,
    volatility_mean = 0.02,
    volatility_std = 0.01,
    clusters = 3,
    regime_switch_prob = 0.05,
    seed = 42
  ):
    np.random.seed(seed)
    print(f"Generating {n_samples} synthetic volatility samples")
    volatility = np.zeros(n_samples)
    regime = np.zeros(n_samples,dtype = int)
    volatility[0] = volatility_mean
    alpha = 0.1
    beta = 0.85
    regime_params = {
      0: {'mean': 0.01, 'vol': 0.005},  
      1: {'mean': 0.025, 'vol': 0.015}, 
      2: {'mean': 0.04, 'vol': 0.025},  
    }
    for t in range(1, n_samples):
      if np.random.rand() < regime_switch_prob:
        regime[t] = np.random.randint(0, clusters)
      else:
        regime[t] = regime[t-1]
      regime_mean = regime_params[regime[t]]['mean']
      regime_vol = regime_params[regime[t]]['vol']
      shock = np.random.normal(0, regime_vol)
      volatility[t] = regime_mean + alpha * (shock ** 2) + beta * (volatility[t-1] - regime_mean)
      volatility[t] = max(0.001, volatility[t])
      if np.random.rand() < 0.02:
        volatility[t] = volatility[t] * np.random.uniform(1.5, 3.0)
    vix = 15 + (volatility * 100) * np.random.uniform(0.8, 1.2, size=n_samples)
    volume = 1e6 + (volatility * 2e7) * np.random.uniform(0.5, 1.5, size=n_samples)
    dates = pd.date_range(start='2015-01-01', periods=n_samples, freq='B') 
    data = pd.DataFrame({
      'date': dates,
      'volatility': volatility,
      'vix': vix,
      'volume': volume
    })
    print("Generated data:")
    print(f"- Mean volatility: {volatility.mean():.4f}")
    print(f"- Std volatility: {volatility.std():.4f}")
    print(f"- Min: {volatility.min():.4f}, Max: {volatility.max():.4f}")
    return data
  @staticmethod
  def load_yahoo_finance(ticker='AAPL', start_date='2020-01-01', end_date='2024-01-01', include_vix=True, include_volume=True):
    try:
      import yfinance as yf
    except ImportError:
      print("Error: yfinance not installed. Run: pip install yfinance")
      return None
    print(f"Loading {ticker} data from Yahoo Finance")
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
      high_col = data['High'].iloc[:, 0] if isinstance(data['High'], pd.DataFrame) else data['High']
      low_col = data['Low'].iloc[:, 0] if isinstance(data['Low'], pd.DataFrame) else data['Low']
      close_col = data['Close'].iloc[:, 0] if isinstance(data['Close'], pd.DataFrame) else data['Close']
      volume_col = data['Volume'].iloc[:, 0] if isinstance(data['Volume'], pd.DataFrame) else data['Volume']
      cols = {
        'High': high_col,
        'Low': low_col,
        'Close': close_col,
        'Volume': volume_col
      }
      intraday_return = (cols['High'] - cols['Low']) / cols['Close']
      volume_data = cols['Volume']
    else:
      intraday_return = (data['High'] - data['Low']) / data['Close']
      if 'Volume' in data.columns:
        volume_data = data['Volume']
      else:
        volume_data = None
    squared_return = intraday_return ** 2
    realized_vol = np.sqrt(squared_return)
    result_dict = {
      'date': data.index,
      'volatility': realized_vol.values
    }
    if include_volume and volume_data is not None:
      result_dict['volume'] = volume_data.values
    result = pd.DataFrame(result_dict)
    if include_vix:
      print("Loading VIX data from Yahoo Finance")
      vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False)
      if not vix_data.empty:
        if isinstance(vix_data.columns, pd.MultiIndex):
          vix_close = vix_data['Close'].iloc[:, 0] if isinstance(vix_data['Close'], pd.DataFrame) else vix_data['Close']
        else:
          vix_close = vix_data['Close']
        vix_df = pd.DataFrame({
          'date': vix_data.index,
          'vix': vix_close.values
        })
        result = pd.merge(result, vix_df, on='date', how='left')
        result['vix'] = result['vix'].ffill().bfill()
    print(f"Loaded {len(result)} trading days")
    return result
class DataValidator:
  @staticmethod
  def validate(data, target_column='volatility'):
    issues = {}
    if target_column not in data.columns:
      return False, {f"Column '{target_column}' not found"}
    vol = data[target_column].values
    nan_count = np.isnan(vol).sum()
    if nan_count > 0:
      issues['missing_values'] = nan_count
    inf_count = np.isinf(vol).sum()
    if inf_count > 0:
      issues['infinite_values'] = inf_count
    negative_count = (vol < 0).sum()
    if negative_count > 0:
      issues['negative_values'] = negative_count
    if np.std(vol) == 0:
      issues['zero_variance'] = True
    is_valid = len(issues) == 0
    if is_valid:
      print("Data validation PASSED")
    else:
      print(f"Data validation FAILED: {issues}")
    return is_valid, issues