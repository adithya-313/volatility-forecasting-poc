import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
class DataPreprocessor:
  def __init__(self, random_seed=42):
    self.random_seed = random_seed
    np.random.seed(random_seed)
    self.scaler = StandardScaler()
    self.is_fitted = False
  def handle_missing_values(self, data, column):
    data = data.copy()
    missing_before = data[column].isna().sum()
    if missing_before == 0:
      print(f"No missing values in '{column}'")
      return data
    data[column].fillna(method='ffill', inplace=True)
    data[column].fillna(method='bfill', inplace=True)
    missing_after = data[column].isna().sum()
    print(f"Missing values handled: {missing_before} {missing_after}")
    return data
  def handle_outliers(self, data, column, method='iqr'):
    data = data.copy()
    if method == 'iqr':
      Q1 = data[column].quantile(0.25)
      Q3 = data[column].quantile(0.75)
      IQR = Q3 - Q1
      lower = Q1 - 1.5 * IQR
      upper = Q3 + 1.5 * IQR
    elif method == 'zscore':
      mean = data[column].mean()
      std = data[column].std()
      lower = mean - 3 * std
      upper = mean + 3 * std
    else:
      raise ValueError(f"Unknown method: {method}")
    outlier_mask = (data[column] < lower) | (data[column] > upper)
    outlier_count = outlier_mask.sum()
    data[column] = data[column].clip(lower=lower, upper=upper)
    print(f"Outliers handled ({method}): {outlier_count} detected and clipped")
    print(f"- Lower bound: {lower:.6f}, Upper bound: {upper:.6f}")
    return data
  def compute_lagged_features(self, data, column, n_lags=5):
    data = data.copy()
    for lag in range(1, n_lags + 1):
      data[f'{column}_lag_{lag}'] = data[column].shift(lag)
    data = data.dropna().reset_index(drop=True)
    print(f"Created {n_lags} lagged features")
    print(f"- New shape: {data.shape}")
    return data
  def scale_features(self, X_train, X_val=None, X_test=None):
    if X_train.ndim == 1:
      X_train = X_train.reshape(-1, 1)
    if X_val is not None and X_val.ndim == 1:
      X_val = X_val.reshape(-1, 1)
    if X_test is not None and X_test.ndim == 1:
      X_test = X_test.reshape(-1, 1)
    X_train_scaled = self.scaler.fit_transform(X_train)
    self.is_fitted = True
    X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None
    X_test_scaled = self.scaler.transform(X_test) if X_test is not None else None
    print("Features scaled (fit on train only)")
    print(f"- Train: mean={X_train_scaled.mean():.4f}, std={X_train_scaled.std():.4f}")
    return X_train_scaled, X_val_scaled, X_test_scaled
  def create_sequences(self, data, seq_length=30):
    X, y = [], []
    for i in range(len(data) - seq_length):
      X.append(data[i:i+seq_length])
      y.append(data[i+seq_length])
    X = np.array(X)
    y = np.array(y)
    print("Created sequences:")
    print(f"- Shape: {X.shape} (samples, sequence_length)")
    print(f"- Total sequences: {len(X)}")
    return X, y
class DataSplitter:
  @staticmethod
  def train_val_test_split(*arrays, train_ratio=0.6, val_ratio=0.2):
    if not arrays:
      raise ValueError("At least one array required as input")
    n = len(arrays[0])
    for arr in arrays:
      if len(arr) != n:
        raise ValueError("All arrays must have the same length")
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    result = []
    for arr in arrays:
      if isinstance(arr, pd.DataFrame) or isinstance(arr, pd.Series):
        train = arr.iloc[:train_end]
        val = arr.iloc[train_end:val_end]
        test = arr.iloc[val_end:]
      else:
        train = arr[:train_end]
        val = arr[train_end:val_end]
        test = arr[val_end:]
      result.extend([train, val, test])
    print("Data split (time-aware):")
    print(f"- Train: {train_end} samples ({train_ratio*100:.0f}%)")
    print(f"- Val: {val_end - train_end} samples ({val_ratio*100:.0f}%)")
    print(f"- Test: {n - val_end} samples ({(1 - train_ratio - val_ratio)*100:.0f}%)")
    if len(arrays) == 1:
      return result[0], result[1], result[2]
    return tuple(result)