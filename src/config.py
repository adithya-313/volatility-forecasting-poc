import os
from pathlib import Path
class Config:
  RANDOM_SEED = 42
  PROJECT_ROOT = Path(__file__).parent.parent 
  DATA_DIR = PROJECT_ROOT / 'data'
  RAW_DATA_DIR = DATA_DIR / 'raw'
  PROCESSED_DATA_DIR = DATA_DIR / 'processed'
  SPLITS_DIR = DATA_DIR / 'splits'
  for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SPLITS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
  DATA_CONFIG = {
    'n_samples': 1500,       
    'volatility_mean': 0.02,    
    'volatility_std': 0.01,    
    'clusters': 3,         
    'regime_switch_prob': 0.05,  
  }
  PREPROCESSING_CONFIG = {
    'missing_method': 'forward_fill', 
    'outlier_method': 'iqr',     
    'scale_method': 'standard',    
  }
  SPLIT_CONFIG = {
    'train_ratio': 0.6,  
    'val_ratio': 0.2,   
  }
  SARIMA_CONFIG = {
    'order': (2, 1, 2),     
    'seasonal_order': (1, 1, 1, 252), 
  }
  EXP_SMOOTHING_CONFIG = {
    'trend': 'add',        
    'seasonal': None,       
  }
  PROPHET_CONFIG = {
    'yearly_seasonality': False,
    'weekly_seasonality': False,
    'daily_seasonality': False,
  }
  LSTM_CONFIG = {
    'sequence_length': 30,    
    'lstm_units': (64, 32),    
    'dropout': 0.2,        
    'epochs': 100,        
    'batch_size': 32,       
    'learning_rate': 0.001,    
  }
  EVALUATION_CONFIG = {
    'confidence_level': 0.95,   
    'test_significance_level': 0.05, 
  }
if __name__ == "__main__":
  print("Configuration loaded")
  print(f"Project root: {Config.PROJECT_ROOT}")
  print(f"Data directory: {Config.DATA_DIR}")
  print(f"Random seed: {Config.RANDOM_SEED}")