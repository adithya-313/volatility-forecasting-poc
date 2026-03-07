import os
from pathlib import Path

class Config:
    """
    Centralized configuration for the volatility forecasting project.
    """
    
    # ========== RANDOM SEED (For reproducibility) ==========
    RANDOM_SEED = 42
    
    # ========== PROJECT PATHS ==========
    PROJECT_ROOT = Path(__file__).parent.parent  # Root directory
    DATA_DIR = PROJECT_ROOT / 'data'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'
    SPLITS_DIR = DATA_DIR / 'splits'
    
    # Create directories if they don't exist
    for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SPLITS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # ========== DATA CONFIGURATION ==========
    DATA_CONFIG = {
        'n_samples': 1500,              # Number of data points to generate
        'volatility_mean': 0.02,        # Average volatility
        'volatility_std': 0.01,         # Standard deviation
        'clusters': 3,                  # Number of market regimes
        'regime_switch_prob': 0.05,     # Probability of regime change
    }
    
    # ========== PREPROCESSING CONFIGURATION ==========
    PREPROCESSING_CONFIG = {
        'missing_method': 'forward_fill',  # How to handle missing values
        'outlier_method': 'iqr',           # How to detect outliers
        'scale_method': 'standard',        # Standardization method
    }
    
    # ========== DATA SPLITTING CONFIGURATION ==========
    SPLIT_CONFIG = {
        'train_ratio': 0.6,    # 60% for training
        'val_ratio': 0.2,      # 20% for validation
        # Rest (20%) goes to test
    }
    
    # ========== MODEL HYPERPARAMETERS ==========
    
    # SARIMA
    SARIMA_CONFIG = {
        'order': (2, 1, 2),           # (p, d, q)
        'seasonal_order': (1, 1, 1, 252),  # (P, D, Q, s)
    }
    
    # Exponential Smoothing
    EXP_SMOOTHING_CONFIG = {
        'trend': 'add',                # Additive trend
        'seasonal': None,              # No seasonality
    }
    
    # Prophet
    PROPHET_CONFIG = {
        'yearly_seasonality': False,
        'weekly_seasonality': False,
        'daily_seasonality': False,
    }
    
    # LSTM
    LSTM_CONFIG = {
        'sequence_length': 30,         # Input window size
        'lstm_units': (64, 32),        # Units in LSTM layers
        'dropout': 0.2,                # Dropout rate
        'epochs': 100,                 # Number of training epochs
        'batch_size': 32,              # Batch size
        'learning_rate': 0.001,        # Learning rate
    }
    
    # ========== EVALUATION CONFIGURATION ==========
    EVALUATION_CONFIG = {
        'confidence_level': 0.95,      # 95% confidence intervals
        'test_significance_level': 0.05,  # α = 0.05
    }


# Test the configuration
if __name__ == "__main__":
    print("✓ Configuration loaded successfully")
    print(f"Project root: {Config.PROJECT_ROOT}")
    print(f"Data directory: {Config.DATA_DIR}")
    print(f"Random seed: {Config.RANDOM_SEED}")