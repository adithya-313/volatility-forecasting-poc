import numpy as np
import sys
import os
from pathlib import Path

# Add project root to sys.path to allow running this script directly
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.models.base import ClassicalModel
from src.config import Config

class ExponentialSmoothingModel(ClassicalModel):
    """
    Exponential Smoothing (Holt-Winters) for volatility forecasting.
    
    What it does:
    - Simple exponential smoothing of past values
    - Captures level and trend
    - Good for data with patterns but no complex seasonality
    
    Formula: volatility_t = α*volatility_{t-1} + (1-α)*prediction_{t-1}
    """
    
    def __init__(self, trend='add', seasonal=None, seasonal_periods=252):
        """
        Initialize Exponential Smoothing model.
        
        Args:
            trend: 'add' (additive) or 'mul' (multiplicative)
            seasonal: 'add', 'mul', or None
            seasonal_periods: 252 = trading days in year
        """
        super().__init__(name='ExponentialSmoothing')
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.fitted_model = None
    
    def fit(self, train_data: np.ndarray) -> None:
        """
        Train Exponential Smoothing on historical volatility.
        
        Args:
            train_data: 1D array of training samples
        """
        super().fit(train_data)
        
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
        except ImportError:
            print("ERROR: statsmodels not installed")
            return
        
        print(f"Training {self.name}...")
        print(f"  - Trend: {self.trend}")
        print(f"  - Seasonal: {self.seasonal}")
        
        try:
            # Create and fit model
            self.model = ExponentialSmoothing(
                train_data,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods if self.seasonal else None
            )
            
            self.fitted_model = self.model.fit(optimized=True)
            
            # Calculate residuals
            resid = self.fitted_model.resid
            self.residuals = resid.values if hasattr(resid, 'values') else resid
            
            print(f"[OK] {self.name} trained successfully")
            print(f"  - SSE: {self.fitted_model.sse:.4f}")
            
        except Exception as e:
            print(f"[FAIL] Error training {self.name}: {str(e)}")
            self.is_fitted = False
    
    def forecast(self, horizon: int = 1, confidence_level: float = 0.95):
        """
        Forecast volatility.
        
        Args:
            horizon: Steps ahead to forecast
            confidence_level: Confidence level (0.95 = 95%)
        
        Returns:
            Tuple of (predictions, (lower_bound, upper_bound))
        """
        super().forecast(horizon=horizon, confidence_level=confidence_level)
        
        # Get point forecasts
        predictions = self.fitted_model.forecast(steps=horizon)
        
        # Estimate confidence intervals using residual std
        residual_std = np.std(self.residuals)
        z_score = 1.96  # For 95% confidence
        
        lower_bound = predictions - z_score * residual_std
        upper_bound = predictions + z_score * residual_std
        
        # Safely convert to numpy arrays if they are pandas objects
        res_predictions = predictions.values if hasattr(predictions, 'values') else predictions
        res_lower = lower_bound.values if hasattr(lower_bound, 'values') else lower_bound
        res_upper = upper_bound.values if hasattr(upper_bound, 'values') else upper_bound
        
        return res_predictions, (res_lower, res_upper)
    
    def get_params(self):
        """Return model hyperparameters."""
        return {
            'name': self.name,
            'trend': self.trend,
            'seasonal': self.seasonal,
            'seasonal_periods': self.seasonal_periods,
            'is_fitted': self.is_fitted,
            'sse': self.fitted_model.sse if self.is_fitted else None,
        }


# Test
if __name__ == "__main__":
    from src.data.loader import DataLoader
    from src.data.preprocessor import DataPreprocessor, DataSplitter
    
    print("Testing Exponential Smoothing Model...")
    print("=" * 60)
    
    data = DataLoader.generate_synthetic_volatility(n_samples=1500)
    prep = DataPreprocessor()
    data = prep.handle_missing_values(data, 'volatility')
    data = prep.handle_outliers(data, 'volatility')
    
    vol = data['volatility'].values
    train, val, test = DataSplitter.train_val_test_split(vol)
    
    print("\nTraining...")
    es = ExponentialSmoothingModel(
        trend=Config.EXP_SMOOTHING_CONFIG['trend'],
        seasonal=Config.EXP_SMOOTHING_CONFIG['seasonal']
    )
    es.fit(train)
    
    print("\nForecasting...")
    predictions, (lower, upper) = es.forecast(horizon=10)
    
    print(f"Predictions (first 5): {predictions[:5]}")
    print(f"Lower bounds (first 5): {lower[:5]}")
    print(f"Upper bounds (first 5): {upper[:5]}")
    
    print(f"\nModel parameters: {es.get_params()}")
    print("[OK] Exponential Smoothing test complete!")