import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add project root to sys.path to allow running this script directly
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.models.base import ClassicalModel
from src.config import Config

class ProphetModel(ClassicalModel):
    """
    Prophet (Facebook's forecasting library) for volatility.
    
    What it does:
    - Piecewise linear trend
    - Seasonality (Fourier series)
    - Holiday effects
    - Automatic changepoint detection
    
    Good for: Business metrics, traffic, sales
    Warning: Not designed for financial data (but we'll test it!)
    """
    
    def __init__(self, 
                 yearly_seasonality=False,
                 weekly_seasonality=False,
                 daily_seasonality=False,
                 changepoint_prior_scale=0.05):
        """
        Initialize Prophet model.
        
        Args:
            yearly_seasonality: Include yearly patterns
            weekly_seasonality: Include weekly patterns
            daily_seasonality: Include daily patterns
            changepoint_prior_scale: Flexibility of trend changes
        """
        super().__init__(name='Prophet')
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.fitted_model = None
    
    def fit(self, train_data: np.ndarray, exog: np.ndarray = None) -> None:
        """
        Train Prophet on historical volatility.
        
        Args:
            train_data: 1D array of training samples
            exog: 2D array of exogenous variables
        """
        super().fit(train_data, exog=exog)
        self.train_exog = exog
        
        try:
            from prophet import Prophet
        except ImportError:
            print("ERROR: prophet not installed. Run: pip install prophet")
            return
        
        print(f"Training {self.name}...")
        print(f"  - Yearly seasonality: {self.yearly_seasonality}")
        print(f"  - Weekly seasonality: {self.weekly_seasonality}")
        print(f"  - Daily seasonality: {self.daily_seasonality}")
        
        try:
            # Create DataFrame (Prophet requires 'ds' and 'y' columns)
            df = pd.DataFrame({
                'ds': pd.date_range(start='2015-01-01', periods=len(train_data), freq='B'),
                'y': train_data
            })
            
            if exog is not None:
                for i in range(exog.shape[1]):
                    df[f'regressor_{i}'] = exog[:, i]
            
            # Create and fit Prophet
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
            
            # Suppress Prophet's verbose output
            # Use 'NUL' on Windows, '/dev/null' on Linux
            null_device = 'NUL' if os.name == 'nt' else '/dev/null'
            with open(null_device, 'w') as devnull:
                import sys
                old_stdout = sys.stdout
                sys.stdout = devnull
                self.fitted_model = self.model.fit(df)
                sys.stdout = old_stdout
            
            print(f"[OK] {self.name} trained successfully")
            
        except Exception as e:
            print(f"[FAIL] Error training {self.name}: {str(e)}")
            self.is_fitted = False
    
    def forecast(self, horizon: int = 1, confidence_level: float = 0.95, exog: np.ndarray = None):
        """
        Forecast volatility using Prophet.
        
        Args:
            horizon: Steps ahead to forecast
            confidence_level: Confidence level (0.95 = 95%)
            exog: 2D array of exogenous variables for the forecast horizon
        
        Returns:
            Tuple of (predictions, (lower_bound, upper_bound))
        """
        super().forecast(horizon=horizon, confidence_level=confidence_level, exog=exog)
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=horizon, freq='B')
        
        if self.train_exog is not None:
            if exog is None or len(exog) < horizon:
                raise ValueError("exog matching horizon length is required for Prophet when trained with exog")
            for i in range(self.train_exog.shape[1]):
                full_exog = np.concatenate([self.train_exog[:, i], exog[:horizon, i]])
                future[f'regressor_{i}'] = full_exog
        
        # Forecast
        forecast = self.model.predict(future)
        
        # Get last 'horizon' predictions
        forecast = forecast.iloc[-horizon:]
        
        predictions = forecast['yhat'].values
        lower_bound = forecast['yhat_lower'].values
        upper_bound = forecast['yhat_upper'].values
        
        return predictions, (lower_bound, upper_bound)
    
    def get_params(self):
        """Return model hyperparameters."""
        return {
            'name': self.name,
            'yearly_seasonality': self.yearly_seasonality,
            'weekly_seasonality': self.weekly_seasonality,
            'daily_seasonality': self.daily_seasonality,
            'changepoint_prior_scale': self.changepoint_prior_scale,
            'is_fitted': self.is_fitted,
        }


# Test
if __name__ == "__main__":
    from src.data.loader import DataLoader
    from src.data.preprocessor import DataPreprocessor, DataSplitter
    
    print("Testing Prophet Model...")
    print("=" * 60)
    
    data = DataLoader.generate_synthetic_volatility(n_samples=1500)
    prep = DataPreprocessor()
    data = prep.handle_missing_values(data, 'volatility')
    data = prep.handle_outliers(data, 'volatility')
    
    vol = data['volatility'].values
    train, val, test = DataSplitter.train_val_test_split(vol)
    
    print("\nTraining...")
    prophet = ProphetModel(
        yearly_seasonality=Config.PROPHET_CONFIG['yearly_seasonality'],
        weekly_seasonality=Config.PROPHET_CONFIG['weekly_seasonality'],
        daily_seasonality=Config.PROPHET_CONFIG['daily_seasonality']
    )
    prophet.fit(train)
    
    print("\nForecasting...")
    predictions, (lower, upper) = prophet.forecast(horizon=10)
    
    print(f"Predictions (first 5): {predictions[:5]}")
    print(f"Lower bounds (first 5): {lower[:5]}")
    print(f"Upper bounds (first 5): {upper[:5]}")
    
    print(f"\nModel parameters: {prophet.get_params()}")
    print("[OK] Prophet test complete!")