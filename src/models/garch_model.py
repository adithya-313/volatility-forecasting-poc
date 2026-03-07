"""
GARCH (Generalized Autoregressive Conditional Heteroskedasticity) Model.

This is the absolute gold standard in finance for modeling volatility. 
While LSTMs try to learn everything from scratch, GARCH is mathematically
built specifically to understand how volatility "clusters" (high volatility
days group together, low volatility days group together).
"""

import numpy as np
import sys
from pathlib import Path

# Fix python path for local imports
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.models.base import ClassicalModel


class GARCHModel(ClassicalModel):
    """
    GARCH(1,1) model for volatility forecasting.
    It predicts tomorrow's variance based on:
    1. A long-term average
    2. Today's shock/surprise (the ARCH component)
    3. Today's variance (the GARCH component)
    """
    
    def __init__(self, p=1, q=1):
        """
        Initialize the GARCH model.
        
        Args:
            p: The number of recent shock terms to use (ARCH)
            q: The number of previous variance terms to use (GARCH)
        """
        super().__init__(name=f'GARCH({p},{q})')
        self.p = p
        self.q = q
        self.model = None
        self.fitted_model = None
    
    def fit(self, train_data: np.ndarray, exog: np.ndarray = None) -> None:
        """
        Train the GARCH model on historical returns/volatility.
        
        Args:
            train_data: 1D array of training samples (usually returns or volatility).
            exog: 2D array of external variables (like VIX, Volume)
        """
        super().fit(train_data, exog=exog)
        
        # We need the 'arch' library to use GARCH easily in Python
        try:
            from arch import arch_model
        except ImportError:
            print("ERROR: 'arch' library not installed. Run: pip install arch")
            self.is_fitted = False
            return
            
        print(f"Training {self.name}...")
        
        try:
            # Create the model using the ARCH library
            # 'GARCH' is the variance model. We assume a constant mean.
            self.model = arch_model(
                train_data,
                x=exog,        # Provide exogenous variables if we have them
                mean='Constant',
                vol='Garch',
                p=self.p,
                q=self.q
            )
            
            # Fit the model to the data without printing a massive wall of text
            self.fitted_model = self.model.fit(disp='off')
            
            # Store the errors (residuals) to help calculate confidence intervals later
            self.residuals = self.fitted_model.resid
            
            print(f"[OK] {self.name} trained successfully")
            print(f"  - AIC: {self.fitted_model.aic:.2f}")
            print(f"  - BIC: {self.fitted_model.bic:.2f}")
            
        except Exception as e:
            print(f"[FAIL] Error training {self.name}: {str(e)}")
            self.is_fitted = False
    
    def forecast(self, horizon: int = 1, confidence_level: float = 0.95, exog: np.ndarray = None):
        """
        Predict future volatility.
        
        Args:
            horizon: How many days into the future to predict
            confidence_level: How sure we want to be (e.g., 0.95 means 95% confident)
            exog: External variables for the future days
        
        Returns:
            Tuple: (predictions, (lower_bound, upper_bound))
        """
        super().forecast(horizon=horizon, confidence_level=confidence_level, exog=exog)
        
        # Ask the fitted model to predict 'horizon' days into the future
        forecast_result = self.fitted_model.forecast(horizon=horizon, x=exog)
        
        # GARCH predicts variance (volatility squared), so we must take the square root!
        # The result matrix puts the forecasts in the very last row.
        predicted_variance = forecast_result.variance.values[-1, :]
        predictions = np.sqrt(predicted_variance)
        
        # To calculate confidence bands, we use some basic statistics
        # (Standard Error = Variance / sqrt(Number of Samples))
        # Since we are predicting variance itself, we create bands using the residual standard deviation
        
        residual_std = np.std(self.residuals)
        
        # Margin of error (Roughly 1.96 standard deviations for 95% confidence)
        margin = 1.96 * residual_std
        
        lower_bound = np.maximum(predictions - margin, 0.001) # Can't be negative
        upper_bound = predictions + margin
        
        return predictions, (lower_bound, upper_bound)
    
    def get_params(self):
        """Show the current settings of the model."""
        return {
            'name': self.name,
            'p_lags': self.p,
            'q_lags': self.q,
            'is_fitted': self.is_fitted,
            'aic': self.fitted_model.aic if self.is_fitted else None,
        }


# A quick test to make sure it works if someone runs this file directly
if __name__ == "__main__":
    from src.data.loader import DataLoader
    from src.data.preprocessor import DataSplitter
    
    print(f"Testing {GARCHModel(1,1).name} Model...")
    print("=" * 60)
    
    # Generate some fake stock market data
    data = DataLoader.generate_synthetic_volatility(n_samples=1000)
    volatility = data['volatility'].values
    
    # Split into train and test
    train, val, test = DataSplitter.train_val_test_split(volatility)
    
    # Train
    garch = GARCHModel(p=1, q=1)
    garch.fit(train)
    
    # Predict
    if garch.is_fitted:
        print("\nPredicting tomorrow and the next 4 days...")
        predictions, (lower, upper) = garch.forecast(horizon=5)
        
        print(f"Predictions: {predictions}")
        print(f"Lower Bounds: {lower}")
        print(f"Upper Bounds: {upper}")
        print("\n[OK] GARCH model test complete!")
