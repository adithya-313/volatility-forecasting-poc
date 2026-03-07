"""
Phase 3 Final Evaluation: Walk-Forward Validation & Multivariate Forecasting.

This script runs the ultimate test:
1. Uses Real Data (AAPL stock)
2. Uses Multivariate Inputs (VIX index as an extra context variable)
3. Uses Walk-Forward Validation (simulating real-world weekly Retraining)
"""

import sys
import numpy as np
from pathlib import Path

# Fix python path for local imports
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.evaluation.walk_forward import WalkForwardValidator

# Import all our models
from src.models.sarima_model import SARIMAModel
from src.models.garch_model import GARCHModel
from src.models.prophet_model import ProphetModel
from src.models.lstm_model import LSTMModel
from src.models.lstm_advanced import AdvancedLSTMModel

def main():
    print("\n" + "="*80)
    print("PHASE 3 PRODUCTION EVALUATION")
    print("Walk-Forward Validation with Multivariate Data")
    print("="*80)
    
    # -------------------------------------------------------------------------
    # 1. LOAD DATA
    # -------------------------------------------------------------------------
    print("\n[STEP 1/4] Loading real data (AAPL Volatility + VIX Index)...")
    try:
        # Load the last 3 years of daily trading data
        data = DataLoader.load_yahoo_finance(
            ticker='AAPL',
            start_date='2021-01-01',
            end_date='2024-01-01',
            include_vix=True,     # <--- MULTIVARIATE: Pull the VIX!
            include_volume=False  # Keep it simple, just VIX for now
        )
        print(f"[OK] Loaded {len(data)} trading days")
    except Exception as e:
        print(f"[FAIL] Could not load Yahoo Finance data: {e}")
        print("Please check your internet connection or run pip install yfinance")
        return
        
    # -------------------------------------------------------------------------
    # 2. PREPROCESS DATA
    # -------------------------------------------------------------------------
    print("\n[STEP 2/4] Preprocessing data...")
    prep = DataPreprocessor()
    
    # Handle any weird missing or extreme values in Volatility
    data = prep.handle_missing_values(data, 'volatility')
    data = prep.handle_outliers(data, 'volatility')
    
    # Handle any weird missing or extreme values in VIX
    data = prep.handle_missing_values(data, 'vix')
    data = prep.handle_outliers(data, 'vix')
    
    # Extract the pure numbers for our models
    volatility = data['volatility'].values
    vix = data['vix'].values.reshape(-1, 1) # Models like 2D arrays for extra features
    
    # Neural Networks need extra features to be scaled (e.g., between -1 and 1)
    # The VIX is usually around 15-30, while Volatility is 0.01-0.05.
    # We MUST scale VIX so the Neural Network doesn't get confused by the big numbers.
    vix_scaled, _, _ = prep.scale_features(X_train=vix)
    
    # -------------------------------------------------------------------------
    # 3. SETUP MODELS
    # -------------------------------------------------------------------------
    print("\n[STEP 3/4] Initializing Models...")
    models_to_test = [
        # Baseline
        GARCHModel(p=1, q=1),
        
        # Classical
        SARIMAModel(order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)),
        ProphetModel(changepoint_prior_scale=0.1),
        
        # Deep Learning (Univariate - No VIX allowed!)
        LSTMModel(
            sequence_length=15, 
            lstm_units=(64, 32), 
            epochs=20, 
            batch_size=16
        ),
        
        # Deep Learning (Multivariate - Looks at Volatility AND the VIX!)
        AdvancedLSTMModel(
            sequence_length=15, 
            lstm_units=(128, 64), 
            epochs=30, 
            batch_size=16,
            dropout=0.3
        )
    ]
    
    # -------------------------------------------------------------------------
    # 4. RUN WALK-FORWARD VALIDATION
    # -------------------------------------------------------------------------
    print("\n[STEP 4/4] Running simulated production evaluation...")
    
    # We will pretend it's the 600th day of the dataset.
    # Every 10 days, we stop, predict the next 10 days, calculate our error, 
    # and then re-train the models with the new data.
    validator = WalkForwardValidator(
        initial_train_size=600, 
        step_size=10, 
        window_type='expanding'
    )
    
    final_results = {}
    
    for model in models_to_test:
        try:
            # The Advanced LSTM uses the VIX data. We pass it using the `exog` argument.
            if model.name == 'AdvancedLSTM':
                # Pass the scaled VIX data
                result = validator.evaluate(model, volatility, exog=vix_scaled)
                
            # The other models are either strictly univariate, or don't benefit 
            # as easily from the scaled VIX without complex code changes, so we run them normally.
            else:
                result = validator.evaluate(model, volatility, exog=None)
                
            final_results[model.name] = result['mae']
            
        except Exception as e:
            print(f"\n[FAIL] Testing {model.name} crashed: {e}")
            final_results[model.name] = float('inf')
    
    # -------------------------------------------------------------------------
    # 5. PRINT THE PODIUM
    # -------------------------------------------------------------------------
    print("\n\n" + "="*80)
    print("FINAL RANKINGS (Mean Absolute Error - Lower is Better)")
    print("="*80)
    
    # Sort the dictionary purely by the score (lowest first)
    sorted_results = sorted(final_results.items(), key=lambda x: x[1])
    
    for rank, (model_name, mae) in enumerate(sorted_results):
        medal = " (1st)" if rank == 0 else " (2nd)" if rank == 1 else " (3rd)" if rank == 2 else "      "
        if mae == float('inf'):
            print(f"{rank+1}. {model_name:20}: FAILED")
        else:
            print(f"{rank+1}. {model_name:20}: {mae:.6f} {medal}")
    
    print("\n" + "="*80)
    print("WHY DO THESE SCORES LOOK DIFFERENT?")
    print("Walk-forward validation scores are usually slightly 'worse' than standard")
    print("Train/Test splits because they capture the chaotic nature of the real market")
    print("shifting day-to-day. This is the TRUE performance you can expect in production.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
