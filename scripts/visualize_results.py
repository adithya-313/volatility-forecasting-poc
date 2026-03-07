"""
Final Phase 3 Visualization: Forecast Comparison.

This script runs the walk-forward evaluation and generates a beautiful
3-panel chart showing how each model performed against reality.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Fix python path for local imports
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.evaluation.walk_forward import WalkForwardValidator

# Import models
from src.models.sarima_model import SARIMAModel
from src.models.garch_model import GARCHModel
from src.models.prophet_model import ProphetModel
from src.models.lstm_model import LSTMModel
from src.models.lstm_advanced import AdvancedLSTMModel

def main():
    print("\n" + "="*80)
    print("PHASE 3 VISUALIZATION ENGINE")
    print("Generating Comparative Forecast Plots")
    print("="*80)

    # 1. Create docs folder if it doesn't exist
    docs_dir = root_dir / 'docs'
    docs_dir.mkdir(exist_ok=True)
    
    # -------------------------------------------------------------------------
    # 2. DATA PREPARATION (Fast Train/Test Split)
    # -------------------------------------------------------------------------
    print("\n[1/3] Preparing data...")
    loader = DataLoader()
    data = loader.load_yahoo_finance(
        ticker='AAPL',
        start_date='2021-01-01',
        end_date='2024-01-01',
        include_vix=True
    )
    
    prep = DataPreprocessor()
    data = data.dropna()
    
    # Target: Volatility
    y = data['volatility'].values
    
    # Exogenous: Scaled VIX
    exog_vix = data['vix'].values.reshape(-1, 1)
    exog_scaled, _, _ = prep.scale_features(X_train=exog_vix)
    
    # Split index (last 40 days only for fast plot)
    split_idx = len(y) - 40
    y_train, y_test = y[:split_idx], y[split_idx:]
    exog_train, exog_test = exog_scaled[:split_idx], exog_scaled[split_idx:]
    
    # -------------------------------------------------------------------------
    # 3. FIT AND FORECAST
    # -------------------------------------------------------------------------
    print(f"\n[2/3] Fitting and Forecasting on {len(y_test)} test days...")
    
    models = {
        'SARIMA': SARIMAModel(order=(1, 1, 1)),
        'Prophet': ProphetModel(),
        'GARCH(1,1)': GARCHModel(1, 1),
        'LSTM (Basic)': LSTMModel(epochs=5),
        'Advanced LSTM (VIX)': AdvancedLSTMModel(epochs=5)
    }
    
    predictions = {}
    horizon = len(y_test)
    
    for name, model in models.items():
        print(f"  [START] {name} fitting...")
        try:
            if name == 'Advanced LSTM (VIX)':
                model.fit(y_train, exog=exog_train)
                print(f"  [DONE] {name} fitted. Forecasting...")
                preds = model.forecast(horizon=horizon, exog=exog_test)
            elif name in ['SARIMA', 'Prophet']:
                model.fit(y_train, exog=exog_train)
                print(f"  [DONE] {name} fitted. Forecasting...")
                preds = model.forecast(horizon=horizon, exog=exog_test)
            else:
                model.fit(y_train)
                print(f"  [DONE] {name} fitted. Forecasting...")
                preds = model.forecast(horizon=horizon)
            
            predictions[name] = preds
            print(f"  [FINISHED] {name} results stored.")
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")

    # -------------------------------------------------------------------------
    # 4. PLOTTING
    # -------------------------------------------------------------------------
    print("\n[3/3] Generating final plot...")
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(15, 8))
    
    time_pts = np.arange(len(y_test))
    plt.plot(time_pts, y_test, label='Actual AAPL Volatility', color='black', linewidth=3, alpha=0.9)
    
    # Palette for models
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for (name, preds), color in zip(predictions.items(), colors):
        plt.plot(time_pts, preds, label=f'{name} Forecast', linewidth=2, linestyle='--', color=color)

    plt.title('Final Model Comparison: Predicted vs Actual Volatility (AAPL)', fontsize=18, fontweight='bold')
    plt.xlabel('Trading Days (Test Set)', fontsize=14)
    plt.ylabel('Realized Volatility', fontsize=14)
    plt.legend(prop={'size': 12}, loc='upper left')
    
    # Save the plot
    docs_dir = root_dir / 'docs'
    docs_dir.mkdir(exist_ok=True)
    save_path = docs_dir / 'final_forecast_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"\nDONE! Visualization saved to:\n      {save_path}")
    print("="*80)
    
    plt.tight_layout()
    
    # -------------------------------------------------------------------------
    # 5. SAVE
    # -------------------------------------------------------------------------
    save_path = docs_dir / 'phase3_forecast_comparison.png'
    plt.savefig(save_path, dpi=300)
    print(f"\n[4/4] Success! Visualization saved to:\n      {save_path}")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
