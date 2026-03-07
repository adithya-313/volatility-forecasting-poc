"""
Hyperparameter Optimization using Optuna.

Optimizes all 4 models on real AAPL volatility data.
Uses Bayesian optimization to find best hyperparameters.
"""

import numpy as np
import optuna
from optuna.pruners import MedianPruner
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor, DataSplitter
from src.models.sarima_model import SARIMAModel
from src.models.exp_smoothing_model import ExponentialSmoothingModel
from src.models.prophet_model import ProphetModel
from src.models.lstm_advanced import AdvancedLSTMModel
from src.evaluation.metrics import VolatilityMetrics


class HyperparameterOptimizer:
    """Optimize hyperparameters for all models."""
    
    def __init__(self, train_data, val_data, test_data):
        """Initialize optimizer."""
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.metrics = VolatilityMetrics()
    
    def optimize_sarima(self, n_trials=20):
        """Optimize SARIMA hyperparameters."""
        
        print("\n" + "="*80)
        print("OPTIMIZING SARIMA HYPERPARAMETERS")
        print("="*80)
        
        def objective(trial):
            """Objective function for Optuna."""
            
            try:
                # Suggest hyperparameters
                p = trial.suggest_int('p', 1, 3)
                d = trial.suggest_int('d', 0, 2)
                q = trial.suggest_int('q', 1, 3)
                
                print(f"\nTrial {trial.number}: SARIMA({p},{d},{q})")
                
                # Train model
                model = SARIMAModel(
                    order=(p, d, q),
                    seasonal_order=(0, 0, 0, 0)
                )
                model.fit(self.train_data)
                
                # Evaluate on validation set
                predictions, _ = model.forecast(horizon=len(self.val_data))
                predictions = predictions[:len(self.val_data)]
                
                # Calculate MAE (lower is better)
                mae = self.metrics.mae(self.val_data, predictions)
                print(f"  MAE: {mae:.6f}")
                
                return mae
            
            except Exception as e:
                print(f"  Trial failed: {str(e)}")
                return float('inf')
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            pruner=MedianPruner()
        )
        
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"\n[OK] Best SARIMA:")
        print(f"  Parameters: {study.best_params}")
        print(f"  MAE: {study.best_value:.6f}")
        
        return study.best_params, study.best_value
    
    def optimize_exp_smoothing(self, n_trials=10):
        """Optimize Exponential Smoothing hyperparameters."""
        
        print("\n" + "="*80)
        print("OPTIMIZING EXPONENTIAL SMOOTHING HYPERPARAMETERS")
        print("="*80)
        
        def objective(trial):
            """Objective function for Optuna."""
            
            try:
                # Suggest hyperparameters
                trend = trial.suggest_categorical('trend', ['add', 'mul'])
                seasonal = trial.suggest_categorical('seasonal', [None, 'add', 'mul'])
                
                print(f"\nTrial {trial.number}: trend={trend}, seasonal={seasonal}")
                
                # Train model
                model = ExponentialSmoothingModel(
                    trend=trend,
                    seasonal=seasonal
                )
                model.fit(self.train_data)
                
                # Evaluate on validation set
                predictions, _ = model.forecast(horizon=len(self.val_data))
                predictions = predictions[:len(self.val_data)]
                
                # Calculate MAE
                mae = self.metrics.mae(self.val_data, predictions)
                print(f"  MAE: {mae:.6f}")
                
                return mae
            
            except Exception as e:
                print(f"  Trial failed: {str(e)}")
                return float('inf')
        
        # Create study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"\n[OK] Best Exponential Smoothing:")
        print(f"  Parameters: {study.best_params}")
        print(f"  MAE: {study.best_value:.6f}")
        
        return study.best_params, study.best_value
    
    def optimize_prophet(self, n_trials=10):
        """Optimize Prophet hyperparameters."""
        
        print("\n" + "="*80)
        print("OPTIMIZING PROPHET HYPERPARAMETERS")
        print("="*80)
        
        def objective(trial):
            """Objective function for Optuna."""
            
            try:
                # Suggest hyperparameters
                changepoint_scale = trial.suggest_float('changepoint_scale', 0.01, 0.5)
                seasonality_scale = trial.suggest_float('seasonality_scale', 1.0, 20.0)
                
                print(f"\nTrial {trial.number}: changepoint={changepoint_scale:.4f}, "
                      f"seasonality={seasonality_scale:.2f}")
                
                # Train model
                model = ProphetModel(
                    yearly_seasonality=False,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    changepoint_prior_scale=changepoint_scale
                )
                # Note: Prophet doesn't expose seasonality_prior_scale easily
                model.fit(self.train_data)
                
                # Evaluate on validation set
                predictions, _ = model.forecast(horizon=len(self.val_data))
                predictions = predictions[:len(self.val_data)]
                
                # Calculate MAE
                mae = self.metrics.mae(self.val_data, predictions)
                print(f"  MAE: {mae:.6f}")
                
                return mae
            
            except Exception as e:
                print(f"  Trial failed: {str(e)}")
                return float('inf')
        
        # Create study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"\n[OK] Best Prophet:")
        print(f"  Parameters: {study.best_params}")
        print(f"  MAE: {study.best_value:.6f}")
        
        return study.best_params, study.best_value
    
    def optimize_lstm(self, n_trials=20):
        """Optimize Advanced LSTM hyperparameters."""
        
        print("\n" + "="*80)
        print("OPTIMIZING ADVANCED LSTM HYPERPARAMETERS")
        print("="*80)
        
        def objective(trial):
            """Objective function for Optuna."""
            
            try:
                # Suggest hyperparameters
                lstm_units_1 = trial.suggest_int('lstm_units_1', 64, 256, step=32)
                lstm_units_2 = trial.suggest_int('lstm_units_2', 32, 128, step=32)
                dropout = trial.suggest_float('dropout', 0.1, 0.5)
                learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01, log=True)
                batch_size = trial.suggest_int('batch_size', 8, 64, step=8)
                
                print(f"\nTrial {trial.number}: units=({lstm_units_1},{lstm_units_2}), "
                      f"dropout={dropout:.2f}, lr={learning_rate:.6f}, bs={batch_size}")
                
                # Train model
                model = AdvancedLSTMModel(
                    sequence_length=30,
                    lstm_units=(lstm_units_1, lstm_units_2),
                    dropout=dropout,
                    epochs=50,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    device='cpu'
                )
                model.fit(self.train_data, val_data=self.val_data)
                
                # Evaluate on validation set
                predictions, _ = model.forecast(
                    recent_data=self.val_data,
                    horizon=min(5, len(self.val_data))
                )
                
                # Calculate MAE
                mae = self.metrics.mae(
                    self.val_data[:len(predictions)],
                    predictions
                )
                print(f"  MAE: {mae:.6f}")
                
                return mae
            
            except Exception as e:
                print(f"  Trial failed: {str(e)}")
                return float('inf')
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            pruner=MedianPruner()
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"\n[OK] Best Advanced LSTM:")
        print(f"  Parameters: {study.best_params}")
        print(f"  MAE: {study.best_value:.6f}")
        
        return study.best_params, study.best_value
    
    def run_all_optimizations(self):
        """Run optimization for all models."""
        
        results = {}
        
        # Optimize each model
        results['SARIMA'] = self.optimize_sarima(n_trials=15)
        results['ExponentialSmoothing'] = self.optimize_exp_smoothing(n_trials=10)
        results['Prophet'] = self.optimize_prophet(n_trials=10)
        results['AdvancedLSTM'] = self.optimize_lstm(n_trials=15)
        
        return results


def main():
    """Main optimization pipeline."""
    
    print("\n" + "="*80)
    print("HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
    print("="*80)
    
    # Load real data
    print("\n[STEP 1/3] Loading real AAPL data...")
    try:
        data = DataLoader.load_yahoo_finance(
            ticker='AAPL',
            start_date='2018-01-01',
            end_date='2024-01-01'
        )
        print(f"[OK] Loaded {len(data)} trading days")
    except Exception as e:
        print(f"[FAIL] Error: {str(e)}")
        print("Using synthetic data instead...")
        data = DataLoader.generate_synthetic_volatility(n_samples=1500)
    
    # Preprocess
    print("\n[STEP 2/3] Preprocessing data...")
    prep = DataPreprocessor()
    data = prep.handle_missing_values(data, 'volatility')
    data = prep.handle_outliers(data, 'volatility')
    volatility_series = data['volatility'].values
    
    # Split
    print("\n[STEP 3/3] Splitting data...")
    train, val, test = DataSplitter.train_val_test_split(volatility_series)
    print(f"[OK] Split: train={len(train)}, val={len(val)}, test={len(test)}")
    
    # Run optimizations
    print("\n[STEP 4/3] Running Optuna optimizations...\n")
    optimizer = HyperparameterOptimizer(train, val, test)
    results = optimizer.run_all_optimizations()
    
    # Summary
    print("\n" + "="*80)
    print("OPTIMIZATION SUMMARY")
    print("="*80)
    
    print("\nBest Hyperparameters Found:")
    for model_name, (params, mae) in results.items():
        print(f"\n{model_name}:")
        print(f"  MAE: {mae:.6f}")
        print(f"  Parameters:")
        for key, value in params.items():
            print(f"    - {key}: {value}")
    
    print("\n[OK] Optimization complete!")
    print("="*80 + "\n")
    
    return results


if __name__ == "__main__":
    results = main()
