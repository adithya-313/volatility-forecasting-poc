"""
Walk-Forward Validation Engine.

In the real world, you don't just train a model once. You train it on Monday,
use it all week, and then re-train it on Friday with the new data.

This script simulates that real-world "rolling" or "expanding" window approach
to give us a much more accurate score of how our models will actually perform
in production over time.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import sys
from pathlib import Path

# Fix python path for local imports
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.models.base import VolatilityModel
from src.evaluation.metrics import VolatilityMetrics


class WalkForwardValidator:
    """
    Evaluates a model by repeatedly training and testing it as time moves forward.
    """
    
    def __init__(self, 
                 initial_train_size: int = 500, 
                 step_size: int = 5,
                 window_type: str = 'expanding'):
        """
        Setup the validation rules.
        
        Args:
            initial_train_size: How many days to use for the very first training run.
            step_size: How many days to forecast before we stop and re-train.
                       (e.g., 5 means we predict a whole week, then re-train on the weekend)
            window_type: 
                - 'expanding': We keep all historical data in our memory permanently.
                - 'rolling': We only ever look at the last `initial_train_size` days, 
                             forgetting the very old data.
        """
        self.initial_train_size = initial_train_size
        self.step_size = step_size
        
        if window_type not in ['expanding', 'rolling']:
            raise ValueError("window_type must be either 'expanding' or 'rolling'")
        self.window_type = window_type
        
        self.metrics = VolatilityMetrics()
    
    def evaluate(self, 
                 model: VolatilityModel, 
                 data: np.ndarray, 
                 exog: np.ndarray = None) -> Dict[str, Any]:
        """
        Run the walk-forward simulation on a model.
        
        Args:
            model: The forecasting model to test.
            data: The entire timeline of historical data (e.g., 1000 days).
            exog: Any extra variables (like VIX) that the model might use.
            
        Returns:
            A dictionary containing the final average error and all predictions.
        """
        
        total_days = len(data)
        
        # Make sure we have enough data to even run the test
        if total_days <= self.initial_train_size + self.step_size:
            raise ValueError("Not enough data to run walk-forward validation.")
            
        print(f"\nStart Walk-Forward Validation for: {model.name}")
        print(f"  - Total Data Size: {total_days} days")
        print(f"  - Initial Training: First {self.initial_train_size} days")
        print(f"  - Step Size (Forecast Horizon): {self.step_size} days")
        print(f"  - Window Style: {self.window_type.capitalize()}")
        
        # Store our results here
        all_actuals = []
        all_predictions = []
        all_lower_bounds = []
        all_upper_bounds = []
        
        # We start looking at the future starting right after the initial train size
        current_step = self.initial_train_size
        simulation_step_counter = 1
        
        # Keep sliding the window forward until we hit the end of our dataset
        while current_step < total_days:
            # 1. Figure out what data we are allowed to 'see' (Train Window)
            
            if self.window_type == 'expanding':
                # Expanding: Start at 0, end at current_step.
                train_data = data[:current_step]
                train_exog = exog[:current_step] if exog is not None else None
            else:
                # Rolling: Start "initial_train_size" steps backwards, end at current_step.
                train_start_index = current_step - self.initial_train_size
                train_data = data[train_start_index:current_step]
                train_exog = exog[train_start_index:current_step] if exog is not None else None
                
            # 2. Figure out what we are trying to predict (Test Window)
            
            # The end of the test window is `current_step` + `step_size`, 
            # OR the end of the array (whichever comes first).
            test_end = min(current_step + self.step_size, total_days)
            horizon = test_end - current_step  # Might be less than step_size at the very end
            
            test_actuals = data[current_step:test_end]
            test_exog = exog[current_step:test_end] if exog is not None else None
            
            # 3. Train the model on the historical window
            
            # Deep Learning models (like LSTM) need a fresh copy of themselves, 
            # otherwise they just keep learning on top of already-memorized data 
            # which breaks the test. For classical models this is less of an issue, 
            # but it doesn't hurt.
            
            print(f"  => Window {simulation_step_counter}: Training on {len(train_data)} days, Predicting next {horizon} days...")
            
            # Reset the fitted flag
            model.is_fitted = False
            
            try:
                # Some models might need validation data (like LSTMs). We don't have a distinct
                # validation chunk in walk-forward, so we just pass train_data alone for now.
                model.fit(train_data, exog=train_exog)
                
                # Predict the future window
                # Note: Deep Learning models often require 'recent_data' passed into forecast
                options = {'horizon': horizon, 'exog': test_exog}
                if hasattr(model, 'neural_model') and model.neural_model is not None:
                     options['recent_data'] = train_data
                
                preds, (lower, upper) = model.forecast(**options)
                
                # Make sure lengths perfectly align (sometimes models return padding)
                preds = preds[:horizon]
                lower = lower[:horizon]
                upper = upper[:horizon]
                
            except Exception as e:
                print(f"  [ERROR] Model failed on window {simulation_step_counter}: {e}")
                print("  Filling predictions with the naive constant (the last known day's value)...")
                
                # If the model crashes, we guess that volatility stays exactly the same as the last known day.
                # This is standard practice in finance ("Random Walk" assumption).
                last_known_value = train_data[-1]
                preds = np.full(horizon, last_known_value)
                lower = np.full(horizon, last_known_value * 0.8)
                upper = np.full(horizon, last_known_value * 1.2)
                
            # 4. Save results
            all_actuals.extend(test_actuals)
            all_predictions.extend(preds)
            all_lower_bounds.extend(lower)
            all_upper_bounds.extend(upper)
            
            # 5. Move time forward!
            current_step += self.step_size
            simulation_step_counter += 1
            
        # --- Final Calculation ---
        
        all_actuals = np.array(all_actuals)
        all_predictions = np.array(all_predictions)
        all_lower_bounds = np.array(all_lower_bounds)
        all_upper_bounds = np.array(all_upper_bounds)
        
        # Grade the model's entire history of predictions
        mae = self.metrics.mae(all_actuals, all_predictions)
        rmse = self.metrics.rmse(all_actuals, all_predictions)
        mape = self.metrics.mape(all_actuals, all_predictions)
        
        print(f"\n[OK] Configuration complete for {model.name}!")
        print(f"  - Final Mean Absolute Error (MAE): {mae:.6f}")
        print(f"  - We made {len(all_predictions)} predictions iteratively.")
        
        return {
            'model_name': model.name,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'actuals': all_actuals,
            'predictions': all_predictions,
            'lower_bounds': all_lower_bounds,
            'upper_bounds': all_upper_bounds
        }

# Quick Test
if __name__ == "__main__":
    from src.data.loader import DataLoader
    from src.models.garch_model import GARCHModel
    
    print("Testing Walk-Forward Validator...")
    print("=" * 60)
    
    # 1. Generate 1000 days of fake data
    data = DataLoader.generate_synthetic_volatility(n_samples=1000)
    volatility = data['volatility'].values
    
    # 2. Setup GARCH
    garch = GARCHModel(1, 1)
    
    # 3. Setup Validator
    # Train on first 800 days, then step forward 10 days at a time
    validator = WalkForwardValidator(initial_train_size=800, step_size=10, window_type='expanding')
    
    # 4. Run test
    results = validator.evaluate(model=garch, data=volatility)
    
    print("\nWalk-forward summary saved successfully.")
