"""
Advanced LSTM with Attention Mechanism for volatility forecasting.

Features:
- Attention mechanism (learns which past days matter most)
- Exogenous variables (can look at VIX and Volume, not just past volatility)
- Beginner-friendly code structure
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader
import sys
from pathlib import Path

# Fix python path for local imports
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.models.base import DeepLearningModel
from src.config import Config


class AttentionLayer(nn.Module):
    """
    Attention mechanism for LSTM.
    It acts like a highlighter, telling the model which past days
    are the most important for predicting tomorrow.
    """
    
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        # A simple linear layer to calculate a "score" for each day
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_output):
        # 1. Calculate scores for each past day
        scores = self.attention(lstm_output)
        
        # 2. Turn scores into percentages (weights) that sum to 100%
        # using softmax.
        weights = torch.softmax(scores, dim=1)
        
        # 3. Multiply the outputs by their weights to shrink less important days
        # and boost important days.
        context = torch.sum(weights * lstm_output, dim=1)
        
        return context, weights


class AdvancedLSTMNetwork(nn.Module):
    """
    Advanced LSTM Neural Network.
    
    Steps:
    1. Look at data using an LSTM layer.
    2. Focus on the most important days (Attention).
    3. Make a final prediction using standard Dense layers.
    """
    
    def __init__(self, input_features=1, sequence_length=30, lstm_units=(64, 32), dropout=0.2):
        super(AdvancedLSTMNetwork, self).__init__()
        
        # First LSTM layer: reads the sequence of data
        self.lstm1 = nn.LSTM(
            input_size=input_features,  # e.g., 1 for just Volatility, 3 for Vol+VIX+Volume
            hidden_size=lstm_units[0],
            batch_first=True,
            num_layers=1
        )
        self.dropout1 = nn.Dropout(dropout)
        
        # Attention layer: decides what is important
        self.attention = AttentionLayer(lstm_units[0])
        
        # Second LSTM layer: processes the context further
        self.lstm2 = nn.LSTM(
            input_size=lstm_units[0],
            hidden_size=lstm_units[1],
            batch_first=True,
            num_layers=1
        )
        self.dropout2 = nn.Dropout(dropout)
        
        # Batch normalization: helps the network learn faster by stabilizing inputs
        self.batch_norm = nn.BatchNorm1d(lstm_units[1])
        
        # Final decision layers (Dense layers)
        self.fc1 = nn.Linear(lstm_units[1], 16)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """Pass data through the network."""
        # 1. Pass through first LSTM
        lstm_out1, _ = self.lstm1(x)
        lstm_out1 = self.dropout1(lstm_out1)
        
        # 2. Apply Attention
        attention_context, _ = self.attention(lstm_out1)
        
        # 3. Pass through second LSTM
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out2 = self.dropout2(lstm_out2)
        
        # Grab the very last output of the sequence
        lstm_last = lstm_out2[:, -1, :]
        lstm_last = self.batch_norm(lstm_last)
        
        # 4. Final dense layers to produce 1 number (the forecast)
        hidden = self.relu(self.fc1(lstm_last))
        output = self.fc2(hidden)
        
        return output


class AdvancedLSTMModel(DeepLearningModel):
    """
    Wrapper around the PyTorch network to make it easy to use
    (like a Scikit-Learn model).
    """
    
    def __init__(self,
                 sequence_length=30,
                 lstm_units=(64, 32),
                 dropout=0.2,
                 epochs=50,
                 batch_size=16,
                 learning_rate=0.001,
                 device='cpu'):
        
        super().__init__(name='AdvancedLSTM')
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        
        # Use GPU if available
        if device == 'cuda' and not torch.cuda.is_available():
            print("WARNING: GPU/CUDA not available, using CPU instead")
            self.device = 'cpu'
        
        self.neural_model = None
        self.train_losses = []
        self.val_losses = []
        self.y_train = None
        self.train_preds = None
        self.num_features = 1 # Default to 1 (just target variable)
    
    def _create_sequences(self, data: np.ndarray, exog: np.ndarray = None) -> tuple:
        """
        Takes a long list of days and cuts it into small "windows" (sequences).
        If we look at 30 days of data, the 31st day is our target.
        """
        X, y = [], []
        
        # If we have extra data (like VIX or Volume), glue it side-by-side with our main data
        if exog is not None:
            # Reshape data to be a column if it's flat
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            # Stack horizontally: [Volatility, Vix, Volume]
            combined_data = np.hstack([data, exog])
            self.num_features = combined_data.shape[1]
        else:
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            combined_data = data
            self.num_features = 1

        # Create the sliding windows
        for i in range(len(combined_data) - self.sequence_length):
            # The window of features
            window = combined_data[i : i + self.sequence_length]
            X.append(window)
            
            # The target to predict (always the first column, which is Volatility)
            target = combined_data[i + self.sequence_length, 0]
            y.append(target)
        
        # Convert to PyTorch tensors so the neural network can process them
        X_tensor = torch.FloatTensor(np.array(X))
        y_tensor = torch.FloatTensor(np.array(y)).reshape(-1, 1)
        
        return X_tensor, y_tensor
    
    def fit(self, train_data: np.ndarray, val_data: np.ndarray = None, 
            exog: np.ndarray = None, val_exog: np.ndarray = None) -> None:
        """
        Train the model on historical data.
        """
        print("\n" + "="*80)
        print("ADVANCED LSTM MODEL - TRAINING")
        print("="*80)
        
        super().fit(train_data, exog=exog)
        self.last_train_exog = exog
        
        # 1. Prepare the data into chunks/sequences
        print(f"\nPreparing data...")
        X_train, y_train = self._create_sequences(train_data, exog)
        self.y_train = y_train.numpy().flatten()
        
        # Prepare validation data if provided
        if val_data is not None:
            X_val, y_val = self._create_sequences(val_data, val_exog)
        else:
            X_val, y_val = None, None
        
        # Create a "DataLoader" to feed data to the network in small batches
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = TorchDataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # 2. Build the Network
        print(f"\nBuilding Advanced LSTM network ({self.num_features} input features)...")
        self.neural_model = AdvancedLSTMNetwork(
            input_features=self.num_features,
            sequence_length=self.sequence_length,
            lstm_units=self.lstm_units,
            dropout=self.dropout
        ).to(self.device)
        
        # Setup how we measure errors (Mean Squared Error) and how we learn (Adam Optimizer)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.neural_model.parameters(), lr=self.learning_rate)
        
        # Automatically lower the learning rate if the model stops improving
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 3. Training Loop
        print(f"\nTraining network...")
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.neural_model.train() # Set to train mode
            train_loss = 0.0
            
            # Learn from each batch
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()            # Clear old calculations
                output = self.neural_model(X_batch) # Guess the answer
                loss = criterion(output, y_batch) # Calculate how wrong it was
                loss.backward()                  # Figure out how to fix it
                optimizer.step()                 # Update the brain
                
                train_loss += loss.item()
            
            # Calculate average training loss for this epoch
            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)
            
            # Check validation score
            if val_data is not None:
                self.neural_model.eval() # Set to evaluation mode
                with torch.no_grad():    # Don't learn, just test
                    val_output = self.neural_model(X_val.to(self.device))
                    val_loss = criterion(val_output, y_val.to(self.device)).item()
                    self.val_losses.append(val_loss)
                    
                    # Tell scheduler to maybe lower learning rate
                    scheduler.step(val_loss)
                    
                    # Early Stopping: Stop if we haven't improved in 10 epochs
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= 10:
                        print(f"  Early stopping at epoch {epoch+1} (No improvement)")
                        break
            
            # Print an update every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                if val_data is not None:
                    print(f"  Epoch {epoch+1}/{self.epochs} - Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                else:
                    print(f"  Epoch {epoch+1}/{self.epochs} - Loss: {train_loss:.6f}")
        
        # Save final training predictions for confidence intervals later
        self.neural_model.eval()
        with torch.no_grad():
            self.train_preds = self.neural_model(X_train.to(self.device)).cpu().numpy().flatten()
        
        print(f"[OK] Advanced LSTM trained successfully")
    
    def forecast(self, recent_data: np.ndarray, horizon: int = 1, 
                 confidence_level: float = 0.95, exog: np.ndarray = None) -> tuple:
        """
        Look into the future.
        """
        super().forecast(horizon, confidence_level, exog=exog)
        
        if self.neural_model is None:
            raise ValueError("Model must be trained first")
            
        # Ensure we have enough recent data
        if len(recent_data) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} recent days to forecast")
            
        # Get the most recent X days
        recent_target = recent_data[-self.sequence_length:]
        if recent_target.ndim == 1:
            recent_target = recent_target.reshape(-1, 1)
            
        predictions = []
        
        self.neural_model.eval()
        with torch.no_grad():
            
            # Start our "current" sequence memory
            current_sequence = recent_target
            
            # If we used extra features during training, we MUST have them for the future
            if self.num_features > 1:
                # We need extra features for the historical window AND the future horizon
                if exog is None:
                    raise ValueError("Model was trained with exog, so exog must be provided for forecasting")
                    
                # Prepare the historical portion of the exogenous data
                if self.last_train_exog is not None and len(self.last_train_exog) >= self.sequence_length:
                    recent_exog = self.last_train_exog[-self.sequence_length:]
                elif exog is not None and len(exog) >= self.sequence_length:
                    # Fallback if fit wasn't called or exog is provided with history
                    recent_exog = exog[-self.sequence_length:]
                else:
                    raise ValueError(f"Model needs at least {self.sequence_length} days of exogenous history")
                
                current_sequence = np.hstack([current_sequence, recent_exog])

            current_sequence = torch.FloatTensor(current_sequence).unsqueeze(0) # Add batch dimension

            for step in range(horizon):
                # Predict next day
                seq_device = current_sequence.to(self.device)
                next_pred = self.neural_model(seq_device)
                pred_value = next_pred.cpu().item()
                pred_value = max(pred_value, 0.001) # Volatility can't be negative
                predictions.append(pred_value)
                
                # Setup the sequence for the step after that:
                # Remove the oldest day
                new_seq = current_sequence.numpy()[0, 1:, :]
                
                # Add the newly predicted day
                new_day_features = [pred_value]
                
                # If we have extra features, guess that tomorrow's VIX/Volume will be the same as today's
                if self.num_features > 1:
                    last_known_extra_features = new_seq[-1, 1:]
                    new_day_features.extend(last_known_extra_features)
                    
                new_day_array = np.array([new_day_features])
                
                # Glue them together to make a new window
                updated_sequence = np.append(new_seq, new_day_array, axis=0)
                current_sequence = torch.FloatTensor(updated_sequence).unsqueeze(0)
        
        predictions = np.array(predictions)
        
        # Basic confidence interval math (using standard deviation of training errors)
        if self.train_preds is not None and len(self.train_preds) > 0:
            residuals = np.abs(self.y_train - self.train_preds)
            residual_std = np.std(residuals)
            if residual_std == 0:
                residual_std = 0.01
            
            lower_bound = np.maximum(predictions - 1.96 * residual_std, 0.001)
            upper_bound = predictions + 1.96 * residual_std
        else:
            lower_bound = predictions * 0.8
            upper_bound = predictions * 1.2
        
        return predictions, (lower_bound, upper_bound)
    
    def get_params(self):
        return {
            'name': self.name,
            'sequence_length': self.sequence_length,
            'lstm_units': self.lstm_units,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'is_fitted': self.is_fitted,
        }


# Quick test
if __name__ == "__main__":
    from src.data.loader import DataLoader
    from src.data.preprocessor import DataSplitter
    
    print("Testing Simplified Advanced LSTM Model...")
    
    # 1. Provide fake data (2 features: Volatility, and VIX)
    vol = np.random.normal(0.02, 0.01, 500)
    vix = np.random.normal(20, 5, 500)
    vix = vix.reshape(-1, 1) # Make column shape
    
    train_vol, val_vol, test_vol = DataSplitter.train_val_test_split(vol)
    train_vix, val_vix, test_vix = DataSplitter.train_val_test_split(vix)
    
    # 2. Train with extra feature
    lstm = AdvancedLSTMModel(epochs=5, sequence_length=10)
    lstm.fit(train_data=train_vol, val_data=val_vol, exog=train_vix, val_exog=val_vix)
    
    # 3. Predict
    preds, (low, high) = lstm.forecast(recent_data=test_vol, horizon=5, exog=test_vix)
    
    print("\nForecast successfully calculated!")
    print(preds)
