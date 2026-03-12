import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader
import sys
import os
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
  sys.path.append(str(root_dir))
from src.models.base import DeepLearningModel
from src.config import Config
class LSTMNetwork(nn.Module):
  def __init__(self, sequence_length=30, lstm_units=(64, 32), dropout=0.2):
    super(LSTMNetwork, self).__init__()
    self.lstm1 = nn.LSTM(input_size=1, hidden_size=lstm_units[0], 
               batch_first=True)
    self.dropout1 = nn.Dropout(dropout)
    self.lstm2 = nn.LSTM(input_size=lstm_units[0], hidden_size=lstm_units[1], 
               batch_first=True)
    self.dropout2 = nn.Dropout(dropout)
    self.fc1 = nn.Linear(lstm_units[1], 16)
    self.fc2 = nn.Linear(16, 1)
  def forward(self, x):
    lstm_out1, _ = self.lstm1(x)
    lstm_out1 = self.dropout1(lstm_out1)
    lstm_out2, _ = self.lstm2(lstm_out1)
    lstm_out2 = self.dropout2(lstm_out2)
    lstm_last = lstm_out2[:, -1, :]
    hidden = torch.relu(self.fc1(lstm_last))
    output = self.fc2(hidden)
    return output
class LSTMModel(DeepLearningModel):
  def __init__(self, 
         sequence_length=30,
         lstm_units=(64, 32),
         dropout=0.2,
         epochs=50,
         batch_size=32,
         learning_rate=0.01,
         device='cpu'):
    super().__init__(name='LSTM')
    self.sequence_length = sequence_length
    self.lstm_units = lstm_units
    self.dropout = dropout
    self.epochs = epochs
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.device = device
    if device == 'cuda' and not torch.cuda.is_available():
      print("WARNING: CUDA not available, using CPU")
      self.device = 'cpu'
    self.neural_model = None
    self.train_losses = []
    self.val_losses = []
    self.y_train = None
    self.train_preds = None
  def _create_sequences(self, data):
    X, y = [], []
    for i in range(len(data) - self.sequence_length):
      X.append(data[i:i+self.sequence_length])
      y.append(data[i+self.sequence_length])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).reshape(-1, 1)
    return X_tensor, y_tensor
  def fit(self, train_data, val_data=None, exog=None):
    print("" + "="*60)
    print("LSTM MODEL - TRAINING")
    print("="*60)
    super().fit(train_data)
    print("Configuration:")
    print(f"- Sequence length: {self.sequence_length}")
    print(f"- LSTM units: {self.lstm_units}")
    print(f"- Dropout: {self.dropout}")
    print(f"- Learning rate: {self.learning_rate}")
    print(f"- Epochs: {self.epochs}")
    print(f"- Device: {self.device}")
    print("Preparing data")
    X_train, y_train = self._create_sequences(train_data)
    self.y_train = y_train.numpy().flatten()
    print(f"- X_train shape: {X_train.shape}")
    print(f"- y_train shape: {self.y_train.shape}")
    if val_data is not None:
      X_val, y_val = self._create_sequences(val_data)
      print(f"- X_val shape: {X_val.shape}")
    else:
      X_val, y_val = None, None
    y_train_tensor = torch.FloatTensor(self.y_train).reshape(-1, 1)
    train_dataset = TensorDataset(X_train, y_train_tensor)
    train_loader = TorchDataLoader(train_dataset, batch_size=self.batch_size, 
                   shuffle=True)
    print("Building neural network")
    self.neural_model = LSTMNetwork(
      sequence_length=self.sequence_length,
      lstm_units=self.lstm_units,
      dropout=self.dropout
    ).to(self.device)
    print("Network built")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(self.neural_model.parameters(), 
               lr=self.learning_rate)
    print("Training network")
    best_val_loss = float('inf')
    patience = 0
    for epoch in range(self.epochs):
      self.neural_model.train()
      train_loss = 0.0
      for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        optimizer.zero_grad()
        output = self.neural_model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
      train_loss /= len(train_loader)
      self.train_losses.append(train_loss)
      if val_data is not None:
        self.neural_model.eval()
        with torch.no_grad():
          X_val_device = X_val.to(self.device)
          y_val_device = y_val.to(self.device)
          val_output = self.neural_model(X_val_device)
          val_loss = criterion(val_output, y_val_device).item()
          self.val_losses.append(val_loss)
          if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
          else:
            patience += 1
          if patience >= 5:
            print(f"Early stopping at epoch {epoch+1}")
            break
      if (epoch + 1) % 10 == 0 or epoch == 0:
        if val_data is not None and len(self.val_losses) > 0:
          print(f" Epoch {epoch+1}/{self.epochs} - "
             f"Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        else:
          print(f"Epoch {epoch+1}/{self.epochs} - Loss: {train_loss:.6f}")
    print("Calculating training predictions")
    self.neural_model.eval()
    with torch.no_grad():
      X_train_device = X_train.to(self.device)
      raw_preds = self.neural_model(X_train_device)
      clamped_preds = torch.clamp(raw_preds, min=0.0001)
      self.train_preds = clamped_preds.cpu().numpy().flatten()
    print("LSTM trained successfully")
    print(f"- Final training loss: {self.train_losses:.6f}")
    if len(self.val_losses) > 0:
      print(f"- Final validation loss: {self.val_losses:.6f}")
    print("="*60)
  def forecast(self, recent_data=None, horizon=1, confidence_level=0.95, exog=None):
    super().forecast(horizon, confidence_level)
    if self.neural_model is None:
      raise ValueError("Model must be trained first")
    if recent_data is None:
      raise ValueError("recent_data required for forecasting")
    current_sequence = recent_data[-self.sequence_length:].reshape(1, -1, 1)
    current_sequence = torch.FloatTensor(current_sequence)
    predictions = []
    self.neural_model.eval()
    with torch.no_grad():
      for _ in range(horizon):
        seq_device = current_sequence.to(self.device)
        next_pred = self.neural_model(seq_device)
        pred_value = next_pred.cpu().item()
        pred_value = max(pred_value, 0.001)
        predictions.append(pred_value)
        new_seq = np.append(current_sequence.numpy()[0, 1:, :], 
                  [[pred_value]], axis=0)
        current_sequence = torch.FloatTensor(new_seq).reshape(1, -1, 1)
    predictions = np.array(predictions)
    mean_train = np.mean(self.y_train)
    std_train = np.std(self.y_train)
    lower_bound = np.maximum(predictions - 1.96 * std_train, 0.001)
    upper_bound = predictions + 1.96 * std_train
    return predictions, (lower_bound, upper_bound)
  def get_params(self):
    return {
      'name': self.name,
      'sequence_length': self.sequence_length,
      'lstm_units': self.lstm_units,
      'dropout': self.dropout,
      'epochs': self.epochs,
      'batch_size': self.batch_size,
      'learning_rate': self.learning_rate,
      'device': self.device,
      'is_fitted': self.is_fitted,
    }
if __name__ == "__main__":
  try:
    from src.data.loader import DataLoader
    from src.data.preprocessor import DataPreprocessor, DataSplitter
    print("" + "="*60)
    print("LSTM MODEL - COMPLETE TEST")
    print("="*60)
    print("Generating data")
    data = DataLoader.generate_synthetic_volatility(n_samples=1500)
    print("Data generated")
    print("Preprocessing")
    prep = DataPreprocessor()
    data = prep.handle_missing_values(data, 'volatility')
    data = prep.handle_outliers(data, 'volatility')
    print("Data preprocessed")
    print("Splitting data")
    vol = data['volatility'].values
    train, val, test = DataSplitter.train_val_test_split(vol)
    print(f"Data split: train={len(train)}, val={len(val)}, test={len(test)}")
    print("Training LSTM")
    lstm = LSTMModel(
      sequence_length=Config.LSTM_CONFIG['sequence_length'],
      lstm_units=Config.LSTM_CONFIG['lstm_units'],
      dropout=Config.LSTM_CONFIG['dropout'],
      epochs=30,
      batch_size=Config.LSTM_CONFIG['batch_size'],
      learning_rate=0.01,
      device='cpu'
    )
    lstm.fit(train, val_data=val)
    print("Forecasting")
    predictions, (lower, upper) = lstm.forecast(recent_data=test, horizon=10)
    print("Forecasting complete")
    print("" + "="*60)
    print("RESULTS")
    print("="*60)
    print("Forecast (next 10 days):")
    print(f"Predictions: {predictions}")
    print(f"Lower bounds: {lower}")
    print(f"Upper bounds: {upper}")
    print("Model Parameters:")
    params = lstm.get_params()
    for key, val in params.items():
      print(f"- {key}: {val}")
    print("ALL TESTS COMPLETE!")
    print("="*60)
  except Exception as e:
    print("ERROR OCCURRED:")
    print(f"{str(e)}")
    import traceback
    print("Full traceback:")
    traceback.print_exc()