import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
  sys.path.append(str(root_dir))
from src.models.base import DeepLearningModel
from src.config import Config
class AttentionLayer(nn.Module):
  def __init__(self, hidden_size):
    super(AttentionLayer, self).__init__()
    self.attention = nn.Linear(hidden_size, 1)
  def forward(self, lstm_output):
    scores = self.attention(lstm_output)
    weights = torch.softmax(scores, dim=1)
    context = torch.sum(weights * lstm_output, dim=1)
    return context, weights
class AdvancedLSTMNetwork(nn.Module):
  def __init__(self, input_features=1, sequence_length=30, lstm_units=(64, 32), dropout=0.2):
    super(AdvancedLSTMNetwork, self).__init__()
    self.lstm1 = nn.LSTM(
      input_size=input_features, 
      hidden_size=lstm_units[0],
      batch_first=True,
      num_layers=1
    )
    self.dropout1 = nn.Dropout(dropout)
    self.attention = AttentionLayer(lstm_units[0])
    self.lstm2 = nn.LSTM(
      input_size=lstm_units[0],
      hidden_size=lstm_units[1],
      batch_first=True,
      num_layers=1
    )
    self.dropout2 = nn.Dropout(dropout)
    self.batch_norm = nn.BatchNorm1d(lstm_units[1])
    self.fc1 = nn.Linear(lstm_units[1], 16)
    self.fc2 = nn.Linear(16, 1)
    self.relu = nn.ReLU()
  def forward(self, x):
    lstm_out1, _ = self.lstm1(x)
    lstm_out1 = self.dropout1(lstm_out1)
    attention_context, _ = self.attention(lstm_out1)
    lstm_out2, _ = self.lstm2(lstm_out1)
    lstm_out2 = self.dropout2(lstm_out2)
    lstm_last = lstm_out2[:, -1, :]
    lstm_last = self.batch_norm(lstm_last)
    hidden = self.relu(self.fc1(lstm_last))
    output = self.fc2(hidden)
    return output
class AdvancedLSTMModel(DeepLearningModel):
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
    if device == 'cuda' and not torch.cuda.is_available():
      print("WARNING: GPU/CUDA not available, using CPU instead")
      self.device = 'cpu'
    self.neural_model = None
    self.train_losses = []
    self.val_losses = []
    self.y_train = None
    self.train_preds = None
    self.num_features = 1
  def _create_sequences(self, data, exog=None):
    X = []
    y = []
    if exog is not None:
      if data.ndim == 1:
        data = data.reshape(-1, 1)
      combined_data = np.hstack([data, exog])
      self.num_features = combined_data.shape[1]
    else:
      if data.ndim == 1:
        data = data.reshape(-1, 1)
      combined_data = data
      self.num_features = 1
    for i in range(len(combined_data) - self.sequence_length):
      window = combined_data[i : i + self.sequence_length]
      X.append(window)
      target = combined_data[i + self.sequence_length, 0]
      y.append(target)
    X_array = np.array(X)
    y_array = np.array(y)
    X_tensor = torch.FloatTensor(X_array)
    y_tensor = torch.FloatTensor(y_array).reshape(-1, 1)
    return X_tensor, y_tensor
  def fit(self, train_data, val_data=None, exog=None, val_exog=None):
    print("" + "="*80)
    print("ADVANCED LSTM MODEL - TRAINING")
    print("="*80)
    super().fit(train_data, exog=exog)
    self.last_train_exog = exog
    print("Preparing data")
    X_train, y_train = self._create_sequences(train_data, exog)
    self.y_train = y_train.numpy().flatten()
    if val_data is not None:
      X_val, y_val = self._create_sequences(val_data, val_exog)
    else:
      X_val, y_val = None, None
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = TorchDataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
    print(f"Building Advanced LSTM network ({self.num_features} input features)")
    self.neural_model = AdvancedLSTMNetwork(
      input_features=self.num_features,
      sequence_length=self.sequence_length,
      lstm_units=self.lstm_units,
      dropout=self.dropout
    ).to(self.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(self.neural_model.parameters(), lr=self.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
      optimizer, mode='min', factor=0.5, patience=5
    )
    print("Training network")
    best_val_loss = float('inf')
    patience_counter = 0
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
          val_output = self.neural_model(X_val.to(self.device))
          val_loss = criterion(val_output, y_val.to(self.device)).item()
          self.val_losses.append(val_loss)
          scheduler.step(val_loss)
          if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
          else:
            patience_counter += 1
          if patience_counter >= 10:
            print(f"Early stopping at epoch {epoch+1} (No improvement)")
            break
      if (epoch + 1) % 10 == 0 or epoch == 0:
        if val_data is not None:
          print(f"Epoch {epoch+1}/{self.epochs} - Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        else:
          print(f"Epoch {epoch+1}/{self.epochs} - Loss: {train_loss:.6f}")
    self.neural_model.eval()
    with torch.no_grad():
      self.train_preds = self.neural_model(X_train.to(self.device)).cpu().numpy().flatten()
    print("Advanced LSTM trained successfully")
  def forecast(self, recent_data, horizon=1, confidence_level=0.95, exog=None):
    super().forecast(horizon, confidence_level, exog=exog)
    if self.neural_model is None:
      raise ValueError("Model must be trained first")
    if len(recent_data) < self.sequence_length:
      raise ValueError(f"Need at least {self.sequence_length} recent days to forecast")
    recent_target = recent_data[-self.sequence_length:]
    if recent_target.ndim == 1:
      recent_target = recent_target.reshape(-1, 1)
    predictions = []
    self.neural_model.eval()
    with torch.no_grad():
      current_sequence = recent_target
      if self.num_features > 1:
        if exog is None:
          raise ValueError("Model was trained with exog, so exog must be provided for forecasting")
        if self.last_train_exog is not None and len(self.last_train_exog) >= self.sequence_length:
          recent_exog = self.last_train_exog[-self.sequence_length:]
        elif exog is not None and len(exog) >= self.sequence_length:
          recent_exog = exog[-self.sequence_length:]
        else:
          raise ValueError(f"Model needs at least {self.sequence_length} days of exogenous history")
        current_sequence = np.hstack([current_sequence, recent_exog])
      current_sequence = torch.FloatTensor(current_sequence).unsqueeze(0)
      for step in range(horizon):
        seq_device = current_sequence.to(self.device)
        next_pred = self.neural_model(seq_device)
        pred_value = next_pred.cpu().item()
        pred_value = max(pred_value, 0.001)
        predictions.append(pred_value)
        new_seq = current_sequence.numpy()[0, 1:, :]
        new_day_features = [pred_value]
        if self.num_features > 1:
          last_known_extra_features = new_seq[-1, 1:]
          new_day_features.extend(last_known_extra_features)
        new_day_array = np.array([new_day_features])
        updated_sequence = np.append(new_seq, new_day_array, axis=0)
        current_sequence = torch.FloatTensor(updated_sequence).unsqueeze(0)
    predictions = np.array(predictions)
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
if __name__ == "__main__":
  from src.data.loader import DataLoader
  from src.data.preprocessor import DataSplitter
  print("Testing Simplified Advanced LSTM Model")
  vol = np.random.normal(0.02, 0.01, 500)
  vix = np.random.normal(20, 5, 500)
  vix = vix.reshape(-1, 1)
  train_vol, val_vol, test_vol = DataSplitter.train_val_test_split(vol)
  train_vix, val_vix, test_vix = DataSplitter.train_val_test_split(vix)
  lstm = AdvancedLSTMModel(epochs=5, sequence_length=10)
  lstm.fit(train_data=train_vol, val_data=val_vol, exog=train_vix, val_exog=val_vix)
  preds, (low, high) = lstm.forecast(recent_data=test_vol, horizon=5, exog=test_vix)
  print("Forecast successfully calculated!")
  print(preds)