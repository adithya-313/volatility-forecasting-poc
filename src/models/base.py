from abc import ABC, abstractmethod
import numpy as np
class VolatilityModel(ABC):
  def __init__(self, name):
    self.name = name
    self.is_fitted = False
    self.residuals = None
  @abstractmethod
  def fit(self, train_data, exog=None):
    if len(train_data) < 10:
      raise ValueError("Need at least 10 samples to train")
    self.is_fitted = True
    print(f"{self.name} trained on {len(train_data)} samples")
  @abstractmethod
  def forecast(self, horizon=1, confidence_level=0.95, exog=None):
    if not self.is_fitted:
      raise RuntimeError(f"{self.name} must be fitted before forecasting")
  @abstractmethod
  def get_params(self):
    pass
  def summary(self):
    return f"{self.name} - Fitted: {self.is_fitted}"
class ClassicalModel(VolatilityModel):
  def __init__(self, name):
    super().__init__(name)
    self.model = None 
class DeepLearningModel(VolatilityModel):
  def __init__(self, name):
    super().__init__(name)
    self.neural_model = None 
    self.scaler = None 
    self.training_history = None 