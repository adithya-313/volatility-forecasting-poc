from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import numpy as np

class VolatilityModel(ABC):
 
    def __init__(self, name: str):
      
        self.name = name
        self.is_fitted = False
        self.residuals = None
    
    @abstractmethod
    def fit(self, train_data: np.ndarray, exog: np.ndarray = None) -> None:
    
        if len(train_data) < 10:
            raise ValueError("Need at least 10 samples to train")
        
        self.is_fitted = True
        print(f"[OK] {self.name} trained on {len(train_data)} samples")
    
    @abstractmethod
    def forecast(self, horizon: int = 1, confidence_level: float = 0.95, exog: np.ndarray = None) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before forecasting")
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
 
        pass
    
    def summary(self) -> str:
        return f"{self.name} - Fitted: {self.is_fitted}"


class ClassicalModel(VolatilityModel):

    def __init__(self, name: str):
        super().__init__(name)
        self.model = None  

class DeepLearningModel(VolatilityModel):

    
    def __init__(self, name: str):
        super().__init__(name)
        self.neural_model = None 
        self.scaler = None  
        self.training_history = None  