import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import DataLoader, DataValidator

def test():
    loader = DataLoader()
    data = loader.generate_synthetic_volatility(n_samples=100)
    
    validator = DataValidator()
    is_valid, issues = validator.validate(data)
    
    if is_valid:
        print("Test PASSED: Data generated and validated.")
    else:
        print(f"Test FAILED: {issues}")

if __name__ == "__main__":
    test()
