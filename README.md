# Volatility Forecasting POC

A proof-of-concept project for forecasting financial market volatility using various machine learning models.

## Models Implemented

1. **SARIMA** - Seasonal AutoRegressive Integrated Moving Average
2. **Exponential Smoothing** - Holt-Winters method
3. **Prophet** - Facebook's time series forecasting library
4. **LSTM** - Long Short-Term Memory neural network
5. **Advanced LSTM** - LSTM with attention mechanism
6. **GARCH** - Generalized AutoRegressive Conditional Heteroskedasticity

## Project Structure

```
volatility-forecasting-poc/
├── src/
│   ├── data/           # Data loading and preprocessing
│   ├── models/         # Model implementations
│   ├── evaluation/     # Metrics and validation
│   └── config.py       # Configuration settings
├── scripts/            # Training and evaluation scripts
├── requirements.txt    # Python dependencies
└── README.md
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. For Prophet model:
```bash
pip install prophet
```

3. For GARCH model:
```bash
pip install arch
```

## Usage

### Training and Evaluation

Run the comprehensive comparison script:
```bash
python scripts/final_comprehensive_comparison.py
```

### Hyperparameter Tuning

Run Optuna hyperparameter optimization:
```bash
python scripts/optuna_hyperparameter_tuning.py
```

### Walk-Forward Validation

Run production evaluation with walk-forward validation:
```bash
python scripts/phase3_walk_forward_eval.py
```

### Visualization

Generate forecast comparison plots:
```bash
python scripts/visualize_results.py
```

## Key Features

- Multiple forecasting models for comparison
- Time-aware train/val/test splitting
- Walk-forward validation for realistic evaluation
- Hyperparameter optimization with Optuna
- Comprehensive metrics (MAE, RMSE, MAPE, directional accuracy)
- Confidence interval generation

## License

MIT License
