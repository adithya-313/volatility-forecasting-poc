import numpy as np
from typing import Tuple

class VolatilityMetrics:
    
    @staticmethod
    def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
        
        return np.mean(np.abs(actual - predicted))
    
    @staticmethod
    def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
       
        return np.sqrt(np.mean((actual - predicted) ** 2))
    
    @staticmethod
    def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
        
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        mask = actual != 0
        if not mask.any():
            return np.inf
        
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    
    @staticmethod
    def directional_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
        
        actual_change = np.diff(actual)
        predicted_change = np.diff(predicted)
        
        correct = np.sign(actual_change) == np.sign(predicted_change)
        
        return np.mean(correct) * 100
    
    @staticmethod
    def confidence_interval_coverage(actual: np.ndarray, 
                                     lower: np.ndarray, 
                                     upper: np.ndarray) -> float:
        
        coverage = np.mean((actual >= lower) & (actual <= upper)) * 100
        return coverage
    
    @staticmethod
    def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
        
        return np.mean(upper - lower)
    
    @staticmethod
    def residuals(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
       
        return actual - predicted


class MetricsReport:
    
    @staticmethod
    def create_report(model_name: str,
                     actual: np.ndarray,
                     predicted: np.ndarray,
                     lower: np.ndarray = None,
                     upper: np.ndarray = None) -> dict:
        
        metrics = VolatilityMetrics()
        
        report = {
            'model': model_name,
            'mae': metrics.mae(actual, predicted),
            'rmse': metrics.rmse(actual, predicted),
            'mape': metrics.mape(actual, predicted),
            'directional_accuracy': metrics.directional_accuracy(actual, predicted),
        }
        
        if lower is not None and upper is not None:
            report['coverage'] = metrics.confidence_interval_coverage(actual, lower, upper)
            report['interval_width'] = metrics.interval_width(lower, upper)
        
        return report
    
    @staticmethod
    def compare_models(results: dict) -> dict:
      
        metrics_df = {}
        
        for metric in ['mae', 'rmse', 'mape', 'directional_accuracy']:
            metric_values = {}
            for model_name, report in results.items():
                if metric in report:
                    metric_values[model_name] = report[metric]
            
            if metric in ['mae', 'rmse', 'mape']:
                sorted_models = sorted(metric_values.items(), key=lambda x: x[1])
            else:
                sorted_models = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
            
            metrics_df[metric] = {
                'ranking': [m[0] for m in sorted_models],
                'values': {m[0]: m[1] for m in sorted_models}
            }
        
        return metrics_df
    
    @staticmethod
    def print_report(model_name: str, report: dict):
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        
        print(f"\nAccuracy Metrics:")
        print(f"  - MAE:  {report.get('mae', 'N/A'):.6f}")
        print(f"  - RMSE: {report.get('rmse', 'N/A'):.6f}")
        print(f"  - MAPE: {report.get('mape', 'N/A'):.2f}%")
        
        print(f"\nDirection Metrics:")
        print(f"  - Directional Accuracy: {report.get('directional_accuracy', 'N/A'):.2f}%")
        
        if 'coverage' in report:
            print(f"\nUncertainty Metrics:")
            print(f"  - Coverage: {report.get('coverage', 'N/A'):.2f}%")
            print(f"  - Interval Width: {report.get('interval_width', 'N/A'):.6f}")
    
    @staticmethod
    def print_comparison(results: dict):
        print(f"\n{'='*60}")
        print("MODEL COMPARISON")
        print(f"{'='*60}")
        
        comparison = MetricsReport.compare_models(results)
        
        for metric, rankings in comparison.items():
            print(f"\n{metric.upper()}:")
            for i, model in enumerate(rankings['ranking'], 1):
                value = rankings['values'][model]
                print(f"  {i}. {model:20} = {value:10.4f}")