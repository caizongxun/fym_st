import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class ProbabilityCalibrator:
    def __init__(self, method='isotonic'):
        self.method = method
        self.calibrator = None
        self.calibration_curve = None
    
    def fit(self, y_true, y_prob):
        if self.method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(y_prob, y_true)
        
        logger.info(f"Calibrator fitted using {self.method} method")
        return self
    
    def transform(self, y_prob):
        if self.calibrator is None:
            logger.warning("Calibrator not fitted. Returning original probabilities.")
            return y_prob
        
        calibrated_prob = self.calibrator.predict(y_prob)
        return np.clip(calibrated_prob, 0, 1)
    
    def analyze_calibration(self, y_true, y_prob, n_bins=10):
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        bin_indices = np.digitize(y_prob, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        calibration_data = []
        
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                predicted_prob = y_prob[mask].mean()
                actual_prob = y_true[mask].mean()
                count = mask.sum()
                
                calibration_data.append({
                    'bin': i,
                    'predicted_prob': predicted_prob,
                    'actual_prob': actual_prob,
                    'count': count,
                    'bin_center': bin_centers[i]
                })
        
        calibration_df = pd.DataFrame(calibration_data)
        
        if len(calibration_df) > 0:
            mse = ((calibration_df['predicted_prob'] - calibration_df['actual_prob']) ** 2).mean()
            logger.info(f"Calibration MSE: {mse:.4f}")
        
        return calibration_df
    
    def get_calibration_metrics(self, y_true, y_prob):
        calibration_df = self.analyze_calibration(y_true, y_prob)
        
        if len(calibration_df) == 0:
            return {}
        
        mse = ((calibration_df['predicted_prob'] - calibration_df['actual_prob']) ** 2).mean()
        mae = (calibration_df['predicted_prob'] - calibration_df['actual_prob']).abs().mean()
        
        bias = (calibration_df['predicted_prob'] - calibration_df['actual_prob']).mean()
        
        return {
            'calibration_mse': mse,
            'calibration_mae': mae,
            'calibration_bias': bias,
            'calibration_curve': calibration_df
        }

class ModelCalibrationAnalyzer:
    def __init__(self):
        pass
    
    def analyze_model_calibration(self, model, X, y, cv=5):
        from sklearn.model_selection import cross_val_predict
        
        logger.info("Analyzing model calibration with cross-validation")
        
        y_prob = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]
        
        calibrator = ProbabilityCalibrator()
        metrics = calibrator.get_calibration_metrics(y, y_prob)
        
        logger.info(f"Calibration bias: {metrics.get('calibration_bias', 0):.4f}")
        logger.info(f"Calibration MAE: {metrics.get('calibration_mae', 0):.4f}")
        
        return metrics
    
    def compare_predictions_vs_reality(self, predictions_df, actual_results_df):
        merged = predictions_df.merge(
            actual_results_df,
            left_on='entry_time',
            right_on='entry_time',
            how='inner'
        )
        
        prob_bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        bin_labels = ['0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
        
        merged['prob_bin'] = pd.cut(merged['win_probability'], bins=prob_bins, labels=bin_labels)
        
        comparison = merged.groupby('prob_bin').agg({
            'win_probability': ['mean', 'count'],
            'actual_win': 'mean'
        }).reset_index()
        
        comparison.columns = ['prob_bin', 'avg_predicted_prob', 'count', 'actual_win_rate']
        comparison['calibration_error'] = comparison['avg_predicted_prob'] - comparison['actual_win_rate']
        
        return comparison