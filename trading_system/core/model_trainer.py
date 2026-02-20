import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import joblib
import os
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class PurgedKFold:
    def __init__(self, n_splits=5, embargo_pct=0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        fold_size = n_samples // self.n_splits
        embargo_size = int(fold_size * self.embargo_pct)
        
        indices = np.arange(n_samples)
        
        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            
            test_indices = indices[test_start:test_end]
            
            train_indices = np.concatenate([
                indices[:max(0, test_start - embargo_size)],
                indices[min(n_samples, test_end + embargo_size):]
            ])
            
            yield train_indices, test_indices

class ModelTrainer:
    def __init__(self, use_calibration=True):
        self.model = None
        self.calibrated_model = None
        self.use_calibration = use_calibration
        self.feature_names = None
        self.training_metrics = {}
    
    def train(self, 
              X_train: pd.DataFrame, 
              y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None,
              params: Optional[Dict] = None) -> Dict:
        
        self.feature_names = X_train.columns.tolist()
        
        default_params = {
            'objective': 'binary:logistic',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'scale_pos_weight': 1.0,
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        
        if params:
            default_params.update(params)
        
        logger.info(f"Training XGBoost model with {len(X_train)} samples")
        
        self.model = xgb.XGBClassifier(**default_params)
        
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        
        train_metrics = self._evaluate(X_train, y_train, "training")
        
        if X_val is not None and y_val is not None:
            val_metrics = self._evaluate(X_val, y_val, "validation")
            self.training_metrics = {**train_metrics, **val_metrics}
        else:
            self.training_metrics = train_metrics
        
        if self.use_calibration:
            logger.info("Calibrating model probabilities...")
            self.calibrated_model = CalibratedClassifierCV(
                self.model, 
                method='isotonic', 
                cv='prefit'
            )
            
            if X_val is not None and y_val is not None:
                self.calibrated_model.fit(X_val, y_val)
                logger.info("Model calibrated on validation set")
            else:
                self.calibrated_model.fit(X_train, y_train)
                logger.info("Model calibrated on training set (not ideal)")
        
        return self.training_metrics
    
    def train_with_purged_kfold(self,
                                X: pd.DataFrame,
                                y: pd.Series,
                                n_splits: int = 5,
                                embargo_pct: float = 0.01,
                                params: Optional[Dict] = None) -> Dict:
        
        logger.info(f"Training with Purged K-Fold CV: {n_splits} splits")
        
        pkf = PurgedKFold(n_splits=n_splits, embargo_pct=embargo_pct)
        
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(pkf.split(X)):
            logger.info(f"Training fold {fold + 1}/{n_splits}")
            
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            metrics = self.train(X_train_fold, y_train_fold, X_val_fold, y_val_fold, params)
            cv_scores.append(metrics)
        
        avg_metrics = {}
        if len(cv_scores) > 0:
            for key in cv_scores[0].keys():
                values = [score.get(key, 0) for score in cv_scores]
                avg_metrics[key.replace('validation_', 'cv_val_')] = np.mean(values)
                avg_metrics[key.replace('validation_', 'cv_val_') + '_std'] = np.std(values)
        
        logger.info(f"Cross-validation complete. Avg val accuracy: {avg_metrics.get('cv_val_accuracy', 0):.4f}")
        
        self.training_metrics = avg_metrics
        
        return avg_metrics
    
    def _evaluate(self, X: pd.DataFrame, y: pd.Series, dataset_name: str) -> Dict:
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)[:, 1]
        
        metrics = {
            f"{dataset_name}_accuracy": accuracy_score(y, y_pred),
            f"{dataset_name}_precision": precision_score(y, y_pred, zero_division=0),
            f"{dataset_name}_recall": recall_score(y, y_pred, zero_division=0),
            f"{dataset_name}_f1": f1_score(y, y_pred, zero_division=0),
            f"{dataset_name}_auc": roc_auc_score(y, y_prob)
        }
        
        logger.info(f"{dataset_name} - Accuracy: {metrics[f'{dataset_name}_accuracy']:.4f}, "
                   f"AUC: {metrics[f'{dataset_name}_auc']:.4f}")
        
        return metrics
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if X.shape[1] != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} features, got {X.shape[1]}")
        
        if self.use_calibration and self.calibrated_model is not None:
            return self.calibrated_model.predict_proba(X)[:, 1]
        else:
            return self.model.predict_proba(X)[:, 1]
    
    def save_model(self, filename: str):
        os.makedirs("trading_system/models", exist_ok=True)
        filepath = os.path.join("trading_system/models", filename)
        
        model_data = {
            'model': self.model,
            'calibrated_model': self.calibrated_model if self.use_calibration else None,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'use_calibration': self.use_calibration
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filename: str):
        filepath = os.path.join("trading_system/models", filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.calibrated_model = model_data.get('calibrated_model')
        self.feature_names = model_data['feature_names']
        self.training_metrics = model_data.get('training_metrics', {})
        self.use_calibration = model_data.get('use_calibration', False)
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df