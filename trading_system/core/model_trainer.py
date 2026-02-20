import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit
import joblib
import logging
from typing import Optional, Tuple, List
import os

logger = logging.getLogger(__name__)

class PurgedKFold:
    def __init__(self, n_splits: int = 5, purge_gap: int = 10):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
    
    def split(self, X: pd.DataFrame, y: pd.Series = None, groups: pd.Series = None):
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        test_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            test_start = i * test_size
            test_end = test_start + test_size if i < self.n_splits - 1 else n_samples
            
            test_indices = indices[test_start:test_end]
            
            purge_start = max(0, test_start - self.purge_gap)
            purge_end = min(n_samples, test_end + self.purge_gap)
            
            train_indices = np.concatenate([
                indices[:purge_start],
                indices[purge_end:]
            ])
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices

class ModelTrainer:
    def __init__(self, model_save_path: str = "trading_system/models"):
        self.model_save_path = model_save_path
        os.makedirs(model_save_path, exist_ok=True)
        self.model = None
        self.feature_columns = None
    
    def prepare_training_data(self, df: pd.DataFrame, target_column: str = 'label') -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        exclude_columns = [
            'open_time', 'close_time', 'label', 'label_return', 'hit_time',
            'primary_signal', 'ignore'
        ]
        
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        X = df[feature_columns]
        y = df[target_column]
        
        logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Feature columns: {feature_columns}")
        
        return X, y, feature_columns
    
    def train_with_purged_cv(self, 
                              X: pd.DataFrame, 
                              y: pd.Series, 
                              sample_weights: Optional[np.ndarray] = None,
                              n_splits: int = 5,
                              purge_gap: int = 24) -> dict:
        
        logger.info(f"Training with purged {n_splits}-fold cross-validation")
        
        cv = PurgedKFold(n_splits=n_splits, purge_gap=purge_gap)
        
        cv_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            if sample_weights is not None:
                weights_train = sample_weights[train_idx]
            else:
                weights_train = None
            
            model = LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(
                X_train, y_train,
                sample_weight=weights_train,
                eval_set=[(X_test, y_test)],
                callbacks=[]
            )
            
            score = model.score(X_test, y_test)
            cv_scores.append(score)
            logger.info(f"Fold {fold+1} accuracy: {score:.4f}")
        
        logger.info(f"Cross-validation complete. Mean accuracy: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")
        
        self.model = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X, y, sample_weight=sample_weights)
        self.feature_columns = X.columns.tolist()
        
        return {
            'cv_scores': cv_scores,
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores)
        }
    
    def save_model(self, filename: str):
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        filepath = os.path.join(self.model_save_path, filename)
        joblib.dump({
            'model': self.model,
            'feature_columns': self.feature_columns
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filename: str):
        filepath = os.path.join(self.model_save_path, filename)
        data = joblib.load(filepath)
        self.model = data['model']
        self.feature_columns = data['feature_columns']
        logger.info(f"Model loaded from {filepath}")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        if self.feature_columns is not None:
            X = X[self.feature_columns]
        
        return self.model.predict_proba(X)