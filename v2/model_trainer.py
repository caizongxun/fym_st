import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import joblib
from typing import Tuple, Optional, List
import os


class ModelTrainer:
    def __init__(
        self,
        model_type: str = 'bounce',
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 7,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        random_state: int = 42
    ):
        self.model_type = model_type
        self.model = LGBMClassifier(
            objective='binary',
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            class_weight='balanced',
            random_state=random_state,
            verbose=-1
        )
        self.feature_names = None
        self.train_score = None
        self.test_score = None
    
    def prepare_features(self, df: pd.DataFrame) -> List[str]:
        exclude_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'timestamp', 'open_time', 'close_time',
            'target', 'target_long', 'target_short',
            'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
            'ignore', 'symbol',
            'long_sl', 'long_tp', 'short_sl', 'short_tp',
            'basis', 'dev', 'upper', 'lower',
            'last_ph', 'last_pl', 'atr',
            'is_touching_lower', 'is_touching_upper'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        return feature_cols
    
    def time_series_split(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        split_idx = int(len(df) * train_ratio)
        
        df_train = df.iloc[:split_idx].copy()
        df_test = df.iloc[split_idx:].copy()
        
        return df_train, df_test
    
    def train(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.8,
        early_stopping_rounds: int = 50
    ) -> dict:
        if 'target' not in df.columns:
            raise ValueError("DataFrame must contain 'target' column")
        
        self.feature_names = self.prepare_features(df)
        
        df_train, df_test = self.time_series_split(df, train_ratio)
        
        X_train = df_train[self.feature_names]
        y_train = df_train['target']
        X_test = df_test[self.feature_names]
        y_test = df_test['target']
        
        print(f"\nTraining {self.model_type} model...")
        print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
        print(f"Features: {len(self.feature_names)}")
        print(f"Feature names: {self.feature_names}")
        print(f"Train label distribution: {y_train.value_counts().to_dict()}")
        print(f"Test label distribution: {y_test.value_counts().to_dict()}")
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[]
        )
        
        y_train_pred_proba = self.model.predict_proba(X_train)[:, 1]
        y_test_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        self.train_score = roc_auc_score(y_train, y_train_pred_proba)
        self.test_score = roc_auc_score(y_test, y_test_pred_proba)
        
        print(f"\nTraining ROC-AUC: {self.train_score:.4f}")
        print(f"Test ROC-AUC: {self.test_score:.4f}")
        
        y_test_pred = (y_test_pred_proba >= 0.5).astype(int)
        
        print("\nClassification Report (Test Set):")
        print(classification_report(y_test, y_test_pred, zero_division=0))
        
        print("\nConfusion Matrix (Test Set):")
        print(confusion_matrix(y_test, y_test_pred))
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Feature Importance:")
        print(feature_importance.head(10))
        
        results = {
            'model': self.model,
            'feature_names': self.feature_names,
            'train_auc': self.train_score,
            'test_auc': self.test_score,
            'feature_importance': feature_importance,
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        return results
    
    def save_model(self, filepath: str):
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'train_auc': self.train_score,
            'test_auc': self.test_score
        }
        
        joblib.dump(model_data, filepath)
        print(f"\nModel saved to: {filepath}")
    
    @staticmethod
    def load_model(filepath: str) -> dict:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        print(f"\nModel loaded from: {filepath}")
        print(f"Model type: {model_data['model_type']}")
        print(f"Features: {len(model_data['feature_names'])}")
        print(f"Train AUC: {model_data['train_auc']:.4f}")
        print(f"Test AUC: {model_data['test_auc']:.4f}")
        
        return model_data


class TrendFilterTrainer(ModelTrainer):
    def __init__(self, **kwargs):
        super().__init__(model_type='trend_filter', **kwargs)
    
    def prepare_filter_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        if 'target' not in df.columns:
            raise ValueError("DataFrame must contain 'target' column")
        
        if 'hit_sl' in df.columns:
            df['filter_target'] = (df['hit_sl'] == 1).astype(int)
        else:
            df['filter_target'] = (df['target'] == 0).astype(int)
        
        df['target'] = df['filter_target']
        df = df.drop('filter_target', axis=1)
        
        return df
    
    def train(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.8,
        early_stopping_rounds: int = 50
    ) -> dict:
        df_prepared = self.prepare_filter_labels(df)
        
        return super().train(df_prepared, train_ratio, early_stopping_rounds)
