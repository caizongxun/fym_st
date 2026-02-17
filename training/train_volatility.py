import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error
import joblib
import os
from typing import Tuple, Dict

from training.labeling import LabelGenerator

class VolatilityModelTrainer:
    """
    Train the 15m Volatility Prediction Model
    Predicts: Volatility regime and trend initiation probability
    """
    
    def __init__(self, model_dir: str = 'models/saved'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.regime_classifier = None
        self.trend_init_regressor = None
        self.feature_cols = None
    
    def prepare_data(self, df_15m: pd.DataFrame, oos_size: int = 1500) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare training data with labels
        """
        labeler = LabelGenerator()
        df_labeled = labeler.label_volatility(df_15m, horizon=5)
        
        df_labeled = df_labeled.dropna(subset=['volatility_regime', 'trend_initiation_prob'])
        
        train_df, oos_df = labeler.split_train_oos(df_labeled, oos_size=oos_size)
        
        return train_df, oos_df
    
    def train(self, train_df: pd.DataFrame) -> Dict[str, float]:
        """
        Train volatility regime classifier and trend initiation regressor
        """
        exclude_cols = ['volatility_regime', 'trend_initiation_prob', 'vol_change',
                       'open_time', 'close_time', 'open', 'high', 'low', 'close', 'volume', 'ignore']
        self.feature_cols = [col for col in train_df.columns if col not in exclude_cols]
        
        X = train_df[self.feature_cols].fillna(0)
        y_regime = train_df['volatility_regime']
        y_trend_init = train_df['trend_initiation_prob']
        
        X_train, X_val, y_regime_train, y_regime_val = train_test_split(
            X, y_regime, test_size=0.2, random_state=42
        )
        _, _, y_trend_train, y_trend_val = train_test_split(
            X, y_trend_init, test_size=0.2, random_state=42
        )
        
        print("Training volatility regime classifier...")
        self.regime_classifier = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        )
        self.regime_classifier.fit(X_train, y_regime_train)
        
        print("Training trend initiation regressor...")
        self.trend_init_regressor = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        )
        self.trend_init_regressor.fit(X_train, y_trend_train)
        
        y_regime_pred = self.regime_classifier.predict(X_val)
        y_trend_pred = self.trend_init_regressor.predict(X_val)
        
        metrics = {
            'regime_accuracy': accuracy_score(y_regime_val, y_regime_pred),
            'trend_init_rmse': np.sqrt(mean_squared_error(y_trend_val, y_trend_pred))
        }
        
        print(f"\nValidation Metrics:")
        print(f"Regime Classification Accuracy: {metrics['regime_accuracy']:.4f}")
        print(f"Trend Initiation RMSE: {metrics['trend_init_rmse']:.4f}")
        print("\nRegime Classification Report:")
        print(classification_report(y_regime_val, y_regime_pred,
                                   target_names=['Low Vol', 'Medium Vol', 'High Vol']))
        
        return metrics
    
    def evaluate_oos(self, oos_df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate on out-of-sample data
        """
        if oos_df.empty:
            return {}
        
        X_oos = oos_df[self.feature_cols].fillna(0)
        y_regime_oos = oos_df['volatility_regime']
        y_trend_oos = oos_df['trend_initiation_prob']
        
        y_regime_pred = self.regime_classifier.predict(X_oos)
        y_trend_pred = self.trend_init_regressor.predict(X_oos)
        
        metrics = {
            'oos_regime_accuracy': accuracy_score(y_regime_oos, y_regime_pred),
            'oos_trend_init_rmse': np.sqrt(mean_squared_error(y_trend_oos, y_trend_pred))
        }
        
        print(f"\nOOS Validation Metrics:")
        print(f"Regime Accuracy: {metrics['oos_regime_accuracy']:.4f}")
        print(f"Trend Initiation RMSE: {metrics['oos_trend_init_rmse']:.4f}")
        
        return metrics
    
    def save_models(self, symbol: str):
        regime_path = os.path.join(self.model_dir, f'{symbol}_volatility_regime.pkl')
        trend_path = os.path.join(self.model_dir, f'{symbol}_trend_init.pkl')
        features_path = os.path.join(self.model_dir, f'{symbol}_volatility_features.pkl')
        
        joblib.dump(self.regime_classifier, regime_path)
        joblib.dump(self.trend_init_regressor, trend_path)
        joblib.dump(self.feature_cols, features_path)
        
        print(f"\nVolatility models saved to {self.model_dir}")
    
    def load_models(self, symbol: str):
        regime_path = os.path.join(self.model_dir, f'{symbol}_volatility_regime.pkl')
        trend_path = os.path.join(self.model_dir, f'{symbol}_trend_init.pkl')
        features_path = os.path.join(self.model_dir, f'{symbol}_volatility_features.pkl')
        
        self.regime_classifier = joblib.load(regime_path)
        self.trend_init_regressor = joblib.load(trend_path)
        self.feature_cols = joblib.load(features_path)
        
        print(f"Volatility models loaded from {self.model_dir}")
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        X = df[self.feature_cols].fillna(0)
        
        df['volatility_regime_pred'] = self.regime_classifier.predict(X)
        df['trend_init_prob_pred'] = self.trend_init_regressor.predict(X)
        df['trend_init_prob_pred'] = df['trend_init_prob_pred'].clip(0, 100)
        
        regime_map = {0: 'Low', 1: 'Medium', 2: 'High'}
        df['volatility_regime_name'] = df['volatility_regime_pred'].map(regime_map)
        
        return df