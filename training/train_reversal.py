import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, mean_absolute_error
import joblib
import os
from typing import Tuple, Dict

from training.labeling import LabelGenerator

class ReversalModelTrainer:
    """
    Train the 15m Reversal Detection Model (OPTIMIZED)
    Predicts: Reversal probability, direction, and support/resistance levels
    """
    
    def __init__(self, model_dir: str = 'models/saved'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.direction_classifier = None
        self.probability_regressor = None
        self.support_regressor = None
        self.resistance_regressor = None
        self.feature_cols = None
    
    def prepare_data(self, df_15m: pd.DataFrame, oos_size: int = 1500) -> Tuple[pd.DataFrame, pd.DataFrame]:
        labeler = LabelGenerator()
        df_labeled = labeler.label_reversal(df_15m, horizon=10)
        
        df_labeled = df_labeled.dropna(subset=['reversal_prob', 'reversal_direction', 
                                              'support_level', 'resistance_level'])
        
        train_df, oos_df = labeler.split_train_oos(df_labeled, oos_size=oos_size)
        
        return train_df, oos_df
    
    def train(self, train_df: pd.DataFrame) -> Dict[str, float]:
        """
        Train reversal detection models (OPTIMIZED)
        """
        exclude_cols = ['reversal_prob', 'reversal_direction', 'support_level', 'resistance_level',
                       'open_time', 'close_time', 'open', 'high', 'low', 'close', 'volume', 'ignore']
        self.feature_cols = [col for col in train_df.columns if col not in exclude_cols]
        
        X = train_df[self.feature_cols].fillna(0)
        y_direction = train_df['reversal_direction'] + 1
        y_prob = train_df['reversal_prob']
        y_support = train_df['support_level']
        y_resistance = train_df['resistance_level']
        
        X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
        y_dir_train, y_dir_val = train_test_split(y_direction, test_size=0.2, random_state=42)
        y_prob_train, y_prob_val = train_test_split(y_prob, test_size=0.2, random_state=42)
        y_sup_train, y_sup_val = train_test_split(y_support, test_size=0.2, random_state=42)
        y_res_train, y_res_val = train_test_split(y_resistance, test_size=0.2, random_state=42)
        
        print("Training reversal direction classifier (OPTIMIZED)...")
        # Reduced complexity for RandomForest
        self.direction_classifier = RandomForestClassifier(
            n_estimators=150,      # Reduced from 200
            max_depth=8,           # Reduced from 10
            min_samples_split=20,  # Increased regularization
            min_samples_leaf=10,   # Increased regularization
            max_features='sqrt',   # Use sqrt(n_features)
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.direction_classifier.fit(X_train, y_dir_train)
        
        print("Training reversal probability regressor (OPTIMIZED - KEY FIX)...")
        # Heavily regularized to fix 4.91 → 10.67 RMSE problem
        self.probability_regressor = GradientBoostingRegressor(
            n_estimators=80,       # Reduced from 200
            learning_rate=0.1,     # Increased from 0.05
            max_depth=3,           # Reduced from 5
            min_samples_split=40,  # Heavily increased
            min_samples_leaf=20,   # Heavily increased
            subsample=0.6,         # Only use 60% of data per tree
            max_features='sqrt',
            random_state=42
        )
        self.probability_regressor.fit(X_train, y_prob_train)
        
        print("Training support level regressor (OPTIMIZED)...")
        # Use percentage-based prediction instead of absolute price
        # Convert to percentage distance from current price
        current_price_train = train_df.loc[X_train.index, 'close']
        current_price_val = train_df.loc[X_val.index, 'close']
        
        y_sup_pct_train = (y_sup_train - current_price_train) / current_price_train * 100
        y_sup_pct_val = (y_sup_val - current_price_val) / current_price_val * 100
        
        self.support_regressor = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=4,
            min_samples_split=25,
            min_samples_leaf=12,
            subsample=0.7,
            max_features='sqrt',
            random_state=42
        )
        self.support_regressor.fit(X_train, y_sup_pct_train)
        
        print("Training resistance level regressor (OPTIMIZED)...")
        y_res_pct_train = (y_res_train - current_price_train) / current_price_train * 100
        y_res_pct_val = (y_res_val - current_price_val) / current_price_val * 100
        
        self.resistance_regressor = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=4,
            min_samples_split=25,
            min_samples_leaf=12,
            subsample=0.7,
            max_features='sqrt',
            random_state=42
        )
        self.resistance_regressor.fit(X_train, y_res_pct_train)
        
        # Calculate metrics
        y_dir_pred = self.direction_classifier.predict(X_val)
        y_prob_pred = self.probability_regressor.predict(X_val)
        y_sup_pct_pred = self.support_regressor.predict(X_val)
        y_res_pct_pred = self.resistance_regressor.predict(X_val)
        
        # Convert percentage predictions back to absolute prices for metrics
        y_sup_pred = current_price_val * (1 + y_sup_pct_pred / 100)
        y_res_pred = current_price_val * (1 + y_res_pct_pred / 100)
        
        metrics = {
            'direction_accuracy': accuracy_score(y_dir_val, y_dir_pred),
            'probability_rmse': np.sqrt(mean_squared_error(y_prob_val, y_prob_pred)),
            'support_mae': mean_absolute_error(y_sup_val, y_sup_pred),
            'support_mae_pct': mean_absolute_error(y_sup_pct_val, y_sup_pct_pred),
            'resistance_mae': mean_absolute_error(y_res_val, y_res_pred),
            'resistance_mae_pct': mean_absolute_error(y_res_pct_val, y_res_pct_pred)
        }
        
        print(f"\nValidation Metrics:")
        print(f"Direction Accuracy: {metrics['direction_accuracy']:.4f}")
        print(f"Probability RMSE: {metrics['probability_rmse']:.4f}")
        print(f"Support MAE (absolute): {metrics['support_mae']:.4f}")
        print(f"Support MAE (percentage): {metrics['support_mae_pct']:.4f}%")
        print(f"Resistance MAE (absolute): {metrics['resistance_mae']:.4f}")
        print(f"Resistance MAE (percentage): {metrics['resistance_mae_pct']:.4f}%")
        print("\nDirection Classification Report:")
        
        try:
            print(classification_report(
                y_dir_val, 
                y_dir_pred,
                labels=[0, 1, 2],
                target_names=['Bearish', 'None', 'Bullish'],
                zero_division=0
            ))
        except Exception as e:
            print(f"Could not generate classification report: {e}")
        
        return metrics
    
    def evaluate_oos(self, oos_df: pd.DataFrame) -> Dict[str, float]:
        if oos_df.empty:
            return {}
        
        X_oos = oos_df[self.feature_cols].fillna(0)
        y_dir_oos = oos_df['reversal_direction'] + 1
        y_prob_oos = oos_df['reversal_prob']
        y_sup_oos = oos_df['support_level']
        y_res_oos = oos_df['resistance_level']
        
        current_price_oos = oos_df['close']
        y_sup_pct_oos = (y_sup_oos - current_price_oos) / current_price_oos * 100
        y_res_pct_oos = (y_res_oos - current_price_oos) / current_price_oos * 100
        
        y_dir_pred = self.direction_classifier.predict(X_oos)
        y_prob_pred = self.probability_regressor.predict(X_oos)
        y_sup_pct_pred = self.support_regressor.predict(X_oos)
        y_res_pct_pred = self.resistance_regressor.predict(X_oos)
        
        y_sup_pred = current_price_oos * (1 + y_sup_pct_pred / 100)
        y_res_pred = current_price_oos * (1 + y_res_pct_pred / 100)
        
        metrics = {
            'oos_direction_accuracy': accuracy_score(y_dir_oos, y_dir_pred),
            'oos_probability_rmse': np.sqrt(mean_squared_error(y_prob_oos, y_prob_pred)),
            'oos_support_mae': mean_absolute_error(y_sup_oos, y_sup_pred),
            'oos_support_mae_pct': mean_absolute_error(y_sup_pct_oos, y_sup_pct_pred),
            'oos_resistance_mae': mean_absolute_error(y_res_oos, y_res_pred),
            'oos_resistance_mae_pct': mean_absolute_error(y_res_pct_oos, y_res_pct_pred)
        }
        
        print(f"\nOOS Validation Metrics:")
        print(f"Direction Accuracy: {metrics['oos_direction_accuracy']:.4f}")
        print(f"Probability RMSE: {metrics['oos_probability_rmse']:.4f} (KEY METRIC - should be <8)")
        print(f"Support MAE (absolute): {metrics['oos_support_mae']:.4f}")
        print(f"Support MAE (percentage): {metrics['oos_support_mae_pct']:.4f}%")
        print(f"Resistance MAE (absolute): {metrics['oos_resistance_mae']:.4f}")
        print(f"Resistance MAE (percentage): {metrics['oos_resistance_mae_pct']:.4f}%")
        
        return metrics
    
    def save_models(self, symbol: str):
        dir_path = os.path.join(self.model_dir, f'{symbol}_reversal_direction.pkl')
        prob_path = os.path.join(self.model_dir, f'{symbol}_reversal_probability.pkl')
        sup_path = os.path.join(self.model_dir, f'{symbol}_reversal_support.pkl')
        res_path = os.path.join(self.model_dir, f'{symbol}_reversal_resistance.pkl')
        feat_path = os.path.join(self.model_dir, f'{symbol}_reversal_features.pkl')
        
        joblib.dump(self.direction_classifier, dir_path)
        joblib.dump(self.probability_regressor, prob_path)
        joblib.dump(self.support_regressor, sup_path)
        joblib.dump(self.resistance_regressor, res_path)
        joblib.dump(self.feature_cols, feat_path)
        
        print(f"\nReversal models saved to {self.model_dir}")
    
    def load_models(self, symbol: str):
        dir_path = os.path.join(self.model_dir, f'{symbol}_reversal_direction.pkl')
        prob_path = os.path.join(self.model_dir, f'{symbol}_reversal_probability.pkl')
        sup_path = os.path.join(self.model_dir, f'{symbol}_reversal_support.pkl')
        res_path = os.path.join(self.model_dir, f'{symbol}_reversal_resistance.pkl')
        feat_path = os.path.join(self.model_dir, f'{symbol}_reversal_features.pkl')
        
        self.direction_classifier = joblib.load(dir_path)
        self.probability_regressor = joblib.load(prob_path)
        self.support_regressor = joblib.load(sup_path)
        self.resistance_regressor = joblib.load(res_path)
        self.feature_cols = joblib.load(feat_path)
        
        print(f"Reversal models loaded from {self.model_dir}")
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        X = df[self.feature_cols].fillna(0)
        
        df['reversal_direction_pred'] = self.direction_classifier.predict(X) - 1
        df['reversal_prob_pred'] = self.probability_regressor.predict(X)
        df['reversal_prob_pred'] = df['reversal_prob_pred'].clip(0, 100)
        
        # Predict as percentage, then convert to absolute price
        support_pct = self.support_regressor.predict(X)
        resistance_pct = self.resistance_regressor.predict(X)
        
        df['support_pred'] = df['close'] * (1 + support_pct / 100)
        df['resistance_pred'] = df['close'] * (1 + resistance_pct / 100)
        
        direction_map = {-1: 'Bearish', 0: 'None', 1: 'Bullish'}
        df['reversal_name'] = df['reversal_direction_pred'].map(direction_map)
        
        return df