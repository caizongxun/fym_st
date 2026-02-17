import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, mean_absolute_error
import joblib
import os
from typing import Tuple, Dict

from training.labeling import LabelGenerator

class ReversalModelTrainer:
    """
    REDESIGNED: Binary Reversal Detection Model
    
    Outputs:
    1. reversal_signal: 0 (no reversal) or 1 (has reversal) - BINARY
    2. reversal_prob: 0-100% (strength of reversal signal)
    3. support/resistance: for stop loss/take profit
    
    Trading logic:
    - Uptrend + reversal=1 → go SHORT
    - Downtrend + reversal=1 → go LONG
    """
    
    def __init__(self, model_dir: str = 'models/saved'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.signal_classifier = None  # Binary: has reversal or not
        self.probability_regressor = None
        self.support_regressor = None
        self.resistance_regressor = None
        self.feature_cols = None
    
    def prepare_data(self, df_15m: pd.DataFrame, oos_size: int = 1500) -> Tuple[pd.DataFrame, pd.DataFrame]:
        labeler = LabelGenerator()
        df_labeled = labeler.label_reversal(df_15m, horizon=10)
        
        df_labeled = df_labeled.dropna(subset=['reversal_signal', 'reversal_prob', 
                                              'support_level', 'resistance_level'])
        
        train_df, oos_df = labeler.split_train_oos(df_labeled, oos_size=oos_size)
        
        return train_df, oos_df
    
    def train(self, train_df: pd.DataFrame) -> Dict[str, float]:
        """
        Train binary reversal detection models
        """
        exclude_cols = ['reversal_signal', 'reversal_prob', 'support_level', 'resistance_level',
                       'open_time', 'close_time', 'open', 'high', 'low', 'close', 'volume', 'ignore',
                       'reversal_direction', 'reversal_direction_pred', 'reversal_name']  # Remove old columns
        self.feature_cols = [col for col in train_df.columns if col not in exclude_cols]
        
        X = train_df[self.feature_cols].fillna(0)
        y_signal = train_df['reversal_signal']  # Binary: 0 or 1
        y_prob = train_df['reversal_prob']
        y_support = train_df['support_level']
        y_resistance = train_df['resistance_level']
        
        X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
        y_sig_train, y_sig_val = train_test_split(y_signal, test_size=0.2, random_state=42)
        y_prob_train, y_prob_val = train_test_split(y_prob, test_size=0.2, random_state=42)
        y_sup_train, y_sup_val = train_test_split(y_support, test_size=0.2, random_state=42)
        y_res_train, y_res_val = train_test_split(y_resistance, test_size=0.2, random_state=42)
        
        print("Training BINARY reversal signal classifier...")
        print(f"Training samples: {len(X_train)}")
        print(f"Reversal signal distribution: {y_sig_train.value_counts().to_dict()}")
        
        # Binary classifier: has reversal or not
        self.signal_classifier = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=30,
            min_samples_leaf=15,
            subsample=0.7,
            max_features='sqrt',
            random_state=42
        )
        self.signal_classifier.fit(X_train, y_sig_train)
        
        print("Training reversal probability regressor...")
        self.probability_regressor = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            min_samples_split=40,
            min_samples_leaf=20,
            subsample=0.6,
            max_features='sqrt',
            random_state=42
        )
        self.probability_regressor.fit(X_train, y_prob_train)
        
        print("Training support level regressor...")
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
        
        print("Training resistance level regressor...")
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
        y_sig_pred = self.signal_classifier.predict(X_val)
        y_prob_pred = self.probability_regressor.predict(X_val)
        y_sup_pct_pred = self.support_regressor.predict(X_val)
        y_res_pct_pred = self.resistance_regressor.predict(X_val)
        
        y_sup_pred = current_price_val * (1 + y_sup_pct_pred / 100)
        y_res_pred = current_price_val * (1 + y_res_pct_pred / 100)
        
        metrics = {
            'signal_accuracy': accuracy_score(y_sig_val, y_sig_pred),
            'probability_rmse': np.sqrt(mean_squared_error(y_prob_val, y_prob_pred)),
            'support_mae': mean_absolute_error(y_sup_val, y_sup_pred),
            'support_mae_pct': mean_absolute_error(y_sup_pct_val, y_sup_pct_pred),
            'resistance_mae': mean_absolute_error(y_res_val, y_res_pred),
            'resistance_mae_pct': mean_absolute_error(y_res_pct_val, y_res_pct_pred)
        }
        
        print(f"\nValidation Metrics:")
        print(f"Reversal Signal Accuracy: {metrics['signal_accuracy']:.4f} (BINARY)")
        print(f"Probability RMSE: {metrics['probability_rmse']:.4f}")
        print(f"Support MAE (absolute): {metrics['support_mae']:.4f}")
        print(f"Support MAE (percentage): {metrics['support_mae_pct']:.4f}%")
        print(f"Resistance MAE (absolute): {metrics['resistance_mae']:.4f}")
        print(f"Resistance MAE (percentage): {metrics['resistance_mae_pct']:.4f}%")
        print("\nReversal Signal Classification Report:")
        print(classification_report(y_sig_val, y_sig_pred, target_names=['No Reversal', 'Has Reversal'], zero_division=0))
        
        return metrics
    
    def evaluate_oos(self, oos_df: pd.DataFrame) -> Dict[str, float]:
        if oos_df.empty:
            return {}
        
        X_oos = oos_df[self.feature_cols].fillna(0)
        y_sig_oos = oos_df['reversal_signal']
        y_prob_oos = oos_df['reversal_prob']
        y_sup_oos = oos_df['support_level']
        y_res_oos = oos_df['resistance_level']
        
        current_price_oos = oos_df['close']
        y_sup_pct_oos = (y_sup_oos - current_price_oos) / current_price_oos * 100
        y_res_pct_oos = (y_res_oos - current_price_oos) / current_price_oos * 100
        
        y_sig_pred = self.signal_classifier.predict(X_oos)
        y_prob_pred = self.probability_regressor.predict(X_oos)
        y_sup_pct_pred = self.support_regressor.predict(X_oos)
        y_res_pct_pred = self.resistance_regressor.predict(X_oos)
        
        y_sup_pred = current_price_oos * (1 + y_sup_pct_pred / 100)
        y_res_pred = current_price_oos * (1 + y_res_pct_pred / 100)
        
        metrics = {
            'oos_signal_accuracy': accuracy_score(y_sig_oos, y_sig_pred),
            'oos_probability_rmse': np.sqrt(mean_squared_error(y_prob_oos, y_prob_pred)),
            'oos_support_mae': mean_absolute_error(y_sup_oos, y_sup_pred),
            'oos_support_mae_pct': mean_absolute_error(y_sup_pct_oos, y_sup_pct_pred),
            'oos_resistance_mae': mean_absolute_error(y_res_oos, y_res_pred),
            'oos_resistance_mae_pct': mean_absolute_error(y_res_pct_oos, y_res_pct_pred)
        }
        
        print(f"\nOOS Validation Metrics:")
        print(f"Reversal Signal Accuracy: {metrics['oos_signal_accuracy']:.4f} (BINARY)")
        print(f"Probability RMSE: {metrics['oos_probability_rmse']:.4f}")
        print(f"Support MAE (absolute): {metrics['oos_support_mae']:.4f}")
        print(f"Support MAE (percentage): {metrics['oos_support_mae_pct']:.4f}%")
        print(f"Resistance MAE (absolute): {metrics['oos_resistance_mae']:.4f}")
        print(f"Resistance MAE (percentage): {metrics['oos_resistance_mae_pct']:.4f}%")
        
        return metrics
    
    def save_models(self, symbol: str):
        sig_path = os.path.join(self.model_dir, f'{symbol}_reversal_signal.pkl')
        prob_path = os.path.join(self.model_dir, f'{symbol}_reversal_probability.pkl')
        sup_path = os.path.join(self.model_dir, f'{symbol}_reversal_support.pkl')
        res_path = os.path.join(self.model_dir, f'{symbol}_reversal_resistance.pkl')
        feat_path = os.path.join(self.model_dir, f'{symbol}_reversal_features.pkl')
        
        joblib.dump(self.signal_classifier, sig_path)
        joblib.dump(self.probability_regressor, prob_path)
        joblib.dump(self.support_regressor, sup_path)
        joblib.dump(self.resistance_regressor, res_path)
        joblib.dump(self.feature_cols, feat_path)
        
        print(f"\nReversal models saved to {self.model_dir}")
    
    def load_models(self, symbol: str):
        sig_path = os.path.join(self.model_dir, f'{symbol}_reversal_signal.pkl')
        prob_path = os.path.join(self.model_dir, f'{symbol}_reversal_probability.pkl')
        sup_path = os.path.join(self.model_dir, f'{symbol}_reversal_support.pkl')
        res_path = os.path.join(self.model_dir, f'{symbol}_reversal_resistance.pkl')
        feat_path = os.path.join(self.model_dir, f'{symbol}_reversal_features.pkl')
        
        self.signal_classifier = joblib.load(sig_path)
        self.probability_regressor = joblib.load(prob_path)
        self.support_regressor = joblib.load(sup_path)
        self.resistance_regressor = joblib.load(res_path)
        self.feature_cols = joblib.load(feat_path)
        
        print(f"Reversal models loaded from {self.model_dir}")
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        X = df[self.feature_cols].fillna(0)
        
        # Binary prediction: 0 or 1 only
        df['reversal_signal'] = self.signal_classifier.predict(X)
        df['reversal_prob_pred'] = self.probability_regressor.predict(X)
        df['reversal_prob_pred'] = df['reversal_prob_pred'].clip(0, 100)
        
        # Predict support/resistance as percentage
        support_pct = self.support_regressor.predict(X)
        resistance_pct = self.resistance_regressor.predict(X)
        
        df['support_pred'] = df['close'] * (1 + support_pct / 100)
        df['resistance_pred'] = df['close'] * (1 + resistance_pct / 100)
        
        # Human-readable name
        df['reversal_name'] = df['reversal_signal'].map({0: 'None', 1: 'Reversal'})
        
        return df