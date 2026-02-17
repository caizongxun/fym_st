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
    Train the 15m Reversal Detection Model
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
        """
        Prepare training data with reversal labels
        """
        labeler = LabelGenerator()
        df_labeled = labeler.label_reversal(df_15m, horizon=10)
        
        df_labeled = df_labeled.dropna(subset=['reversal_prob', 'reversal_direction', 
                                              'support_level', 'resistance_level'])
        
        train_df, oos_df = labeler.split_train_oos(df_labeled, oos_size=oos_size)
        
        return train_df, oos_df
    
    def train(self, train_df: pd.DataFrame) -> Dict[str, float]:
        """
        Train reversal detection models
        """
        exclude_cols = ['reversal_prob', 'reversal_direction', 'support_level', 'resistance_level',
                       'open_time', 'close_time', 'open', 'high', 'low', 'close', 'volume', 'ignore']
        self.feature_cols = [col for col in train_df.columns if col not in exclude_cols]
        
        X = train_df[self.feature_cols].fillna(0)
        y_direction = train_df['reversal_direction'] + 1  # Convert -1,0,1 to 0,1,2 for classification
        y_prob = train_df['reversal_prob']
        y_support = train_df['support_level']
        y_resistance = train_df['resistance_level']
        
        X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
        y_dir_train, y_dir_val = train_test_split(y_direction, test_size=0.2, random_state=42)
        y_prob_train, y_prob_val = train_test_split(y_prob, test_size=0.2, random_state=42)
        y_sup_train, y_sup_val = train_test_split(y_support, test_size=0.2, random_state=42)
        y_res_train, y_res_val = train_test_split(y_resistance, test_size=0.2, random_state=42)
        
        print("Training reversal direction classifier...")
        self.direction_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.direction_classifier.fit(X_train, y_dir_train)
        
        print("Training reversal probability regressor...")
        self.probability_regressor = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
        self.probability_regressor.fit(X_train, y_prob_train)
        
        print("Training support level regressor...")
        self.support_regressor = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        )
        self.support_regressor.fit(X_train, y_sup_train)
        
        print("Training resistance level regressor...")
        self.resistance_regressor = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        )
        self.resistance_regressor.fit(X_train, y_res_train)
        
        # Calculate metrics
        y_dir_pred = self.direction_classifier.predict(X_val)
        y_prob_pred = self.probability_regressor.predict(X_val)
        y_sup_pred = self.support_regressor.predict(X_val)
        y_res_pred = self.resistance_regressor.predict(X_val)
        
        metrics = {
            'direction_accuracy': accuracy_score(y_dir_val, y_dir_pred),
            'probability_rmse': np.sqrt(mean_squared_error(y_prob_val, y_prob_pred)),
            'support_mae': mean_absolute_error(y_sup_val, y_sup_pred),
            'resistance_mae': mean_absolute_error(y_res_val, y_res_pred)
        }
        
        print(f"\nValidation Metrics:")
        print(f"Direction Accuracy: {metrics['direction_accuracy']:.4f}")
        print(f"Probability RMSE: {metrics['probability_rmse']:.4f}")
        print(f"Support MAE: {metrics['support_mae']:.4f}")
        print(f"Resistance MAE: {metrics['resistance_mae']:.4f}")
        print("\nDirection Classification Report:")
        print(classification_report(y_dir_val, y_dir_pred,
                                   target_names=['Bearish', 'None', 'Bullish']))
        
        return metrics
    
    def evaluate_oos(self, oos_df: pd.DataFrame) -> Dict[str, float]:
        if oos_df.empty:
            return {}
        
        X_oos = oos_df[self.feature_cols].fillna(0)
        y_dir_oos = oos_df['reversal_direction'] + 1
        y_prob_oos = oos_df['reversal_prob']
        y_sup_oos = oos_df['support_level']
        y_res_oos = oos_df['resistance_level']
        
        y_dir_pred = self.direction_classifier.predict(X_oos)
        y_prob_pred = self.probability_regressor.predict(X_oos)
        y_sup_pred = self.support_regressor.predict(X_oos)
        y_res_pred = self.resistance_regressor.predict(X_oos)
        
        metrics = {
            'oos_direction_accuracy': accuracy_score(y_dir_oos, y_dir_pred),
            'oos_probability_rmse': np.sqrt(mean_squared_error(y_prob_oos, y_prob_pred)),
            'oos_support_mae': mean_absolute_error(y_sup_oos, y_sup_pred),
            'oos_resistance_mae': mean_absolute_error(y_res_oos, y_res_pred)
        }
        
        print(f"\nOOS Validation Metrics:")
        print(f"Direction Accuracy: {metrics['oos_direction_accuracy']:.4f}")
        print(f"Probability RMSE: {metrics['oos_probability_rmse']:.4f}")
        print(f"Support MAE: {metrics['oos_support_mae']:.4f}")
        print(f"Resistance MAE: {metrics['oos_resistance_mae']:.4f}")
        
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
        
        df['reversal_direction_pred'] = self.direction_classifier.predict(X) - 1  # Back to -1,0,1
        df['reversal_prob_pred'] = self.probability_regressor.predict(X)
        df['reversal_prob_pred'] = df['reversal_prob_pred'].clip(0, 100)
        df['support_pred'] = self.support_regressor.predict(X)
        df['resistance_pred'] = self.resistance_regressor.predict(X)
        
        direction_map = {-1: 'Bearish', 0: 'None', 1: 'Bullish'}
        df['reversal_name'] = df['reversal_direction_pred'].map(direction_map)
        
        return df