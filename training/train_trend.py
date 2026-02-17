import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error
import joblib
import os
from typing import Tuple, Dict

from training.labeling import LabelGenerator
from data.feature_engineer import FeatureEngineer

class TrendModelTrainer:
    """
    Train the 1h Trend Detection Model
    Predicts: Trend direction/strength and trend score
    """
    
    def __init__(self, model_dir: str = 'models/saved'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.classifier = None  # For trend label classification
        self.regressor = None   # For trend strength score
        self.feature_cols = None
    
    def prepare_data(self, df_1h: pd.DataFrame, oos_size: int = 1500) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare training data with labels
        
        Args:
            df_1h: 1h DataFrame with features
            oos_size: Size of OOS validation set
        
        Returns:
            Tuple of (train_df, oos_df)
        """
        # Generate labels
        labeler = LabelGenerator()
        df_labeled = labeler.label_trend(df_1h, horizon=10)
        
        # Drop rows with NaN labels
        df_labeled = df_labeled.dropna(subset=['trend_label', 'trend_strength'])
        
        # Split train/OOS
        train_df, oos_df = labeler.split_train_oos(df_labeled, oos_size=oos_size)
        
        return train_df, oos_df
    
    def train(self, train_df: pd.DataFrame) -> Dict[str, float]:
        """
        Train both classification and regression models
        
        Args:
            train_df: Training dataset with labels
        
        Returns:
            Dictionary with training metrics
        """
        # Select feature columns (exclude labels and metadata)
        exclude_cols = ['trend_label', 'trend_strength', 'open_time', 'close_time', 
                       'open', 'high', 'low', 'close', 'volume', 'ignore', 'net_move']
        self.feature_cols = [col for col in train_df.columns if col not in exclude_cols]
        
        X = train_df[self.feature_cols].fillna(0)
        y_class = train_df['trend_label']
        y_reg = train_df['trend_strength']
        
        # Split for validation
        X_train, X_val, y_class_train, y_class_val = train_test_split(
            X, y_class, test_size=0.2, random_state=42, stratify=y_class
        )
        _, _, y_reg_train, y_reg_val = train_test_split(
            X, y_reg, test_size=0.2, random_state=42
        )
        
        # Train classifier
        print("Training trend classifier...")
        self.classifier = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
        self.classifier.fit(X_train, y_class_train)
        
        # Train regressor
        print("Training trend strength regressor...")
        self.regressor = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.regressor.fit(X_train, y_reg_train)
        
        # Calculate validation metrics
        y_class_pred = self.classifier.predict(X_val)
        y_reg_pred = self.regressor.predict(X_val)
        
        metrics = {
            'classification_accuracy': accuracy_score(y_class_val, y_class_pred),
            'regression_rmse': np.sqrt(mean_squared_error(y_reg_val, y_reg_pred))
        }
        
        print(f"\nValidation Metrics:")
        print(f"Classification Accuracy: {metrics['classification_accuracy']:.4f}")
        print(f"Regression RMSE: {metrics['regression_rmse']:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_class_val, y_class_pred, 
                                   target_names=['Strong Bear', 'Weak Bear', 'Range', 'Weak Bull', 'Strong Bull']))
        
        return metrics
    
    def evaluate_oos(self, oos_df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate on out-of-sample data
        
        Args:
            oos_df: OOS dataset
        
        Returns:
            Dictionary with OOS metrics
        """
        if oos_df.empty:
            return {}
        
        X_oos = oos_df[self.feature_cols].fillna(0)
        y_class_oos = oos_df['trend_label']
        y_reg_oos = oos_df['trend_strength']
        
        y_class_pred = self.classifier.predict(X_oos)
        y_reg_pred = self.regressor.predict(X_oos)
        
        metrics = {
            'oos_classification_accuracy': accuracy_score(y_class_oos, y_class_pred),
            'oos_regression_rmse': np.sqrt(mean_squared_error(y_reg_oos, y_reg_pred))
        }
        
        print(f"\nOOS Validation Metrics:")
        print(f"Classification Accuracy: {metrics['oos_classification_accuracy']:.4f}")
        print(f"Regression RMSE: {metrics['oos_regression_rmse']:.4f}")
        
        return metrics
    
    def save_models(self, symbol: str):
        """
        Save trained models to disk
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
        """
        classifier_path = os.path.join(self.model_dir, f'{symbol}_trend_classifier.pkl')
        regressor_path = os.path.join(self.model_dir, f'{symbol}_trend_regressor.pkl')
        features_path = os.path.join(self.model_dir, f'{symbol}_trend_features.pkl')
        
        joblib.dump(self.classifier, classifier_path)
        joblib.dump(self.regressor, regressor_path)
        joblib.dump(self.feature_cols, features_path)
        
        print(f"\nModels saved to {self.model_dir}")
    
    def load_models(self, symbol: str):
        """
        Load trained models from disk
        
        Args:
            symbol: Trading symbol
        """
        classifier_path = os.path.join(self.model_dir, f'{symbol}_trend_classifier.pkl')
        regressor_path = os.path.join(self.model_dir, f'{symbol}_trend_regressor.pkl')
        features_path = os.path.join(self.model_dir, f'{symbol}_trend_features.pkl')
        
        self.classifier = joblib.load(classifier_path)
        self.regressor = joblib.load(regressor_path)
        self.feature_cols = joblib.load(features_path)
        
        print(f"Models loaded from {self.model_dir}")
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on new data
        
        Args:
            df: DataFrame with features
        
        Returns:
            DataFrame with predictions added
        """
        df = df.copy()
        X = df[self.feature_cols].fillna(0)
        
        df['trend_pred'] = self.classifier.predict(X)
        df['trend_strength_pred'] = self.regressor.predict(X)
        
        # Map trend labels to readable names
        trend_map = {0: 'Strong Bear', 1: 'Weak Bear', 2: 'Range', 3: 'Weak Bull', 4: 'Strong Bull'}
        df['trend_name'] = df['trend_pred'].map(trend_map)
        
        return df