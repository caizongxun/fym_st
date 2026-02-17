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
    Simplified 3-class system: Bull (1), Range (0), Bear (-1)
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
        """
        labeler = LabelGenerator()
        df_labeled = labeler.label_trend(df_1h, horizon=10)
        
        df_labeled = df_labeled.dropna(subset=['trend_label', 'trend_strength'])
        
        train_df, oos_df = labeler.split_train_oos(df_labeled, oos_size=oos_size)
        
        return train_df, oos_df
    
    def train(self, train_df: pd.DataFrame) -> Dict[str, float]:
        """
        Train both classification and regression models
        """
        exclude_cols = ['trend_label', 'trend_strength', 'open_time', 'close_time', 
                       'open', 'high', 'low', 'close', 'volume', 'ignore', 'net_move']
        self.feature_cols = [col for col in train_df.columns if col not in exclude_cols]
        
        X = train_df[self.feature_cols].fillna(0)
        y_class = train_df['trend_label']
        y_reg = train_df['trend_strength']
        
        # Check class distribution
        class_counts = y_class.value_counts()
        print(f"\nClass distribution in training data:")
        print(f"Bear (-1): {class_counts.get(-1, 0)} samples")
        print(f"Range (0): {class_counts.get(0, 0)} samples")
        print(f"Bull (1): {class_counts.get(1, 0)} samples")
        
        # Split for validation with stratification
        X_train, X_val, y_class_train, y_class_val = train_test_split(
            X, y_class, test_size=0.2, random_state=42, stratify=y_class
        )
        _, _, y_reg_train, y_reg_val = train_test_split(
            X, y_reg, test_size=0.2, random_state=42
        )
        
        # Train classifier with adjusted parameters for 3-class problem
        print("\nTraining trend classifier (3-class: Bull/Range/Bear)...")
        self.classifier = GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.03,
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42,
            verbose=0
        )
        self.classifier.fit(X_train, y_class_train)
        
        # Train regressor
        print("Training trend strength regressor...")
        self.regressor = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
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
        
        try:
            print(classification_report(
                y_class_val, 
                y_class_pred, 
                labels=[-1, 0, 1],
                target_names=['Bear', 'Range', 'Bull'],
                zero_division=0
            ))
        except Exception as e:
            print(f"Could not generate classification report: {e}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        return metrics
    
    def evaluate_oos(self, oos_df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate on out-of-sample data
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
        
        # OOS confusion matrix
        try:
            print("\nOOS Classification Report:")
            print(classification_report(
                y_class_oos,
                y_class_pred,
                labels=[-1, 0, 1],
                target_names=['Bear', 'Range', 'Bull'],
                zero_division=0
            ))
        except Exception as e:
            print(f"Could not generate OOS report: {e}")
        
        return metrics
    
    def save_models(self, symbol: str):
        classifier_path = os.path.join(self.model_dir, f'{symbol}_trend_classifier.pkl')
        regressor_path = os.path.join(self.model_dir, f'{symbol}_trend_regressor.pkl')
        features_path = os.path.join(self.model_dir, f'{symbol}_trend_features.pkl')
        
        joblib.dump(self.classifier, classifier_path)
        joblib.dump(self.regressor, regressor_path)
        joblib.dump(self.feature_cols, features_path)
        
        print(f"\nModels saved to {self.model_dir}")
    
    def load_models(self, symbol: str):
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
        """
        df = df.copy()
        X = df[self.feature_cols].fillna(0)
        
        df['trend_pred'] = self.classifier.predict(X)
        df['trend_strength_pred'] = self.regressor.predict(X)
        
        # Map trend labels to readable names
        trend_map = {-1: 'Bear', 0: 'Range', 1: 'Bull'}
        df['trend_name'] = df['trend_pred'].map(trend_map)
        
        return df