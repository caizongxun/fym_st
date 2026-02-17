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
    Binary classification: Trending (1) vs Ranging (0)
    Direction determined by technical indicators
    """
    
    def __init__(self, model_dir: str = 'models/saved'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.classifier = None  # Binary: Trending vs Ranging
        self.regressor = None   # Trend strength score
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
        Train binary classifier (trending vs ranging) and strength regressor
        """
        exclude_cols = ['trend_label', 'trend_strength', 'actual_direction', 
                       'open_time', 'close_time', 'open', 'high', 'low', 'close', 
                       'volume', 'ignore', 'net_move']
        self.feature_cols = [col for col in train_df.columns if col not in exclude_cols]
        
        X = train_df[self.feature_cols].fillna(0)
        y_class = train_df['trend_label']
        y_reg = train_df['trend_strength']
        
        # Check class distribution
        class_counts = y_class.value_counts()
        print(f"\nClass distribution in training data:")
        print(f"Ranging (0): {class_counts.get(0, 0)} samples ({class_counts.get(0, 0)/len(y_class)*100:.1f}%)")
        print(f"Trending (1): {class_counts.get(1, 0)} samples ({class_counts.get(1, 0)/len(y_class)*100:.1f}%)")
        
        # Split for validation
        X_train, X_val, y_class_train, y_class_val = train_test_split(
            X, y_class, test_size=0.2, random_state=42, stratify=y_class
        )
        _, _, y_reg_train, y_reg_val = train_test_split(
            X, y_reg, test_size=0.2, random_state=42
        )
        
        # Train binary classifier
        print("\nTraining binary trend classifier (Trending vs Ranging)...")
        self.classifier = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.02,
            max_depth=7,
            min_samples_split=15,
            min_samples_leaf=8,
            subsample=0.85,
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
        print(f"Binary Classification Accuracy: {metrics['classification_accuracy']:.4f}")
        print(f"Strength Regression RMSE: {metrics['regression_rmse']:.4f}")
        print("\nClassification Report:")
        
        try:
            print(classification_report(
                y_class_val, 
                y_class_pred, 
                labels=[0, 1],
                target_names=['Ranging', 'Trending'],
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
        print(f"Binary Classification Accuracy: {metrics['oos_classification_accuracy']:.4f}")
        print(f"Strength Regression RMSE: {metrics['oos_regression_rmse']:.4f}")
        
        try:
            print("\nOOS Classification Report:")
            print(classification_report(
                y_class_oos,
                y_class_pred,
                labels=[0, 1],
                target_names=['Ranging', 'Trending'],
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
        Make predictions and determine direction using indicators
        """
        df = df.copy()
        X = df[self.feature_cols].fillna(0)
        
        # Predict if there's a trend
        df['is_trending'] = self.classifier.predict(X)
        df['trend_strength_pred'] = self.regressor.predict(X)
        
        # Determine direction using multiple indicators
        df['trend_direction'] = self._calculate_trend_direction(df)
        
        # Combine: trend_pred = direction if trending, else 0
        df['trend_pred'] = df['trend_direction'] * df['is_trending']
        
        # Map to readable names
        trend_map = {-1: 'Bear', 0: 'Range', 1: 'Bull'}
        df['trend_name'] = df['trend_pred'].map(trend_map)
        
        return df
    
    def _calculate_trend_direction(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate trend direction using technical indicators
        Returns: 1 (bullish), -1 (bearish), 0 (neutral)
        """
        df = df.copy()
        
        # Method 1: EMA crossover (20/50)
        if '1h_ema_20' in df.columns and '1h_ema_50' in df.columns:
            ema_signal = np.where(df['1h_ema_20'] > df['1h_ema_50'], 1, -1)
        else:
            ema_signal = 0
        
        # Method 2: Price vs EMA200
        if 'close' in df.columns and '1h_ema_200' in df.columns:
            price_position = np.where(df['close'] > df['1h_ema_200'], 1, -1)
        else:
            price_position = 0
        
        # Method 3: MACD
        if '1h_macd' in df.columns:
            macd_signal = np.sign(df['1h_macd'])
        else:
            macd_signal = 0
        
        # Method 4: ADX with +DI/-DI
        if '1h_plus_di' in df.columns and '1h_minus_di' in df.columns:
            di_signal = np.where(df['1h_plus_di'] > df['1h_minus_di'], 1, -1)
        else:
            di_signal = 0
        
        # Method 5: Simple momentum (last 10 candles)
        if 'close' in df.columns:
            momentum = df['close'] - df['close'].shift(10)
            momentum_signal = np.sign(momentum)
        else:
            momentum_signal = 0
        
        # Weighted voting system
        total_signal = (
            ema_signal * 0.25 +
            price_position * 0.20 +
            macd_signal * 0.20 +
            di_signal * 0.20 +
            momentum_signal * 0.15
        )
        
        # Convert to discrete direction
        direction = pd.Series(0, index=df.index)
        direction[total_signal > 0.3] = 1   # Bullish
        direction[total_signal < -0.3] = -1  # Bearish
        
        return direction