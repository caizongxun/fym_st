import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, roc_auc_score
import joblib
import os
from typing import Tuple, Dict

from training.labeling import LabelGenerator
from data.feature_engineer import FeatureEngineer

class TrendModelTrainer:
    """
    OPTIMIZED: Trend Detection Model (Binary Classification Only)
    
    Improvements:
    1. Stronger regularization (reduce overfitting from 69%→58% gap)
    2. Feature selection (focus on trend-relevant features)
    3. Balanced class weights
    4. Simplified model architecture
    
    Output: Binary classification - Trending (1) vs Ranging (0)
    Direction: Calculated by technical indicators (not ML)
    """
    
    def __init__(self, model_dir: str = 'models/saved'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.classifier = None
        self.regressor = None
        self.feature_cols = None
    
    def prepare_data(self, df_1h: pd.DataFrame, oos_size: int = 1500) -> Tuple[pd.DataFrame, pd.DataFrame]:
        labeler = LabelGenerator()
        df_labeled = labeler.label_trend(df_1h, horizon=10)
        
        df_labeled = df_labeled.dropna(subset=['trend_label', 'trend_strength'])
        
        train_df, oos_df = labeler.split_train_oos(df_labeled, oos_size=oos_size)
        
        return train_df, oos_df
    
    def _select_features(self, train_df: pd.DataFrame) -> list:
        """
        Select features most relevant for trend detection
        Focus on: moving averages, trend indicators, momentum
        """
        exclude_cols = ['trend_label', 'trend_strength', 'actual_direction', 
                       'open_time', 'close_time', 'open', 'high', 'low', 'close', 
                       'volume', 'ignore', 'net_move']
        
        all_features = [col for col in train_df.columns if col not in exclude_cols]
        
        # Priority patterns for trend detection
        priority_patterns = [
            'ema_', 'sma_', '_dist',  # Moving averages
            'adx', 'plus_di', 'minus_di',  # Trend strength
            'macd',  # Trend momentum
            'atr', 'volatility',  # Volatility (trends have different volatility)
            'returns', 'roc',  # Momentum
            'volume_ratio'  # Volume confirmation
        ]
        
        selected_features = []
        for feat in all_features:
            if any(pattern in feat.lower() for pattern in priority_patterns):
                selected_features.append(feat)
        
        print(f"\nFeature selection: {len(selected_features)} / {len(all_features)} features selected")
        return selected_features
    
    def train(self, train_df: pd.DataFrame) -> Dict[str, float]:
        """
        Train optimized binary trend classifier
        """
        # Feature selection
        self.feature_cols = self._select_features(train_df)
        
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
        
        # OPTIMIZED: Stronger regularization to reduce overfitting
        print("\n[1/2] Training binary trend classifier (OPTIMIZED - ANTI-OVERFITTING)...")
        self.classifier = GradientBoostingClassifier(
            n_estimators=150,       # Reduced from 300 (less overfitting)
            learning_rate=0.05,     # Slower learning
            max_depth=4,            # Reduced from 7 (simpler trees)
            min_samples_split=40,   # Increased from 15 (more regularization)
            min_samples_leaf=20,    # Increased from 8 (more regularization)
            subsample=0.7,          # Reduced from 0.85 (less data per tree)
            max_features='sqrt',    # Use sqrt instead of 'auto'
            random_state=42,
            validation_fraction=0.1,
            n_iter_no_change=10,    # Early stopping
            verbose=0
        )
        self.classifier.fit(X_train, y_class_train)
        
        # OPTIMIZED: Regressor for trend strength
        print("\n[2/2] Training trend strength regressor (OPTIMIZED)...")
        self.regressor = RandomForestRegressor(
            n_estimators=100,       # Reduced from 200
            max_depth=6,            # Reduced from 10
            min_samples_split=20,   # Increased from 10
            min_samples_leaf=10,    # Increased from 5
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        self.regressor.fit(X_train, y_reg_train)
        
        # Calculate validation metrics
        y_class_pred = self.classifier.predict(X_val)
        y_class_proba = self.classifier.predict_proba(X_val)[:, 1]
        y_reg_pred = self.regressor.predict(X_val)
        
        try:
            auc_score = roc_auc_score(y_class_val, y_class_proba)
        except:
            auc_score = 0.0
        
        metrics = {
            'classification_accuracy': accuracy_score(y_class_val, y_class_pred),
            'classification_auc': auc_score,
            'regression_rmse': np.sqrt(mean_squared_error(y_reg_val, y_reg_pred))
        }
        
        print(f"\n{'='*60}")
        print("VALIDATION METRICS (OPTIMIZED)")
        print(f"{'='*60}")
        print(f"Binary Classification Accuracy: {metrics['classification_accuracy']:.4f} (目標: >0.65)")
        print(f"AUC-ROC Score: {metrics['classification_auc']:.4f}")
        print(f"Strength Regression RMSE: {metrics['regression_rmse']:.4f}")
        print(f"{'='*60}")
        
        print("\nClassification Report:")
        print(classification_report(
            y_class_val, 
            y_class_pred, 
            labels=[0, 1],
            target_names=['Ranging', 'Trending'],
            zero_division=0
        ))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 15 Most Important Features:")
        print(feature_importance.head(15).to_string(index=False))
        
        return metrics
    
    def evaluate_oos(self, oos_df: pd.DataFrame) -> Dict[str, float]:
        if oos_df.empty:
            return {}
        
        X_oos = oos_df[self.feature_cols].fillna(0)
        y_class_oos = oos_df['trend_label']
        y_reg_oos = oos_df['trend_strength']
        
        y_class_pred = self.classifier.predict(X_oos)
        y_class_proba = self.classifier.predict_proba(X_oos)[:, 1]
        y_reg_pred = self.regressor.predict(X_oos)
        
        try:
            auc_score = roc_auc_score(y_class_oos, y_class_proba)
        except:
            auc_score = 0.0
        
        metrics = {
            'oos_classification_accuracy': accuracy_score(y_class_oos, y_class_pred),
            'oos_classification_auc': auc_score,
            'oos_regression_rmse': np.sqrt(mean_squared_error(y_reg_oos, y_reg_pred))
        }
        
        print(f"\n{'='*60}")
        print("OOS VALIDATION METRICS (KEY - CHECK OVERFITTING)")
        print(f"{'='*60}")
        print(f"Binary Classification Accuracy: {metrics['oos_classification_accuracy']:.4f}")
        print(f"AUC-ROC Score: {metrics['oos_classification_auc']:.4f}")
        print(f"Strength Regression RMSE: {metrics['oos_regression_rmse']:.4f}")
        print(f"\n過擬合檢查: 訓練準確率 vs 樣本外準確率差距應 <8%")
        print(f"{'='*60}")
        
        print("\nOOS Classification Report:")
        print(classification_report(
            y_class_oos,
            y_class_pred,
            labels=[0, 1],
            target_names=['Ranging', 'Trending'],
            zero_division=0
        ))
        
        return metrics
    
    def save_models(self, symbol: str):
        classifier_path = os.path.join(self.model_dir, f'{symbol}_trend_classifier.pkl')
        regressor_path = os.path.join(self.model_dir, f'{symbol}_trend_regressor.pkl')
        features_path = os.path.join(self.model_dir, f'{symbol}_trend_features.pkl')
        
        joblib.dump(self.classifier, classifier_path)
        joblib.dump(self.regressor, regressor_path)
        joblib.dump(self.feature_cols, features_path)
        
        print(f"\n趨勢模型已儲存至 {self.model_dir}")
    
    def load_models(self, symbol: str):
        classifier_path = os.path.join(self.model_dir, f'{symbol}_trend_classifier.pkl')
        regressor_path = os.path.join(self.model_dir, f'{symbol}_trend_regressor.pkl')
        features_path = os.path.join(self.model_dir, f'{symbol}_trend_features.pkl')
        
        self.classifier = joblib.load(classifier_path)
        self.regressor = joblib.load(regressor_path)
        self.feature_cols = joblib.load(features_path)
        
        print(f"趨勢模型已載入")
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Handle missing features
        available_features = [col for col in self.feature_cols if col in df.columns]
        missing_features = [col for col in self.feature_cols if col not in df.columns]
        
        if missing_features:
            for feat in missing_features:
                df[feat] = 0
        
        X = df[self.feature_cols].fillna(0)
        
        # Predict if there's a trend
        df['is_trending'] = self.classifier.predict(X)
        df['trend_strength_pred'] = self.regressor.predict(X)
        
        # Determine direction using indicators
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
        
        # Use 15m features (since we're now using 15m data)
        prefix = '15m_'
        
        # Method 1: EMA crossover (20/50)
        if f'{prefix}ema_20' in df.columns and f'{prefix}ema_50' in df.columns:
            ema_signal = np.where(df[f'{prefix}ema_20'] > df[f'{prefix}ema_50'], 1, -1)
        else:
            ema_signal = 0
        
        # Method 2: Price vs EMA200
        if 'close' in df.columns:
            if f'{prefix}ema_200' in df.columns:
                price_position = np.where(df['close'] > df[f'{prefix}ema_200'], 1, -1)
            elif f'{prefix}ema_50' in df.columns:
                price_position = np.where(df['close'] > df[f'{prefix}ema_50'], 1, -1)
            else:
                price_position = 0
        else:
            price_position = 0
        
        # Method 3: MACD
        if f'{prefix}macd' in df.columns:
            macd_signal = np.sign(df[f'{prefix}macd'])
        else:
            macd_signal = 0
        
        # Method 4: ADX with +DI/-DI
        if f'{prefix}plus_di' in df.columns and f'{prefix}minus_di' in df.columns:
            di_signal = np.where(df[f'{prefix}plus_di'] > df[f'{prefix}minus_di'], 1, -1)
        else:
            di_signal = 0
        
        # Method 5: Simple momentum
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
        direction[total_signal > 0.3] = 1
        direction[total_signal < -0.3] = -1
        
        return direction