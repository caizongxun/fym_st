"""ML-Based Range-Bound Strategy - Strategy A"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class MLRangeBoundStrategy:
    """Machine Learning based range-bound trading strategy"""
    
    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        adx_period: int = 14,
        adx_threshold: float = 25,
        atr_period: int = 14,
        model_path: Optional[str] = None
    ):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.atr_period = atr_period
        
        self.long_model = None
        self.short_model = None
        self.scaler = StandardScaler()
        
        if model_path:
            self.load_models(model_path)
    
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators and ML features"""
        df = df.copy()
        
        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(window=self.bb_period).mean()
        bb_std = df['close'].rolling(window=self.bb_period).std()
        df['bb_upper'] = df['bb_mid'] + (self.bb_std * bb_std)
        df['bb_lower'] = df['bb_mid'] - (self.bb_std * bb_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ADX
        df['adx'] = self._calculate_adx(df)
        df['adx_change'] = df['adx'].diff()
        
        # ATR
        df['atr'] = self._calculate_atr(df)
        df['atr_change'] = df['atr'].pct_change()
        
        # Price features
        df['dist_to_upper_pct'] = (df['bb_upper'] - df['close']) / df['close'] * 100
        df['dist_to_lower_pct'] = (df['close'] - df['bb_lower']) / df['close'] * 100
        df['bb_width_pct'] = df['bb_width'] * 100
        df['bb_width_change'] = df['bb_width'].pct_change()
        
        # Volatility features
        for period in [5, 10, 20]:
            df[f'volatility_{period}'] = df['close'].rolling(period).std() / df['close'].rolling(period).mean()
        
        df['volatility_rank'] = df['volatility_20'].rank(pct=True) * 100
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['volume_change'] = df['volume'].pct_change()
        
        # Price action
        df['body_size'] = abs(df['close'] - df['open']) / df['open'] * 100
        df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open'] * 100
        df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open'] * 100
        
        # Historical touch outcomes
        df['touch_upper'] = (df['high'] >= df['bb_upper'] * 0.999).astype(int)
        df['touch_lower'] = (df['low'] <= df['bb_lower'] * 1.001).astype(int)
        df['touch_upper_count_10'] = df['touch_upper'].rolling(10).sum()
        df['touch_lower_count_10'] = df['touch_lower'].rolling(10).sum()
        
        # EMA trend
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['ema_trend'] = (df['ema_12'] > df['ema_26']).astype(int)
        
        # Clean infinite and NaN values
        df = self._clean_data(df)
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean infinite and NaN values"""
        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN with forward fill then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # If still has NaN, fill with 0
        df = df.fillna(0)
        
        return df
    
    def extract_features(self, df: pd.DataFrame, idx: int) -> pd.Series:
        """Extract features for ML model at specific index"""
        feature_names = [
            'bb_position', 'bb_width_pct', 'bb_width_change',
            'dist_to_upper_pct', 'dist_to_lower_pct',
            'adx', 'adx_change',
            'atr_change', 'volatility_5', 'volatility_10', 'volatility_20',
            'volatility_rank',
            'volume_ratio', 'volume_change',
            'body_size', 'upper_wick', 'lower_wick',
            'touch_upper_count_10', 'touch_lower_count_10',
            'ema_trend'
        ]
        
        return df.iloc[idx][feature_names]
    
    def create_labels(self, df: pd.DataFrame, forward_bars: int = 10) -> pd.DataFrame:
        """Create training labels based on future price movement"""
        df = df.copy()
        
        # Calculate future max/min
        df['future_max'] = df['high'].shift(-1).rolling(forward_bars).max()
        df['future_min'] = df['low'].shift(-1).rolling(forward_bars).min()
        
        # Define profitable scenarios
        df['label_long'] = (
            (df['close'] <= df['bb_lower']) &
            (df['adx'] < self.adx_threshold) &
            (df['future_max'] > df['close'] + 1.5 * df['atr'])
        ).astype(int)
        
        df['label_short'] = (
            (df['close'] >= df['bb_upper']) &
            (df['adx'] < self.adx_threshold) &
            (df['future_min'] < df['close'] - 1.5 * df['atr'])
        ).astype(int)
        
        return df
    
    def train(self, df: pd.DataFrame, forward_bars: int = 10) -> Dict:
        """Train long and short models"""
        df = self.add_indicators(df)
        df = self.create_labels(df, forward_bars)
        
        feature_names = [
            'bb_position', 'bb_width_pct', 'bb_width_change',
            'dist_to_upper_pct', 'dist_to_lower_pct',
            'adx', 'adx_change',
            'atr_change', 'volatility_5', 'volatility_10', 'volatility_20',
            'volatility_rank',
            'volume_ratio', 'volume_change',
            'body_size', 'upper_wick', 'lower_wick',
            'touch_upper_count_10', 'touch_lower_count_10',
            'ema_trend'
        ]
        
        # Prepare data
        df_clean = df[feature_names + ['label_long', 'label_short']].dropna()
        
        # Remove any remaining infinite values
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()
        
        X = df_clean[feature_names]
        X_scaled = self.scaler.fit_transform(X)
        
        # Train long model
        y_long = df_clean['label_long']
        self.long_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1
        )
        self.long_model.fit(X_scaled, y_long)
        
        # Train short model
        y_short = df_clean['label_short']
        self.short_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1
        )
        self.short_model.fit(X_scaled, y_short)
        
        return {
            'long_samples': int(y_long.sum()),
            'short_samples': int(y_short.sum()),
            'total_samples': len(df_clean),
            'feature_names': feature_names
        }
    
    def predict(self, df: pd.DataFrame, idx: int) -> Tuple[float, float]:
        """Predict long and short probabilities"""
        if self.long_model is None or self.short_model is None:
            return 0.0, 0.0
        
        try:
            features = self.extract_features(df, idx).values.reshape(1, -1)
            
            # Check for infinite or NaN values
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                return 0.0, 0.0
            
            features_scaled = self.scaler.transform(features)
            
            long_proba = self.long_model.predict_proba(features_scaled)[0][1]
            short_proba = self.short_model.predict_proba(features_scaled)[0][1]
            
            return float(long_proba), float(short_proba)
        except:
            return 0.0, 0.0
    
    def _calculate_adx(self, df: pd.DataFrame) -> pd.Series:
        """Calculate ADX"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.adx_period).mean()
        
        plus_di = 100 * (plus_dm.rolling(window=self.adx_period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=self.adx_period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(window=self.adx_period).mean()
        
        return adx
    
    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate ATR"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean()
        
        return atr
    
    def save_models(self, path: str):
        """Save trained models"""
        import joblib
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'long_model': self.long_model,
            'short_model': self.short_model,
            'scaler': self.scaler
        }, path)
    
    def load_models(self, path: str):
        """Load trained models"""
        import joblib
        data = joblib.load(path)
        self.long_model = data['long_model']
        self.short_model = data['short_model']
        self.scaler = data['scaler']
