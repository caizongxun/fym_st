import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class RealtimePredictor:
    def __init__(self, model_trainer, feature_engineer, kelly_calculator):
        self.model_trainer = model_trainer
        self.feature_engineer = feature_engineer
        self.kelly_calculator = kelly_calculator
    
    def predict_from_completed_bars(self, df: pd.DataFrame, use_last_n_bars: Optional[int] = None) -> pd.DataFrame:
        logger.info(f"Generating predictions from completed bars")
        
        if use_last_n_bars is not None:
            df = df.iloc[-use_last_n_bars:].copy()
        
        if self.model_trainer.feature_names is None:
            raise ValueError("Model feature names not defined. Load or train a model first.")
        
        exclude_cols = [
            'open_time', 'close_time', 'open', 'high', 'low', 'close', 'volume',
            'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote_volume',
            'label', 'label_return', 'hit_time',
            'exit_type', 'exit_price', 'exit_bars', 'return', 'ignore'
        ]
        
        available_features = [col for col in df.columns if col not in exclude_cols]
        
        missing_features = [f for f in self.model_trainer.feature_names if f not in available_features]
        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features, will use zeros: {missing_features[:5]}...")
        
        X_pred = pd.DataFrame(index=df.index)
        
        for feature_name in self.model_trainer.feature_names:
            if feature_name in available_features:
                X_pred[feature_name] = df[feature_name]
            else:
                X_pred[feature_name] = 0
                logger.debug(f"Feature '{feature_name}' not found, using 0")
        
        X_pred = X_pred.fillna(0)
        X_pred = X_pred.replace([np.inf, -np.inf], 0)
        
        for col in X_pred.select_dtypes(include=['bool']).columns:
            X_pred[col] = X_pred[col].astype(int)
        
        logger.info(f"Predicting with {X_pred.shape[1]} features on {len(X_pred)} samples")
        
        probabilities = self.model_trainer.predict_proba(X_pred)
        
        df = df.copy()
        df['win_probability'] = probabilities
        
        df['position_size'] = df['win_probability'].apply(
            lambda p: self.kelly_calculator.calculate_position_size(p)
        )
        
        df['signal'] = (df['position_size'] > 0).astype(int)
        
        logger.info(f"Predictions complete: {df['signal'].sum()} signals generated")
        
        return df
    
    def get_latest_signal(self, df: pd.DataFrame) -> Optional[Dict]:
        predictions = self.predict_from_completed_bars(df, use_last_n_bars=100)
        
        if len(predictions) == 0:
            return None
        
        latest = predictions.iloc[-1]
        
        if latest['signal'] == 0:
            return None
        
        return {
            'timestamp': latest['open_time'],
            'price': latest['close'],
            'win_probability': latest['win_probability'],
            'position_size': latest['position_size'],
            'atr': latest.get('atr', 0),
            'rsi': latest.get('rsi', 0),
            'macd': latest.get('macd', 0)
        }