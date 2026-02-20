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
        
        df_features = self.feature_engineer.build_features(df)
        
        if self.model_trainer.feature_columns is None:
            raise ValueError("Model feature columns not defined. Load or train a model first.")
        
        X_pred = df_features[self.model_trainer.feature_columns]
        
        probabilities = self.model_trainer.predict_proba(X_pred)
        
        df_features['win_probability'] = probabilities[:, 1]
        
        df_features['position_size'] = df_features['win_probability'].apply(
            lambda p: self.kelly_calculator.calculate_position_size(p)
        )
        
        df_features['signal'] = (df_features['position_size'] > 0).astype(int)
        
        logger.info(f"Predictions complete: {df_features['signal'].sum()} signals generated")
        
        return df_features
    
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
            'atr': latest['atr'],
            'rsi': latest['rsi'],
            'macd': latest['macd']
        }