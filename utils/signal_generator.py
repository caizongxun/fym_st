import pandas as pd
import numpy as np
from typing import Dict, List

class SignalGenerator:
    """
    Generate trading signals by combining predictions from all three models
    """
    
    def __init__(self,
                 min_reversal_prob: float = 75.0,
                 min_trend_strength: float = 60.0,
                 min_trend_init_prob: float = 70.0,
                 volume_multiplier: float = 1.3):
        
        self.min_reversal_prob = min_reversal_prob
        self.min_trend_strength = min_trend_strength
        self.min_trend_init_prob = min_trend_init_prob
        self.volume_multiplier = volume_multiplier
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on model predictions
        
        Args:
            df: DataFrame with all predictions from three models
                Required columns:
                - trend_pred: 0-4 (Strong Bear to Strong Bull)
                - trend_strength_pred: 0-100
                - volatility_regime_pred: 0-2 (Low, Medium, High)
                - trend_init_prob_pred: 0-100
                - reversal_direction_pred: -1, 0, 1
                - reversal_prob_pred: 0-100
                - support_pred: price level
                - resistance_pred: price level
                - close: current close price
                - volume, volume_sma (if available)
        
        Returns:
            DataFrame with added 'signal' column (1=long, -1=short, 0=none)
        """
        df = df.copy()
        df['signal'] = 0
        
        # Long signal conditions
        long_conditions = (
            # Trend filter: Must be in bullish trend
            (df['trend_pred'] >= 3) &
            (df['trend_strength_pred'] >= self.min_trend_strength) &
            
            # Volatility: Trend initiation or high volatility
            (df['trend_init_prob_pred'] >= self.min_trend_init_prob) &
            
            # Reversal: Bullish reversal detected
            (df['reversal_direction_pred'] == 1) &
            (df['reversal_prob_pred'] >= self.min_reversal_prob) &
            
            # Price near support
            (df['close'] <= df['support_pred'] * 1.003)
        )
        
        # Short signal conditions
        short_conditions = (
            # Trend filter: Must be in bearish trend
            (df['trend_pred'] <= 1) &
            (df['trend_strength_pred'] >= self.min_trend_strength) &
            
            # Volatility
            (df['trend_init_prob_pred'] >= self.min_trend_init_prob) &
            
            # Reversal: Bearish reversal detected
            (df['reversal_direction_pred'] == -1) &
            (df['reversal_prob_pred'] >= self.min_reversal_prob) &
            
            # Price near resistance
            (df['close'] >= df['resistance_pred'] * 0.997)
        )
        
        # Volume confirmation (if available)
        if '15m_volume_ratio' in df.columns:
            long_conditions = long_conditions & (df['15m_volume_ratio'] >= self.volume_multiplier)
            short_conditions = short_conditions & (df['15m_volume_ratio'] >= self.volume_multiplier)
        
        df.loc[long_conditions, 'signal'] = 1
        df.loc[short_conditions, 'signal'] = -1
        
        return df
    
    def add_signal_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add human-readable signal information
        """
        df = df.copy()
        
        signal_map = {1: 'LONG', -1: 'SHORT', 0: 'NONE'}
        df['signal_name'] = df['signal'].map(signal_map)
        
        # Calculate signal strength score (0-100)
        df['signal_strength'] = (
            df['reversal_prob_pred'] * 0.4 +
            df['trend_strength_pred'] * 0.3 +
            df['trend_init_prob_pred'] * 0.3
        )
        
        return df