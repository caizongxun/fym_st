import pandas as pd
import numpy as np
from typing import Dict, List

class SignalGenerator:
    """
    Pure Reversal Trading Strategy (SIMPLIFIED)
    - Always know current trend direction using indicators
    - When reversal signal appears in uptrend -> Go SHORT
    - When reversal signal appears in downtrend -> Go LONG
    - Exit and flip position on next reversal signal
    """
    
    def __init__(self,
                 min_reversal_prob: float = 75.0,
                 use_volatility_filter: bool = False):
        
        self.min_reversal_prob = min_reversal_prob
        self.use_volatility_filter = use_volatility_filter
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate flip-flop reversal signals
        
        Args:
            df: DataFrame with predictions
                Required columns:
                - trend_direction: 1 (bull), -1 (bear), 0 (neutral) from indicators
                - reversal_direction_pred: 1 (bullish reversal), -1 (bearish reversal)
                - reversal_prob_pred: 0-100
                - close: current price
        
        Returns:
            DataFrame with 'signal' column:
            - 1: Open/Flip to LONG
            - -1: Open/Flip to SHORT
            - 0: Hold current position
        """
        df = df.copy()
        df['signal'] = 0
        
        # Determine if reversal is strong enough
        df['strong_reversal'] = df['reversal_prob_pred'] >= self.min_reversal_prob
        
        # Logic: Counter-trend reversal trading
        # If in UPTREND and see BEARISH reversal -> SHORT (trend exhaustion)
        # If in DOWNTREND and see BULLISH reversal -> LONG (trend exhaustion)
        
        # Long signal: Downtrend + Bullish reversal = Bottom reversal
        long_conditions = (
            (df['trend_direction'] == -1) &  # Currently in downtrend
            (df['reversal_direction_pred'] == 1) &  # Bullish reversal detected
            df['strong_reversal']
        )
        
        # Short signal: Uptrend + Bearish reversal = Top reversal
        short_conditions = (
            (df['trend_direction'] == 1) &  # Currently in uptrend
            (df['reversal_direction_pred'] == -1) &  # Bearish reversal detected
            df['strong_reversal']
        )
        
        df.loc[long_conditions, 'signal'] = 1
        df.loc[short_conditions, 'signal'] = -1
        
        return df
    
    def add_signal_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add human-readable signal information
        """
        df = df.copy()
        
        signal_map = {1: 'LONG', -1: 'SHORT', 0: 'HOLD'}
        df['signal_name'] = df['signal'].map(signal_map)
        
        # Signal strength = reversal probability
        df['signal_strength'] = df['reversal_prob_pred']
        
        return df