import pandas as pd
import numpy as np
from typing import Dict, List

class SignalGenerator:
    """
    Pure Reversal Trading Strategy
    - Relaxed trend detection to capture more reversal opportunities
    - Every reversal signal triggers entry/flip
    """
    
    def __init__(self):
        pass
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate reversal signals with RELAXED trend requirements
        
        Strategy:
        - Downtrend OR Neutral + Bullish reversal -> LONG
        - Uptrend OR Neutral + Bearish reversal -> SHORT
        
        This allows catching reversals even in sideways markets
        """
        df = df.copy()
        df['signal'] = 0
        
        # RELAXED LOGIC - Include neutral trend
        # Long: (Downtrend OR Neutral) + Bullish reversal
        long_conditions = (
            (df['trend_direction'] <= 0) &  # -1 or 0
            (df['reversal_direction_pred'] == 1)
        )
        
        # Short: (Uptrend OR Neutral) + Bearish reversal
        short_conditions = (
            (df['trend_direction'] >= 0) &  # 1 or 0
            (df['reversal_direction_pred'] == -1)
        )
        
        df.loc[long_conditions, 'signal'] = 1
        df.loc[short_conditions, 'signal'] = -1
        
        return df
    
    def add_signal_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        signal_map = {1: 'LONG', -1: 'SHORT', 0: 'HOLD'}
        df['signal_name'] = df['signal'].map(signal_map)
        df['signal_strength'] = df.get('reversal_prob_pred', 0)
        
        return df