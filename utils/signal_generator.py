import pandas as pd
import numpy as np
from typing import Dict, List

class SignalGenerator:
    """
    Pure Reversal Trading Strategy with ATR Risk Management
    - Detect trend on SAME timeframe (15m) for faster response
    - Every reversal signal = potential entry/flip
    - ATR-based TP/SL has priority
    - When profitable: TP hit OR reversal signal -> Flip
    - When losing: SL hit -> Flip
    """
    
    def __init__(self):
        pass
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate reversal signals on 15m timeframe
        
        Args:
            df: DataFrame with 15m predictions
                Required columns:
                - trend_direction: 1 (bull), -1 (bear), 0 (neutral)
                - reversal_direction_pred: 1 (bullish), -1 (bearish)
                - close, atr: for TP/SL calculation
        
        Returns:
            DataFrame with 'signal' column (1=LONG, -1=SHORT, 0=HOLD)
        """
        df = df.copy()
        df['signal'] = 0
        
        # Pure reversal logic - no filters
        # Long: Downtrend + Bullish reversal
        long_conditions = (
            (df['trend_direction'] == -1) &
            (df['reversal_direction_pred'] == 1)
        )
        
        # Short: Uptrend + Bearish reversal
        short_conditions = (
            (df['trend_direction'] == 1) &
            (df['reversal_direction_pred'] == -1)
        )
        
        df.loc[long_conditions, 'signal'] = 1
        df.loc[short_conditions, 'signal'] = -1
        
        return df
    
    def add_signal_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add signal metadata
        """
        df = df.copy()
        
        signal_map = {1: 'LONG', -1: 'SHORT', 0: 'HOLD'}
        df['signal_name'] = df['signal'].map(signal_map)
        df['signal_strength'] = df.get('reversal_prob_pred', 0)
        
        return df