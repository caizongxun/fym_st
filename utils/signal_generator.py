import pandas as pd
import numpy as np
from typing import Dict, List

class SignalGenerator:
    """
    Pure Reversal Trading Strategy (ULTRA SIMPLIFIED)
    - Detect current trend direction using indicators (Bull/Bear/Neutral)
    - When reversal signal appears -> Flip position
    - NO filters, NO thresholds, just pure reversal signals
    """
    
    def __init__(self):
        pass  # No parameters needed
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate flip-flop reversal signals with ZERO filters
        
        Args:
            df: DataFrame with predictions
                Required columns:
                - trend_direction: 1 (bull), -1 (bear), 0 (neutral) from indicators
                - reversal_direction_pred: 1 (bullish reversal), -1 (bearish reversal)
                - close: current price
        
        Returns:
            DataFrame with 'signal' column:
            - 1: Go LONG (downtrend + bullish reversal)
            - -1: Go SHORT (uptrend + bearish reversal)
            - 0: Hold
        """
        df = df.copy()
        df['signal'] = 0
        
        # ULTRA SIMPLE LOGIC:
        # Long: Downtrend (-1) + Bullish reversal (1) = Bottom reversal
        long_conditions = (
            (df['trend_direction'] == -1) &  # In downtrend
            (df['reversal_direction_pred'] == 1)  # Bullish reversal detected
        )
        
        # Short: Uptrend (1) + Bearish reversal (-1) = Top reversal
        short_conditions = (
            (df['trend_direction'] == 1) &  # In uptrend
            (df['reversal_direction_pred'] == -1)  # Bearish reversal detected
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
        df['signal_strength'] = df.get('reversal_prob_pred', 0)
        
        return df