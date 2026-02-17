import pandas as pd
import numpy as np
from typing import Dict, List

class SignalGenerator:
    """
    Pure Reversal Trading Strategy - RELAXED for more signals
    """
    
    def __init__(self):
        pass
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate reversal signals with VERY RELAXED conditions
        
        Strategy (更寬鬆):
        - 任何反轉信號都接受 (忽略趨勢限制)
        - Bullish reversal -> LONG
        - Bearish reversal -> SHORT
        """
        df = df.copy()
        df['signal'] = 0
        
        # VERY RELAXED: Accept all reversal signals regardless of trend
        long_conditions = (df['reversal_direction_pred'] == 1)
        short_conditions = (df['reversal_direction_pred'] == -1)
        
        df.loc[long_conditions, 'signal'] = 1
        df.loc[short_conditions, 'signal'] = -1
        
        return df
    
    def add_signal_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        signal_map = {1: 'LONG', -1: 'SHORT', 0: 'HOLD'}
        df['signal_name'] = df['signal'].map(signal_map)
        df['signal_strength'] = df.get('reversal_prob_pred', 0)
        
        return df