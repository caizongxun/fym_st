import pandas as pd
import numpy as np
from typing import Dict, List

class SignalGenerator:
    """
    REDESIGNED: Trend + Binary Reversal Strategy
    
    Logic:
    1. Detect current trend (uptrend/downtrend/ranging)
    2. Wait for reversal signal (binary: yes/no)
    3. Open position in OPPOSITE direction of trend
       - Uptrend + reversal → SHORT
       - Downtrend + reversal → LONG
       - Ranging + reversal → Use momentum to decide
    """
    
    def __init__(self):
        pass
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals using trend + binary reversal
        
        Strategy:
        - Uptrend (trend=1) + reversal=1 → signal=-1 (SHORT)
        - Downtrend (trend=-1) + reversal=1 → signal=1 (LONG)
        - Ranging (trend=0) + reversal=1 → check price momentum
        """
        df = df.copy()
        df['signal'] = 0
        
        # Get trend direction and reversal signal
        trend = df['trend_direction']  # 1=up, -1=down, 0=ranging
        reversal = df['reversal_signal']  # 0=no, 1=yes (BINARY)
        
        # Calculate short-term momentum for ranging markets
        df['momentum_3'] = df['close'].pct_change(3) * 100
        
        # CORE LOGIC: Trend + Reversal → Opposite Direction
        
        # Uptrend + Reversal → Go SHORT
        short_conditions = (
            (trend == 1) & (reversal == 1)
        )
        
        # Downtrend + Reversal → Go LONG
        long_conditions = (
            (trend == -1) & (reversal == 1)
        )
        
        # Ranging + Reversal → Use momentum
        ranging_long = (
            (trend == 0) & (reversal == 1) & (df['momentum_3'] < -0.3)
        )
        
        ranging_short = (
            (trend == 0) & (reversal == 1) & (df['momentum_3'] > 0.3)
        )
        
        # Assign signals
        df.loc[long_conditions | ranging_long, 'signal'] = 1
        df.loc[short_conditions | ranging_short, 'signal'] = -1
        
        # Cleanup
        df.drop(['momentum_3'], axis=1, inplace=True)
        
        return df
    
    def add_signal_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        signal_map = {1: 'LONG', -1: 'SHORT', 0: 'HOLD'}
        df['signal_name'] = df['signal'].map(signal_map)
        df['signal_strength'] = df.get('reversal_prob_pred', 0)
        
        return df