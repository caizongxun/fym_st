import pandas as pd
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class MetaLabeling:
    def __init__(self):
        pass
    
    def generate_primary_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        
        ema_9 = df['close'].ewm(span=9, adjust=False).mean()
        ema_21 = df['close'].ewm(span=21, adjust=False).mean()
        
        macd = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        
        result['primary_signal'] = 0
        
        long_condition = (ema_9 > ema_21) & (macd > macd_signal) & (macd.shift(1) <= macd_signal.shift(1))
        result.loc[long_condition, 'primary_signal'] = 1
        
        logger.info(f"Generated {result['primary_signal'].sum()} primary signals")
        return result
    
    def prepare_meta_features(self, df: pd.DataFrame, signal_column: str = 'primary_signal') -> pd.DataFrame:
        signals_df = df[df[signal_column] == 1].copy()
        logger.info(f"Preparing meta-features for {len(signals_df)} signals")
        return signals_df