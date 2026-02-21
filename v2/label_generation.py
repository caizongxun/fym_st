import pandas as pd
import numpy as np
from typing import Tuple, Optional


class LabelGenerator:
    def __init__(
        self,
        atr_period: int = 14,
        sl_atr_mult: float = 1.5,
        tp_atr_mult: float = 3.0,
        lookahead_bars: int = 16,
        lower_tolerance: float = 1.001,
        upper_tolerance: float = 0.999
    ):
        self.atr_period = atr_period
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.lookahead_bars = lookahead_bars
        self.lower_tolerance = lower_tolerance
        self.upper_tolerance = upper_tolerance
    
    def calculate_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = np.abs(df['high'] - df['close'].shift(1))
        df['low_close'] = np.abs(df['low'] - df['close'].shift(1))
        
        df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        
        df['atr'] = df['true_range'].rolling(window=self.atr_period).mean()
        
        df.drop(['high_low', 'high_close', 'low_close', 'true_range'], axis=1, inplace=True)
        
        return df
    
    def identify_entry_conditions(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        if 'lower' not in df.columns or 'upper' not in df.columns:
            raise ValueError("必須先計算布林帶")
        
        df['is_touching_lower'] = (df['low'] <= df['lower'] * self.lower_tolerance).astype(int)
        df['is_touching_upper'] = (df['high'] >= df['upper'] * self.upper_tolerance).astype(int)
        
        return df
    
    def calculate_stop_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        if 'atr' not in df.columns:
            raise ValueError("必須先計算 ATR")
        
        df['long_sl'] = df['close'] - (df['atr'] * self.sl_atr_mult)
        df['long_tp'] = df['close'] + (df['atr'] * self.tp_atr_mult)
        
        df['short_sl'] = df['close'] + (df['atr'] * self.sl_atr_mult)
        df['short_tp'] = df['close'] - (df['atr'] * self.tp_atr_mult)
        
        return df
    
    def generate_long_label(self, df: pd.DataFrame, idx: int) -> int:
        if idx + self.lookahead_bars >= len(df):
            return np.nan
        
        entry_price = df['close'].iloc[idx]
        sl_price = df['long_sl'].iloc[idx]
        tp_price = df['long_tp'].iloc[idx]
        
        for i in range(idx + 1, min(idx + 1 + self.lookahead_bars, len(df))):
            future_low = df['low'].iloc[i]
            future_high = df['high'].iloc[i]
            
            if future_low <= sl_price:
                return 0
            
            if future_high >= tp_price:
                return 1
        
        return 0
    
    def generate_short_label(self, df: pd.DataFrame, idx: int) -> int:
        if idx + self.lookahead_bars >= len(df):
            return np.nan
        
        entry_price = df['close'].iloc[idx]
        sl_price = df['short_sl'].iloc[idx]
        tp_price = df['short_tp'].iloc[idx]
        
        for i in range(idx + 1, min(idx + 1 + self.lookahead_bars, len(df))):
            future_low = df['low'].iloc[i]
            future_high = df['high'].iloc[i]
            
            if future_high >= sl_price:
                return 0
            
            if future_low <= tp_price:
                return 1
        
        return 0
    
    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame必須包含: {required_cols}")
        
        df = self.calculate_atr(df)
        df = self.identify_entry_conditions(df)
        df = self.calculate_stop_levels(df)
        
        target_long = []
        target_short = []
        
        for idx in range(len(df)):
            if df['is_touching_lower'].iloc[idx] == 1:
                target_long.append(self.generate_long_label(df, idx))
            else:
                target_long.append(np.nan)
            
            if df['is_touching_upper'].iloc[idx] == 1:
                target_short.append(self.generate_short_label(df, idx))
            else:
                target_short.append(np.nan)
        
        df['target_long'] = target_long
        df['target_short'] = target_short
        
        return df
    
    def prepare_training_data(self, df: pd.DataFrame, direction: str = 'long') -> pd.DataFrame:
        df = df.copy()
        
        if direction not in ['long', 'short']:
            raise ValueError("direction 必須為 'long' 或 'short'")
        
        if direction == 'long':
            df_filtered = df[df['is_touching_lower'] == 1].copy()
            df_filtered = df_filtered.dropna(subset=['target_long'])
            df_filtered['target'] = df_filtered['target_long']
        else:
            df_filtered = df[df['is_touching_upper'] == 1].copy()
            df_filtered = df_filtered.dropna(subset=['target_short'])
            df_filtered['target'] = df_filtered['target_short']
        
        return df_filtered
    
    def get_label_statistics(self, df: pd.DataFrame) -> dict:
        stats = {}
        
        if 'target_long' in df.columns:
            long_samples = df['target_long'].dropna()
            if len(long_samples) > 0:
                stats['long_total'] = len(long_samples)
                stats['long_success'] = int(long_samples.sum())
                stats['long_fail'] = int((long_samples == 0).sum())
                stats['long_success_rate'] = long_samples.mean() * 100
        
        if 'target_short' in df.columns:
            short_samples = df['target_short'].dropna()
            if len(short_samples) > 0:
                stats['short_total'] = len(short_samples)
                stats['short_success'] = int(short_samples.sum())
                stats['short_fail'] = int((short_samples == 0).sum())
                stats['short_success_rate'] = short_samples.mean() * 100
        
        return stats
