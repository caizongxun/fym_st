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
            raise ValueError("Must calculate Bollinger Bands first")
        
        df['is_touching_lower'] = (df['low'] <= df['lower'] * self.lower_tolerance).astype(int)
        df['is_touching_upper'] = (df['high'] >= df['upper'] * self.upper_tolerance).astype(int)
        
        return df
    
    def calculate_stop_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        if 'atr' not in df.columns:
            raise ValueError("Must calculate ATR first")
        
        df['long_sl'] = df['close'] - (df['atr'] * self.sl_atr_mult)
        df['long_tp'] = df['close'] + (df['atr'] * self.tp_atr_mult)
        
        df['short_sl'] = df['close'] + (df['atr'] * self.sl_atr_mult)
        df['short_tp'] = df['close'] - (df['atr'] * self.tp_atr_mult)
        
        return df
    
    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain: {required_cols}")
        
        df = self.calculate_atr(df)
        df = self.identify_entry_conditions(df)
        df = self.calculate_stop_levels(df)
        
        highs = df['high'].values
        lows = df['low'].values
        long_sls = df['long_sl'].values
        long_tps = df['long_tp'].values
        short_sls = df['short_sl'].values
        short_tps = df['short_tp'].values
        
        touch_lower = df['is_touching_lower'].values
        touch_upper = df['is_touching_upper'].values
        n = len(df)
        
        target_long = np.full(n, np.nan)
        target_short = np.full(n, np.nan)
        hit_sl_long = np.full(n, np.nan)
        hit_sl_short = np.full(n, np.nan)
        
        for i in range(n - self.lookahead_bars):
            if touch_lower[i] == 1:
                sl = long_sls[i]
                tp = long_tps[i]
                target_long[i] = 0
                hit_sl_long[i] = 0
                
                for j in range(1, self.lookahead_bars + 1):
                    future_idx = i + j
                    if lows[future_idx] <= sl:
                        target_long[i] = 0
                        hit_sl_long[i] = 1
                        break
                    elif highs[future_idx] >= tp:
                        target_long[i] = 1
                        hit_sl_long[i] = 0
                        break
            
            if touch_upper[i] == 1:
                sl = short_sls[i]
                tp = short_tps[i]
                target_short[i] = 0
                hit_sl_short[i] = 0
                
                for j in range(1, self.lookahead_bars + 1):
                    future_idx = i + j
                    if highs[future_idx] >= sl:
                        target_short[i] = 0
                        hit_sl_short[i] = 1
                        break
                    elif lows[future_idx] <= tp:
                        target_short[i] = 1
                        hit_sl_short[i] = 0
                        break
        
        df['target_long'] = target_long
        df['target_short'] = target_short
        df['hit_sl_long'] = hit_sl_long
        df['hit_sl_short'] = hit_sl_short
        
        return df
    
    def prepare_training_data(self, df: pd.DataFrame, direction: str = 'long') -> pd.DataFrame:
        df = df.copy()
        
        if direction not in ['long', 'short']:
            raise ValueError("direction must be 'long' or 'short'")
        
        if direction == 'long':
            df_filtered = df[df['is_touching_lower'] == 1].copy()
            df_filtered = df_filtered.dropna(subset=['target_long'])
            df_filtered['target'] = df_filtered['target_long']
            if 'hit_sl_long' in df_filtered.columns:
                df_filtered['hit_sl'] = df_filtered['hit_sl_long']
        else:
            df_filtered = df[df['is_touching_upper'] == 1].copy()
            df_filtered = df_filtered.dropna(subset=['target_short'])
            df_filtered['target'] = df_filtered['target_short']
            if 'hit_sl_short' in df_filtered.columns:
                df_filtered['hit_sl'] = df_filtered['hit_sl_short']
        
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
                
                if 'hit_sl_long' in df.columns:
                    hit_sl = df['hit_sl_long'].dropna()
                    if len(hit_sl) > 0:
                        stats['long_hit_sl'] = int(hit_sl.sum())
                        stats['long_timeout'] = int((long_samples == 0).sum() - hit_sl.sum())
        
        if 'target_short' in df.columns:
            short_samples = df['target_short'].dropna()
            if len(short_samples) > 0:
                stats['short_total'] = len(short_samples)
                stats['short_success'] = int(short_samples.sum())
                stats['short_fail'] = int((short_samples == 0).sum())
                stats['short_success_rate'] = short_samples.mean() * 100
                
                if 'hit_sl_short' in df.columns:
                    hit_sl = df['hit_sl_short'].dropna()
                    if len(hit_sl) > 0:
                        stats['short_hit_sl'] = int(hit_sl.sum())
                        stats['short_timeout'] = int((short_samples == 0).sum() - hit_sl.sum())
        
        return stats
