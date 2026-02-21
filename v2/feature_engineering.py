import pandas as pd
import numpy as np
from typing import Tuple, Optional


class FeatureEngineer:
    def __init__(self, bb_period: int = 20, bb_std: int = 2, lookback: int = 100, pivot_left: int = 3, pivot_right: int = 3):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.lookback = lookback
        self.pivot_left = pivot_left
        self.pivot_right = pivot_right
    
    def calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['basis'] = df['close'].rolling(window=self.bb_period).mean()
        df['dev'] = df['close'].rolling(window=self.bb_period).std()
        df['upper'] = df['basis'] + (self.bb_std * df['dev'])
        df['lower'] = df['basis'] - (self.bb_std * df['dev'])
        
        df['bandwidth'] = (df['upper'] - df['lower']) / df['basis']
        
        df['bandwidth_percentile'] = df['bandwidth'].rolling(window=self.lookback).apply(
            lambda x: (x.iloc[-1] <= x).sum() / len(x) * 100 if len(x) > 0 else np.nan
        )
        
        df['is_squeeze'] = (df['bandwidth_percentile'] < 20).astype(int)
        df['is_expansion'] = (df['bandwidth_percentile'] > 80).astype(int)
        
        return df
    
    def calculate_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        if 'basis' not in df.columns or 'dev' not in df.columns:
            raise ValueError("必須先計算布林帶")
        
        df['z_score'] = (df['close'] - df['basis']) / df['dev']
        
        return df
    
    def calculate_pivot_points(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        def find_pivot_high(highs: pd.Series, idx: int) -> bool:
            if idx < self.pivot_left or idx >= len(highs) - self.pivot_right:
                return False
            center = highs.iloc[idx]
            left_max = highs.iloc[idx - self.pivot_left:idx].max()
            right_max = highs.iloc[idx + 1:idx + 1 + self.pivot_right].max()
            return center > left_max and center > right_max
        
        def find_pivot_low(lows: pd.Series, idx: int) -> bool:
            if idx < self.pivot_left or idx >= len(lows) - self.pivot_right:
                return False
            center = lows.iloc[idx]
            left_min = lows.iloc[idx - self.pivot_left:idx].min()
            right_min = lows.iloc[idx + 1:idx + 1 + self.pivot_right].min()
            return center < left_min and center < right_min
        
        pivot_highs = pd.Series([np.nan] * len(df), index=df.index)
        pivot_lows = pd.Series([np.nan] * len(df), index=df.index)
        
        for i in range(self.pivot_left, len(df) - self.pivot_right):
            if find_pivot_high(df['high'].reset_index(drop=True), i):
                pivot_highs.iloc[i] = df['high'].iloc[i]
            if find_pivot_low(df['low'].reset_index(drop=True), i):
                pivot_lows.iloc[i] = df['low'].iloc[i]
        
        df['pivot_high'] = pivot_highs.shift(self.pivot_right).ffill()
        df['pivot_low'] = pivot_lows.shift(self.pivot_right).ffill()
        
        df.rename(columns={'pivot_high': 'last_ph', 'pivot_low': 'last_pl'}, inplace=True)
        
        return df
    
    def calculate_smc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        if 'last_ph' not in df.columns or 'last_pl' not in df.columns:
            raise ValueError("必須先計算樞紐點")
        
        df['bear_sweep'] = ((df['high'] > df['last_ph']) & (df['close'] < df['last_ph'])).astype(int)
        df['bull_sweep'] = ((df['low'] < df['last_pl']) & (df['close'] > df['last_pl'])).astype(int)
        
        df['bull_bos'] = (df['close'] > df['last_ph']).astype(int)
        df['bear_bos'] = (df['close'] < df['last_pl']).astype(int)
        
        return df
    
    def process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame必須包含: {required_cols}")
        
        df = self.calculate_bollinger_bands(df)
        df = self.calculate_zscore(df)
        df = self.calculate_pivot_points(df)
        df = self.calculate_smc_features(df)
        
        df = df.dropna()
        
        return df
    
    def get_feature_columns(self) -> list:
        return [
            'basis', 'dev', 'upper', 'lower', 'bandwidth', 'bandwidth_percentile',
            'is_squeeze', 'is_expansion', 'z_score', 'last_ph', 'last_pl',
            'bear_sweep', 'bull_sweep', 'bull_bos', 'bear_bos'
        ]
