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
        
        def percentile_rank(x):
            if len(x) == 0:
                return np.nan
            return (x.iloc[-1] <= x).sum() / len(x) * 100
        
        df['bandwidth_percentile'] = df['bandwidth'].rolling(window=self.lookback).apply(
            percentile_rank, raw=False
        )
        
        df['is_squeeze'] = (df['bandwidth_percentile'] < 20).astype(int)
        df['is_expansion'] = (df['bandwidth_percentile'] > 80).astype(int)
        
        return df
    
    def calculate_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        if 'basis' not in df.columns or 'dev' not in df.columns:
            raise ValueError("Must calculate Bollinger Bands first")
        
        df['z_score'] = (df['close'] - df['basis']) / df['dev']
        
        return df
    
    def calculate_pivot_points(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        highs = df['high'].values
        lows = df['low'].values
        n = len(df)
        
        pivot_highs = np.full(n, np.nan)
        pivot_lows = np.full(n, np.nan)
        
        for i in range(self.pivot_left, n - self.pivot_right):
            left_high = highs[i - self.pivot_left:i]
            right_high = highs[i + 1:i + 1 + self.pivot_right]
            if highs[i] > left_high.max() and highs[i] > right_high.max():
                pivot_highs[i] = highs[i]
            
            left_low = lows[i - self.pivot_left:i]
            right_low = lows[i + 1:i + 1 + self.pivot_right]
            if lows[i] < left_low.min() and lows[i] < right_low.min():
                pivot_lows[i] = lows[i]
        
        df['pivot_high'] = pd.Series(pivot_highs, index=df.index).shift(self.pivot_right).ffill()
        df['pivot_low'] = pd.Series(pivot_lows, index=df.index).shift(self.pivot_right).ffill()
        
        df.rename(columns={'pivot_high': 'last_ph', 'pivot_low': 'last_pl'}, inplace=True)
        
        return df
    
    def calculate_smc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        if 'last_ph' not in df.columns or 'last_pl' not in df.columns:
            raise ValueError("Must calculate pivot points first")
        
        df['bear_sweep'] = ((df['high'] > df['last_ph']) & (df['close'] < df['last_ph'])).astype(int)
        df['bull_sweep'] = ((df['low'] < df['last_pl']) & (df['close'] > df['last_pl'])).astype(int)
        
        df['bull_bos'] = (df['close'] > df['last_ph']).astype(int)
        df['bear_bos'] = (df['close'] < df['last_pl']).astype(int)
        
        return df
    
    def process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain: {required_cols}")
        
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
