import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        pass
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def fractional_diff(self, series: pd.Series, d: float = 0.4, threshold: float = 0.01) -> pd.Series:
        weights = [1.0]
        k = 1
        while abs(weights[-1]) > threshold:
            weight = -weights[-1] * (d - k + 1) / k
            weights.append(weight)
            k += 1
        
        weights = np.array(weights[::-1])
        result = np.convolve(series.values, weights, mode='valid')
        result = pd.Series(result, index=series.index[len(weights)-1:])
        return result
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
        result = df.copy()
        result['bb_middle'] = df['close'].rolling(window=period).mean()
        result['bb_std'] = df['close'].rolling(window=period).std()
        result['bb_upper'] = result['bb_middle'] + (std * result['bb_std'])
        result['bb_lower'] = result['bb_middle'] - (std * result['bb_std'])
        result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']
        return result
    
    def calculate_vsr(self, df: pd.DataFrame, bb_period: int = 20, lookback: int = 50) -> pd.Series:
        df_bb = self.calculate_bollinger_bands(df, period=bb_period)
        bb_width = df_bb['bb_width']
        vsr = bb_width / bb_width.rolling(window=lookback).mean()
        return vsr
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        result = df.copy()
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        result['macd'] = ema_fast - ema_slow
        result['macd_signal'] = result['macd'].ewm(span=signal, adjust=False).mean()
        result['macd_hist'] = result['macd'] - result['macd_signal']
        return result
    
    def calculate_returns(self, df: pd.DataFrame, periods: list = [1, 5, 10, 20]) -> pd.DataFrame:
        result = df.copy()
        for period in periods:
            result[f'return_{period}'] = df['close'].pct_change(period)
        return result
    
    def calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result['volume_ma_20'] = df['volume'].rolling(window=20).mean()
        result['volume_ratio'] = df['volume'] / result['volume_ma_20']
        result['taker_buy_ratio'] = df['taker_buy_base_asset_volume'] / df['volume']
        return result
    
    def build_features(self, df: pd.DataFrame, use_fractional_diff: bool = False) -> pd.DataFrame:
        logger.info(f"Building features for {len(df)} rows")
        result = df.copy()
        
        result['atr'] = self.calculate_atr(result)
        
        result = self.calculate_bollinger_bands(result)
        result['vsr'] = self.calculate_vsr(result)
        
        result['rsi'] = self.calculate_rsi(result)
        
        result = self.calculate_macd(result)
        
        result = self.calculate_returns(result)
        
        result = self.calculate_volume_features(result)
        
        if use_fractional_diff:
            result['price_frac_diff'] = self.fractional_diff(result['close'])
        
        result['ema_9'] = result['close'].ewm(span=9, adjust=False).mean()
        result['ema_21'] = result['close'].ewm(span=21, adjust=False).mean()
        result['ema_50'] = result['close'].ewm(span=50, adjust=False).mean()
        result['ema_cross'] = (result['ema_9'] > result['ema_21']).astype(int)
        
        result = result.dropna()
        logger.info(f"Features built, {len(result)} rows remaining after dropna")
        return result