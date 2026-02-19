"""Range-Bound Trading Strategy"""

import pandas as pd
import numpy as np
from typing import Dict


class RangeBoundStrategy:
    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        adx_period: int = 14,
        adx_threshold: float = 25,
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        volume_ma_period: int = 20,
        volume_threshold: float = 0.8,
        atr_period: int = 14,
        use_atr_stops: bool = True,
        atr_multiplier: float = 2.0,
        fixed_stop_pct: float = 0.02,
        target_rr: float = 2.0
    ):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.volume_ma_period = volume_ma_period
        self.volume_threshold = volume_threshold
        self.atr_period = atr_period
        self.use_atr_stops = use_atr_stops
        self.atr_multiplier = atr_multiplier
        self.fixed_stop_pct = fixed_stop_pct
        self.target_rr = target_rr
    
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""
        df = df.copy()
        
        df['bb_mid'] = df['close'].rolling(window=self.bb_period).mean()
        bb_std = df['close'].rolling(window=self.bb_period).std()
        df['bb_upper'] = df['bb_mid'] + (self.bb_std * bb_std)
        df['bb_lower'] = df['bb_mid'] - (self.bb_std * bb_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
        
        df['adx'] = self._calculate_adx(df)
        
        df['rsi'] = self._calculate_rsi(df)
        
        df['volume_ma'] = df['volume'].rolling(window=self.volume_ma_period).mean()
        
        df['atr'] = self._calculate_atr(df)
        
        return df
    
    def _calculate_rsi(self, df: pd.DataFrame) -> pd.Series:
        """Calculate RSI"""
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_adx(self, df: pd.DataFrame) -> pd.Series:
        """Calculate ADX"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.adx_period).mean()
        
        plus_di = 100 * (plus_dm.rolling(window=self.adx_period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=self.adx_period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=self.adx_period).mean()
        
        return adx
    
    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate ATR"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean()
        
        return atr
    
    def get_parameters(self) -> Dict:
        """Get strategy parameters"""
        return {
            'bb_period': self.bb_period,
            'bb_std': self.bb_std,
            'adx_threshold': self.adx_threshold,
            'rsi_oversold': self.rsi_oversold,
            'rsi_overbought': self.rsi_overbought,
            'volume_threshold': self.volume_threshold,
            'use_atr_stops': self.use_atr_stops,
            'atr_multiplier': self.atr_multiplier,
            'fixed_stop_pct': self.fixed_stop_pct,
            'target_rr': self.target_rr
        }
