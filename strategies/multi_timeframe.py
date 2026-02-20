"""
Multi-Timeframe Data Loader - 多時間框架數據載入器

功能:
同時載入 15m, 1h, 1d 數據並對齊時間戳
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict


class MultiTimeframeLoader:
    """
    多時間框架數據載入器
    """
    
    def __init__(self, loader):
        """
        Args:
            loader: BinanceDataLoader 或 HuggingFaceKlineLoader
        """
        self.loader = loader
    
    def load_multi_timeframe(
        self, 
        symbol: str, 
        days: int = 150
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        載入多時間框架數據
        
        Args:
            symbol: 交易對
            days: 數據天數
            
        Returns:
            (df_15m, df_1h, df_1d)
        """
        from data.binance_loader import BinanceDataLoader
        
        if isinstance(self.loader, BinanceDataLoader):
            return self._load_from_binance(symbol, days)
        else:
            return self._load_from_huggingface(symbol, days)
    
    def _load_from_binance(self, symbol: str, days: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """從 Binance API 載入"""
        end = datetime.now()
        start = end - timedelta(days=days)
        
        df_15m = self.loader.load_historical_data(symbol, '15m', start, end)
        df_1h = self.loader.load_historical_data(symbol, '1h', start, end)
        
        # 1d 需要更長時間 (用於計算 EMA50)
        start_1d = end - timedelta(days=days + 60)
        df_1d = self.loader.load_historical_data(symbol, '1d', start_1d, end)
        
        return df_15m, df_1h, df_1d
    
    def _load_from_huggingface(self, symbol: str, days: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """從 HuggingFace 離線數據載入"""
        # 計算需要的根數
        bars_15m = days * 96  # 15分鐘 * 96 = 1天
        bars_1h = days * 24
        bars_1d = days + 60  # 多載入 60 天用於指標計算
        
        df_15m_all = self.loader.load_klines(symbol, '15m')
        df_1h_all = self.loader.load_klines(symbol, '1h')
        df_1d_all = self.loader.load_klines(symbol, '1d')
        
        df_15m = df_15m_all.tail(bars_15m).copy()
        df_1h = df_1h_all.tail(bars_1h).copy()
        df_1d = df_1d_all.tail(bars_1d).copy()
        
        return df_15m, df_1h, df_1d
    
    def align_timeframes(self, df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> pd.DataFrame:
        """
        將 15m 數據對齊到 1h
        每個 1h 時間點對應最新的 15m 數據
        
        Returns:
            包含 15m 特徵的 1h DataFrame
        """
        # 確保 index 是 datetime
        if not isinstance(df_15m.index, pd.DatetimeIndex):
            df_15m.index = pd.to_datetime(df_15m.index)
        if not isinstance(df_1h.index, pd.DatetimeIndex):
            df_1h.index = pd.to_datetime(df_1h.index)
        
        # 將 15m 時間戳向下對齊到整點
        df_15m['hour'] = df_15m.index.floor('H')
        
        # 對每個小時，取最後一根 15m K 棒的值
        df_15m_aligned = df_15m.groupby('hour').last()
        df_15m_aligned.index.name = 'datetime'
        
        # 合併到 1h
        df_merged = df_1h.copy()
        
        # 只保留需要的 15m 欄位
        cols_15m = ['close', 'rsi', 'macd_hist']  # 這些會在 market_regime 中計算
        for col in cols_15m:
            if col in df_15m_aligned.columns:
                df_merged[f'{col}_15m'] = df_15m_aligned[col]
        
        return df_merged
    
    def resample_to_higher_timeframe(self, df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
        """
        將低級別時間框架重採樣到高級別
        
        Args:
            df: 原始 DataFrame
            target_tf: 目標時間框架 ('1h', '4h', '1d')
        """
        ohlc_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        return df.resample(target_tf).agg(ohlc_dict).dropna()
    
    def validate_data(self, df_15m: pd.DataFrame, df_1h: pd.DataFrame, df_1d: pd.DataFrame) -> Dict[str, bool]:
        """
        驗證數據質量
        
        Returns:
            驗證結果字典
        """
        results = {
            '15m_sufficient': len(df_15m) >= 1000,
            '1h_sufficient': len(df_1h) >= 200,
            '1d_sufficient': len(df_1d) >= 60,
            '15m_no_gaps': self._check_no_gaps(df_15m, '15min'),
            '1h_no_gaps': self._check_no_gaps(df_1h, '1H'),
            '1d_no_gaps': self._check_no_gaps(df_1d, '1D')
        }
        return results
    
    def _check_no_gaps(self, df: pd.DataFrame, freq: str) -> bool:
        """檢查時間序列是否連續"""
        if len(df) < 2:
            return True
        
        expected_periods = len(df) - 1
        time_diff = (df.index[-1] - df.index[0])
        
        # 容忍 5% 的缺失
        if freq == '15min':
            expected_minutes = expected_periods * 15
            actual_minutes = time_diff.total_seconds() / 60
            return abs(actual_minutes - expected_minutes) / expected_minutes < 0.05
        elif freq == '1H':
            expected_hours = expected_periods
            actual_hours = time_diff.total_seconds() / 3600
            return abs(actual_hours - expected_hours) / expected_hours < 0.05
        else:  # 1D
            expected_days = expected_periods
            actual_days = time_diff.days
            return abs(actual_days - expected_days) / expected_days < 0.05
