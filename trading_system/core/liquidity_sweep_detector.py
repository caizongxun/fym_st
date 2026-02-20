import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class LiquiditySweepDetector:
    """
    流動性掃蕩偵測器
    
    基於機構視角的市場微觀結構,偵測:
    1. 價格行為: 假突破與流動性清掃
    2. 合約微觀結構: 未平倉量銳減
    3. 訂單流背離: CVD 背離
    """
    
    def __init__(self,
                 lookback_period: int = 50,
                 wick_multiplier: float = 2.0,
                 oi_std_threshold: float = 2.0,
                 cvd_divergence_lookback: int = 10):
        """
        Args:
            lookback_period: 尋找高低點的回期期間
            wick_multiplier: 影線長度必須大於實體的倍數
            oi_std_threshold: OI 下降門檻 (標準差倍數)
            cvd_divergence_lookback: CVD 背離檢測回期
        """
        self.lookback_period = lookback_period
        self.wick_multiplier = wick_multiplier
        self.oi_std_threshold = oi_std_threshold
        self.cvd_divergence_lookback = cvd_divergence_lookback
        
        logger.info(f"LiquiditySweepDetector: lookback={lookback_period}, wick_mult={wick_multiplier}")
    
    def calculate_cvd(self, df: pd.DataFrame) -> pd.Series:
        """
        計算累計成交量差 (Cumulative Volume Delta)
        
        CVD = 累計(Taker Buy Volume - Taker Sell Volume)
        """
        if 'taker_buy_volume' not in df.columns:
            logger.warning("No taker_buy_volume, calculating from buy ratio")
            # 備用: 假設買賣比 50:50
            buy_volume = df['volume'] * 0.5
            sell_volume = df['volume'] * 0.5
        else:
            buy_volume = df['taker_buy_volume']
            sell_volume = df['volume'] - df['taker_buy_volume']
        
        volume_delta = buy_volume - sell_volume
        cvd = volume_delta.cumsum()
        
        return cvd
    
    def detect_long_wick(self, df: pd.DataFrame, direction: str = 'lower') -> pd.Series:
        """
        偵測長影線 (2x 實體)
        
        Args:
            direction: 'lower' 做多信號 (長下影), 'upper' 做空信號 (長上影)
        """
        body = abs(df['close'] - df['open'])
        
        if direction == 'lower':
            # 長下影線: 下影 > 2x 實體
            lower_wick = df[['open', 'close']].min(axis=1) - df['low']
            is_long_wick = lower_wick > (body * self.wick_multiplier)
            # 且收盤價在上半部
            close_in_upper_half = df['close'] > (df['high'] + df['low']) / 2
            return is_long_wick & close_in_upper_half
        else:
            # 長上影線: 上影 > 2x 實體
            upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
            is_long_wick = upper_wick > (body * self.wick_multiplier)
            # 且收盤價在下半部
            close_in_lower_half = df['close'] < (df['high'] + df['low']) / 2
            return is_long_wick & close_in_lower_half
    
    def detect_support_resistance_breach(self, df: pd.DataFrame, direction: str = 'lower') -> pd.Series:
        """
        偵測支撑/壓力突破
        
        使用滾動窗口找過去 N 根 K 線的最高/最低點
        """
        if direction == 'lower':
            # 做多: 偵測跌破支撑 (低點)
            support = df['low'].rolling(window=self.lookback_period, min_periods=1).min().shift(1)
            breached = df['low'] < support
            # 但收盤價回到支撑之上
            recovered = df['close'] > support
            return breached & recovered
        else:
            # 做空: 偵測突破壓力 (高點)
            resistance = df['high'].rolling(window=self.lookback_period, min_periods=1).max().shift(1)
            breached = df['high'] > resistance
            recovered = df['close'] < resistance
            return breached & recovered
    
    def detect_oi_flush(self, df: pd.DataFrame) -> pd.Series:
        """
        偵測 OI 銳減 (散戶爆倉)
        
        檢測單根 K 線 OI 下降超過 24h 平均波動的 2 倍標準差
        """
        if 'open_interest' not in df.columns:
            logger.warning("No open_interest data, returning False")
            return pd.Series(False, index=df.index)
        
        # OI 變化率
        oi_change = df['open_interest'].pct_change()
        
        # 24h 滾動標準差
        oi_std_24h = oi_change.rolling(window=24, min_periods=1).std()
        
        # 偵測銳減: 下降 > threshold * std
        oi_flush = oi_change < (-self.oi_std_threshold * oi_std_24h)
        
        return oi_flush
    
    def detect_cvd_divergence(self, df: pd.DataFrame, direction: str = 'lower') -> pd.Series:
        """
        偵測 CVD 背離
        
        做多: 價格新低 (LL) 但 CVD 較高低點 (HL)
        做空: 價格新高 (HH) 但 CVD 較低高點 (LH)
        """
        cvd = self.calculate_cvd(df)
        
        lookback = self.cvd_divergence_lookback
        
        if direction == 'lower':
            # 做多: 價格低點下降, CVD 低點上升
            price_lower = df['low'] < df['low'].shift(1).rolling(window=lookback).min()
            cvd_higher = cvd > cvd.shift(1).rolling(window=lookback).min()
            return price_lower & cvd_higher
        else:
            # 做空: 價格高點上升, CVD 高點下降
            price_higher = df['high'] > df['high'].shift(1).rolling(window=lookback).max()
            cvd_lower = cvd < cvd.shift(1).rolling(window=lookback).max()
            return price_higher & cvd_lower
    
    def detect_liquidity_sweep(self, df: pd.DataFrame, direction: str = 'lower') -> pd.DataFrame:
        """
        主函數: 偵測流動性掃蕩事件
        
        必須同時符合三大標準:
        1. 價格行為: 長影線 + 突破後收回
        2. OI 銳減
        3. CVD 背離
        
        Args:
            direction: 'lower' 做多信號, 'upper' 做空信號
        
        Returns:
            df 加上 sweep 標記欄位
        """
        df = df.copy()
        
        # 1. 價格行為
        has_long_wick = self.detect_long_wick(df, direction)
        breached_level = self.detect_support_resistance_breach(df, direction)
        
        # 2. OI 銳減
        oi_flush = self.detect_oi_flush(df)
        
        # 3. CVD 背離
        cvd_divergence = self.detect_cvd_divergence(df, direction)
        
        # 三個條件共振
        liquidity_sweep = has_long_wick & breached_level & oi_flush & cvd_divergence
        
        # 記錄每個條件
        df[f'sweep_{direction}_wick'] = has_long_wick
        df[f'sweep_{direction}_breach'] = breached_level
        df[f'sweep_{direction}_oi_flush'] = oi_flush
        df[f'sweep_{direction}_cvd_div'] = cvd_divergence
        df[f'sweep_{direction}_signal'] = liquidity_sweep
        
        # 計算 CVD
        df['cvd'] = self.calculate_cvd(df)
        
        sweep_count = liquidity_sweep.sum()
        total_count = len(df)
        logger.info(f"Liquidity Sweep ({direction}): {sweep_count}/{total_count} ({100*sweep_count/total_count:.2f}%)")
        
        return df
    
    def calculate_sweep_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        計算流動性掃蕩特徵
        
        用於機器學習模型輸入
        """
        df = df.copy()
        
        # 影線比例
        body = abs(df['close'] - df['open'])
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        
        df['lower_wick_ratio'] = lower_wick / (body + 0.0001)
        df['upper_wick_ratio'] = upper_wick / (body + 0.0001)
        
        # OI 變化
        if 'open_interest' in df.columns:
            df['oi_change_pct'] = df['open_interest'].pct_change()
            df['oi_change_24h'] = df['open_interest'].pct_change(24)
        else:
            df['oi_change_pct'] = 0
            df['oi_change_24h'] = 0
        
        # CVD 斜率
        cvd = self.calculate_cvd(df)
        df['cvd'] = cvd
        df['cvd_slope'] = cvd.diff(5) / 5  # 5 根 K 線斜率
        df['cvd_normalized'] = (cvd - cvd.rolling(50).mean()) / (cvd.rolling(50).std() + 0.0001)
        
        # 距離支撑/壓力的距離
        support = df['low'].rolling(window=50, min_periods=1).min()
        resistance = df['high'].rolling(window=50, min_periods=1).max()
        
        df['dist_to_support_pct'] = (df['close'] - support) / support * 100
        df['dist_to_resistance_pct'] = (resistance - df['close']) / df['close'] * 100
        
        return df

if __name__ == "__main__":
    # 測試
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    
    from core import CryptoDataLoader
    
    loader = CryptoDataLoader()
    df = loader.fetch_latest_klines('BTCUSDT', '1h', days=30)
    
    detector = LiquiditySweepDetector()
    df_sweep = detector.detect_liquidity_sweep(df, direction='lower')
    df_sweep = detector.calculate_sweep_features(df_sweep)
    
    print(f"Total sweeps: {df_sweep['sweep_lower_signal'].sum()}")
    print(df_sweep[df_sweep['sweep_lower_signal']][['open_time', 'close', 'cvd', 'oi_change_pct']].tail())