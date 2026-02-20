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
        close = df['close']
        result['bb_middle'] = close.rolling(window=period).mean()
        result['bb_std'] = close.rolling(window=period).std()
        result['bb_upper'] = result['bb_middle'] + (std * result['bb_std'])
        result['bb_lower'] = result['bb_middle'] - (std * result['bb_std'])
        
        result['bb_position'] = (close - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'])
        result['bb_width_pct'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']
        
        return result
    
    def calculate_vsr(self, df: pd.DataFrame, bb_period: int = 20, lookback: int = 50) -> pd.Series:
        df_bb = self.calculate_bollinger_bands(df, period=bb_period)
        bb_width = df_bb['bb_width_pct']
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
        close = df['close']
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        result['macd_normalized'] = macd / close
        result['macd_signal'] = macd.ewm(span=signal, adjust=False).mean()
        result['macd_hist'] = macd - result['macd_signal']
        result['macd_hist_normalized'] = result['macd_hist'] / close
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
        
        if 'taker_buy_base_asset_volume' in df.columns:
            result['taker_buy_ratio'] = df['taker_buy_base_asset_volume'] / df['volume']
        elif 'taker_buy_volume' in df.columns:
            result['taker_buy_ratio'] = df['taker_buy_volume'] / df['volume']
        else:
            result['taker_buy_ratio'] = 0.5
        
        return result
    
    def calculate_ema_features(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        close = df['close']
        
        ema_9 = close.ewm(span=9, adjust=False).mean()
        ema_21 = close.ewm(span=21, adjust=False).mean()
        ema_50 = close.ewm(span=50, adjust=False).mean()
        
        result['ema_9_dist'] = (close - ema_9) / ema_9
        result['ema_21_dist'] = (close - ema_21) / ema_21
        result['ema_50_dist'] = (close - ema_50) / ema_50
        result['ema_9_21_ratio'] = ema_9 / ema_21
        result['ema_21_50_ratio'] = ema_21 / ema_50
        result['ema_cross'] = (ema_9 > ema_21).astype(int)
        
        return result
    
    def calculate_price_action(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        
        result['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        result['close_open_ratio'] = (df['close'] - df['open']) / df['open']
        
        result['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
        result['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
        result['body_size'] = abs(df['close'] - df['open']) / df['close']
        
        return result
    
    def calculate_liquidity_sweep_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        計算流動性掃蕩相關特徵
        
        包含:
        - 影線比例
        - OI 變化
        - CVD (累計成交量差)
        - 資金費率
        """
        result = df.copy()
        
        # 影線比例 (已在 price_action 中)
        body = abs(df['close'] - df['open'])
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        
        result['lower_wick_ratio'] = lower_wick / (body + 0.0001)
        result['upper_wick_ratio'] = upper_wick / (body + 0.0001)
        
        # OI 特徵
        if 'open_interest' in df.columns:
            result['oi_change_pct'] = df['open_interest'].pct_change()
            result['oi_change_1h'] = df['open_interest'].pct_change(1)
            result['oi_change_4h'] = df['open_interest'].pct_change(4)
            result['oi_change_24h'] = df['open_interest'].pct_change(24)
            result['oi_normalized'] = (df['open_interest'] - df['open_interest'].rolling(50).mean()) / (df['open_interest'].rolling(50).std() + 0.0001)
        else:
            result['oi_change_pct'] = 0
            result['oi_change_1h'] = 0
            result['oi_change_4h'] = 0
            result['oi_change_24h'] = 0
            result['oi_normalized'] = 0
        
        # CVD 特徵
        if 'taker_buy_volume' in df.columns:
            buy_volume = df['taker_buy_volume']
            sell_volume = df['volume'] - df['taker_buy_volume']
            volume_delta = buy_volume - sell_volume
            cvd = volume_delta.cumsum()
            
            result['cvd'] = cvd
            result['cvd_slope_5'] = cvd.diff(5) / 5
            result['cvd_slope_10'] = cvd.diff(10) / 10
            result['cvd_normalized'] = (cvd - cvd.rolling(50).mean()) / (cvd.rolling(50).std() + 0.0001)
        else:
            result['cvd'] = 0
            result['cvd_slope_5'] = 0
            result['cvd_slope_10'] = 0
            result['cvd_normalized'] = 0
        
        # 資金費率
        if 'funding_rate' in df.columns:
            result['funding_rate_ma_3'] = df['funding_rate'].rolling(3).mean()
            result['funding_rate_ma_7'] = df['funding_rate'].rolling(7).mean()
        else:
            result['funding_rate_ma_3'] = 0
            result['funding_rate_ma_7'] = 0
        
        # 距離支撐/壓力
        support = df['low'].rolling(window=50, min_periods=1).min()
        resistance = df['high'].rolling(window=50, min_periods=1).max()
        
        result['dist_to_support_pct'] = (df['close'] - support) / support * 100
        result['dist_to_resistance_pct'] = (resistance - df['close']) / df['close'] * 100
        
        return result
    
    def build_features(self, df: pd.DataFrame, use_fractional_diff: bool = False,
                      include_liquidity_features: bool = False) -> pd.DataFrame:
        """
        建立所有特徵
        
        Args:
            df: 原始 OHLCV 數據
            use_fractional_diff: 是否使用分數差分
            include_liquidity_features: 是否包含流動性掃蕩特徵
        """
        logger.info(f"Building features for {len(df)} rows")
        result = df.copy()
        
        # ATR
        result['atr'] = self.calculate_atr(result)
        result['atr_pct'] = result['atr'] / result['close']
        
        # Bollinger Bands
        result = self.calculate_bollinger_bands(result)
        result['vsr'] = self.calculate_vsr(result)
        
        # RSI
        result['rsi'] = self.calculate_rsi(result)
        result['rsi_normalized'] = (result['rsi'] - 50) / 50
        
        # MACD
        result = self.calculate_macd(result)
        
        # Returns
        result = self.calculate_returns(result)
        
        # Volume
        result = self.calculate_volume_features(result)
        
        # EMA
        result = self.calculate_ema_features(result)
        
        # Price Action
        result = self.calculate_price_action(result)
        
        # 流動性掃蕩特徵 (新增)
        if include_liquidity_features:
            logger.info("Adding liquidity sweep features...")
            result = self.calculate_liquidity_sweep_features(result)
        
        # Fractional Diff
        if use_fractional_diff:
            result['price_frac_diff'] = self.fractional_diff(result['close'])
        
        # Volatility & Momentum
        result['volatility_20'] = result['close'].pct_change().rolling(window=20).std()
        result['momentum_10'] = result['close'].pct_change(10)
        
        result = result.dropna()
        logger.info(f"Features built, {len(result)} rows remaining after dropna")
        
        feature_count = len([col for col in result.columns if col not in 
                           ['open_time', 'close_time', 'open', 'high', 'low', 'close', 'volume']])
        logger.info(f"Total features: {feature_count}")
        
        return result