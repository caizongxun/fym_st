import pandas as pd
import numpy as np
import ta
from typing import Optional

class FeatureEngineer:
    def __init__(self):
        pass
    
    def create_features(self, df: pd.DataFrame, timeframe: str = '1h') -> pd.DataFrame:
        """
        Create technical indicators and features (main interface)
        
        Args:
            df: OHLCV DataFrame
            timeframe: Timeframe label (e.g., '1h', '15m')
        
        Returns:
            DataFrame with added feature columns
        """
        return self.compute_features(df, timeframe_label=f'{timeframe}_')
    
    def compute_features(self, df: pd.DataFrame, timeframe_label: str = '') -> pd.DataFrame:
        """
        Compute technical indicators and features
        
        Args:
            df: OHLCV DataFrame
            timeframe_label: Prefix for column names (e.g., '1h_' or '15m_')
        
        Returns:
            DataFrame with added feature columns
        """
        if df.empty or len(df) < 50:
            return df
        
        df = df.copy()
        prefix = timeframe_label
        
        # Price-based features
        df[f'{prefix}returns'] = df['close'].pct_change()
        df[f'{prefix}log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df[f'{prefix}price_range'] = (df['high'] - df['low']) / df['close']
        df[f'{prefix}body_size'] = abs(df['close'] - df['open']) / df['close']
        df[f'{prefix}upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
        df[f'{prefix}lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
        
        # Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            if len(df) >= period:
                df[f'{prefix}sma_{period}'] = ta.trend.sma_indicator(df['close'], window=period)
                df[f'{prefix}ema_{period}'] = ta.trend.ema_indicator(df['close'], window=period)
                df[f'{prefix}sma_{period}_dist'] = (df['close'] - df[f'{prefix}sma_{period}']) / df[f'{prefix}sma_{period}']
        
        # Trend Indicators
        df[f'{prefix}adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        df[f'{prefix}plus_di'] = ta.trend.adx_pos(df['high'], df['low'], df['close'], window=14)
        df[f'{prefix}minus_di'] = ta.trend.adx_neg(df['high'], df['low'], df['close'], window=14)
        
        macd = ta.trend.MACD(df['close'])
        df[f'{prefix}macd'] = macd.macd()
        df[f'{prefix}macd_signal'] = macd.macd_signal()
        df[f'{prefix}macd_diff'] = macd.macd_diff()
        
        # Volatility Indicators
        df[f'{prefix}atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        df[f'{prefix}atr_ratio'] = df[f'{prefix}atr'] / df['close']
        
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df[f'{prefix}bb_high'] = bb.bollinger_hband()
        df[f'{prefix}bb_low'] = bb.bollinger_lband()
        df[f'{prefix}bb_mid'] = bb.bollinger_mavg()
        df[f'{prefix}bb_width'] = (df[f'{prefix}bb_high'] - df[f'{prefix}bb_low']) / df[f'{prefix}bb_mid']
        df[f'{prefix}bb_position'] = (df['close'] - df[f'{prefix}bb_low']) / (df[f'{prefix}bb_high'] - df[f'{prefix}bb_low'])
        
        kc = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'], window=20)
        df[f'{prefix}kc_high'] = kc.keltner_channel_hband()
        df[f'{prefix}kc_low'] = kc.keltner_channel_lband()
        df[f'{prefix}kc_mid'] = kc.keltner_channel_mband()
        
        # Momentum Indicators
        df[f'{prefix}rsi'] = ta.momentum.rsi(df['close'], window=14)
        df[f'{prefix}rsi_sma'] = df[f'{prefix}rsi'].rolling(14).mean()
        
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df[f'{prefix}stoch_k'] = stoch.stoch()
        df[f'{prefix}stoch_d'] = stoch.stoch_signal()
        
        df[f'{prefix}cci'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)
        df[f'{prefix}roc'] = ta.momentum.roc(df['close'], window=12)
        df[f'{prefix}williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
        
        # Volume Indicators
        df[f'{prefix}volume_sma'] = df['volume'].rolling(20).mean()
        df[f'{prefix}volume_ratio'] = df['volume'] / df[f'{prefix}volume_sma']
        df[f'{prefix}obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        df[f'{prefix}obv_change'] = df[f'{prefix}obv'].pct_change()
        
        df[f'{prefix}mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'], window=14)
        df[f'{prefix}vwap'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'])
        
        # Price patterns
        df[f'{prefix}high_20'] = df['high'].rolling(20).max()
        df[f'{prefix}low_20'] = df['low'].rolling(20).min()
        df[f'{prefix}distance_high'] = (df[f'{prefix}high_20'] - df['close']) / df['close']
        df[f'{prefix}distance_low'] = (df['close'] - df[f'{prefix}low_20']) / df['close']
        
        # Volatility metrics
        df[f'{prefix}volatility_5'] = df[f'{prefix}returns'].rolling(5).std()
        df[f'{prefix}volatility_20'] = df[f'{prefix}returns'].rolling(20).std()
        df[f'{prefix}volatility_ratio'] = df[f'{prefix}volatility_5'] / df[f'{prefix}volatility_20']
        
        # Drop rows with NaN (from indicator calculations)
        df = df.replace([np.inf, -np.inf], np.nan)
        
        return df
    
    def merge_timeframes(self, df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> pd.DataFrame:
        """
        Merge 15m and 1h dataframes, aligning 1h features to 15m candles
        
        Args:
            df_15m: 15-minute DataFrame with features
            df_1h: 1-hour DataFrame with features
        
        Returns:
            Merged DataFrame on 15m timeframe
        """
        if df_15m.empty or df_1h.empty:
            return df_15m
        
        # Ensure datetime index
        df_15m = df_15m.copy()
        df_1h = df_1h.copy()
        
        df_15m['temp_time'] = df_15m['open_time']
        df_1h['temp_time'] = df_1h['open_time']
        
        # Round 15m times to nearest hour for alignment
        df_15m['hour_align'] = df_15m['temp_time'].dt.floor('H')
        df_1h['hour_align'] = df_1h['temp_time'].dt.floor('H')
        
        # Select 1h columns to merge
        hour_cols = [col for col in df_1h.columns if col.startswith('1h_')]
        hour_cols.append('hour_align')
        df_1h_subset = df_1h[hour_cols].copy()
        
        # Merge
        df_merged = df_15m.merge(df_1h_subset, on='hour_align', how='left', suffixes=('', '_1h'))
        
        # Clean up
        df_merged = df_merged.drop(columns=['temp_time', 'hour_align'])
        
        # Forward fill 1h values (each 1h candle applies to 4x 15m candles)
        for col in hour_cols:
            if col != 'hour_align' and col in df_merged.columns:
                df_merged[col] = df_merged[col].fillna(method='ffill')
        
        return df_merged
    
    def prepare_ml_features(self, df: pd.DataFrame, drop_time_cols: bool = True) -> pd.DataFrame:
        """
        Prepare final feature set for ML models
        
        Args:
            df: DataFrame with all features
            drop_time_cols: Whether to drop time-related columns
        
        Returns:
            Clean DataFrame ready for ML
        """
        df = df.copy()
        
        # Drop non-feature columns
        drop_cols = ['ignore']
        if drop_time_cols:
            drop_cols.extend(['open_time', 'close_time', 'open', 'high', 'low', 'close', 'volume'])
        
        for col in drop_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # Drop rows with NaN
        df = df.dropna()
        
        return df