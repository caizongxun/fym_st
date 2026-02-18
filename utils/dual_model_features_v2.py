import pandas as pd
import numpy as np
import ta
from typing import Tuple, Dict

class EnhancedDualModelFeatureExtractor:
    """
    增強版雙模型特徵提取器
    
    改進:
    1. 訂單流特徵 (買賣壓力)
    2. 價格動量特徵
    3. 多時間框架特徵
    4. 市場微觀結構特徵
    5. 更強的趨勢特徵
    
    目標: 提升方向預測準確率到 55-60%
    """
    
    def __init__(self, lookback_candles: int = 20):
        self.lookback_candles = lookback_candles
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # ===== 基礎價格特徵 =====
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # 多期報酬率
        for period in [1, 2, 3, 5, 10]:
            df[f'returns_{period}'] = df['close'].pct_change(period)
        
        # ===== 訂單流特徵 (關鍵!) =====
        # 買賣壓力 - 根據收盤位置判斷
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
        df['buy_pressure'] = df['close_position'] * df['volume']
        df['sell_pressure'] = (1 - df['close_position']) * df['volume']
        
        # 買賣壓力比
        df['pressure_ratio'] = df['buy_pressure'] / df['sell_pressure'].replace(0, 1)
        df['pressure_ratio'] = df['pressure_ratio'].clip(0.1, 10)  # 限制極端值
        
        # 累積買賣壓力
        df['cum_buy_pressure_5'] = df['buy_pressure'].rolling(5).sum()
        df['cum_sell_pressure_5'] = df['sell_pressure'].rolling(5).sum()
        df['net_pressure_5'] = df['cum_buy_pressure_5'] - df['cum_sell_pressure_5']
        
        # ===== K棒形態特徵 =====
        df['body_size'] = abs(df['close'] - df['open']) / df['open'].replace(0, np.nan)
        df['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['open'].replace(0, np.nan)
        df['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['open'].replace(0, np.nan)
        
        # K棒方向
        df['candle_direction'] = (df['close'] > df['open']).astype(int)
        df['prev_candle_direction'] = df['candle_direction'].shift(1)
        
        # 連續漲跌
        df['consecutive_up'] = (df['candle_direction'] == 1).astype(int).groupby(
            (df['candle_direction'] != df['candle_direction'].shift()).cumsum()
        ).cumsum()
        df['consecutive_down'] = (df['candle_direction'] == 0).astype(int).groupby(
            (df['candle_direction'] != df['candle_direction'].shift()).cumsum()
        ).cumsum()
        
        # ===== 波動率特徵 =====
        df['volatility_5'] = df['returns'].rolling(5).std()
        df['volatility_10'] = df['returns'].rolling(10).std()
        df['volatility_20'] = df['returns'].rolling(20).std()
        df['volatility_ratio'] = df['volatility_5'] / df['volatility_20'].replace(0, np.nan)
        
        # Parkinson波動率 (使用高低價)
        df['parkinson_vol'] = np.sqrt(
            (np.log(df['high'] / df['low'])**2 / (4 * np.log(2)))
        ).rolling(10).mean()
        
        # ===== 動量特徵 (增強版) =====
        df['rsi_7'] = ta.momentum.RSIIndicator(df['close'], window=7).rsi()
        df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['rsi_21'] = ta.momentum.RSIIndicator(df['close'], window=21).rsi()
        
        # RSI變化率
        df['rsi_change'] = df['rsi_14'].diff()
        df['rsi_momentum'] = df['rsi_14'] - df['rsi_14'].shift(3)
        
        # MACD
        macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        df['macd_diff_change'] = df['macd_diff'].diff()
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        df['stoch_cross'] = df['stoch_k'] - df['stoch_d']
        
        # ===== 趨勢特徵 (增強版) =====
        # 多條移動平均
        df['ema_5'] = df['close'].ewm(span=5).mean()
        df['ema_10'] = df['close'].ewm(span=10).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        
        # 價格相對MA位置
        df['price_to_ema5'] = (df['close'] - df['ema_5']) / df['ema_5'].replace(0, np.nan)
        df['price_to_ema20'] = (df['close'] - df['ema_20']) / df['ema_20'].replace(0, np.nan)
        
        # MA排列 (多頭/空頭)
        df['ema_alignment'] = (
            ((df['ema_5'] > df['ema_10']) & (df['ema_10'] > df['ema_20'])).astype(int) - 
            ((df['ema_5'] < df['ema_10']) & (df['ema_10'] < df['ema_20'])).astype(int)
        )
        
        # MA斜率
        df['ema5_slope'] = df['ema_5'].pct_change()
        df['ema20_slope'] = df['ema_20'].pct_change()
        
        # ADX
        adx_indicator = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx_indicator.adx()
        df['adx_pos'] = adx_indicator.adx_pos()
        df['adx_neg'] = adx_indicator.adx_neg()
        df['di_diff'] = df['adx_pos'] - df['adx_neg']
        
        # ===== Bollinger Bands =====
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        
        bb_range = (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)
        df['bb_width'] = bb_range / df['bb_middle'].replace(0, np.nan)
        df['bb_position'] = (df['close'] - df['bb_lower']) / bb_range
        df['bb_width_change'] = df['bb_width'].pct_change()
        
        # ===== 成交量特徵 (增強版) =====
        df['volume_sma5'] = df['volume'].rolling(5).mean()
        df['volume_sma20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma20'].replace(0, 1)
        df['volume_trend'] = df['volume_sma5'] / df['volume_sma20'].replace(0, 1)
        
        # 成交量加權價格
        df['vwap_5'] = (df['close'] * df['volume']).rolling(5).sum() / df['volume'].rolling(5).sum()
        df['price_to_vwap'] = (df['close'] - df['vwap_5']) / df['vwap_5'].replace(0, np.nan)
        
        # ===== 價格距離特徵 =====
        df['hl_range'] = (df['high'] - df['low']) / df['close'].replace(0, np.nan)
        df['hl_range_ma'] = df['hl_range'].rolling(10).mean()
        df['hl_range_ratio'] = df['hl_range'] / df['hl_range_ma'].replace(0, np.nan)
        
        # ===== ATR =====
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        df['atr_pct'] = df['atr'] / df['close'].replace(0, np.nan)
        
        # ===== 時間特徵 =====
        if 'open_time' in df.columns:
            df['hour'] = pd.to_datetime(df['open_time']).dt.hour
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # 填充和清理
        df = df.ffill().bfill()
        df = df.replace([np.inf, -np.inf], np.nan)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0)
        
        return df
    
    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # 標準方向預測
        df['next_close'] = df['close'].shift(-1)
        df['next_open'] = df['open'].shift(-1)
        df['direction_label'] = (df['next_close'] > df['next_open']).astype(int)
        
        # 價格範圍預測
        df['next_high'] = df['high'].shift(-1)
        df['next_low'] = df['low'].shift(-1)
        
        df['next_high_pct'] = (df['next_high'] - df['close']) / df['close'].replace(0, np.nan) * 100
        df['next_low_pct'] = (df['next_low'] - df['close']) / df['close'].replace(0, np.nan) * 100
        
        # 限制範圍
        df['next_high_pct'] = df['next_high_pct'].clip(-5, 5)
        df['next_low_pct'] = df['next_low_pct'].clip(-5, 5)
        
        return df
    
    def get_feature_columns(self) -> list:
        return [
            # 價格動量
            'returns_1', 'returns_2', 'returns_3', 'returns_5', 'returns_10',
            'log_returns',
            
            # 訂單流
            'close_position', 'pressure_ratio', 'net_pressure_5',
            
            # K棒形態
            'body_size', 'upper_shadow', 'lower_shadow',
            'candle_direction', 'prev_candle_direction',
            'consecutive_up', 'consecutive_down',
            
            # 波動率
            'volatility_5', 'volatility_10', 'volatility_ratio',
            'parkinson_vol', 'hl_range', 'hl_range_ratio',
            
            # 動量指標
            'rsi_7', 'rsi_14', 'rsi_change', 'rsi_momentum',
            'macd_diff', 'macd_diff_change',
            'stoch_k', 'stoch_cross',
            
            # 趨勢
            'price_to_ema5', 'price_to_ema20', 'ema_alignment',
            'ema5_slope', 'ema20_slope',
            'adx', 'di_diff',
            
            # BB
            'bb_position', 'bb_width', 'bb_width_change',
            
            # 成交量
            'volume_ratio', 'volume_trend', 'price_to_vwap',
            
            # 其他
            'atr_pct'
        ]
    
    def process(self, df: pd.DataFrame, create_labels: bool = True) -> pd.DataFrame:
        df = self.add_technical_indicators(df)
        
        if create_labels:
            df = self.create_labels(df)
        
        df = df.dropna()
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        return df
    
    def get_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        feature_cols = self.get_feature_columns()
        
        X = df[feature_cols].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        y_dict = {
            'direction': df['direction_label'].copy(),
            'high_pct': df['next_high_pct'].copy(),
            'low_pct': df['next_low_pct'].copy(),
            'next_high': df['next_high'].copy(),
            'next_low': df['next_low'].copy()
        }
        
        for key in ['high_pct', 'low_pct']:
            y_dict[key] = y_dict[key].replace([np.inf, -np.inf], np.nan)
            y_dict[key] = y_dict[key].fillna(0)
        
        return X, y_dict