import pandas as pd
import numpy as np
import ta
from typing import Optional

class ScalpingFeatureExtractor:
    """
    15m 剝頭皮特徵提取器
    
    融合多維度技術指標:
    - 趨勢: EMA, MACD, ADX
    - 動能: RSI, Stochastic, Williams %R
    - 波動率: ATR, BB, Keltner
    - 量價: Volume, OBV
    - 市場結構: Support/Resistance, K線形態
    - 時間: Hour, Minute (可選)
    """
    
    def __init__(self,
                 ema_short: int = 9,
                 ema_long: int = 21,
                 rsi_period: int = 14,
                 atr_period: int = 14,
                 bb_period: int = 20,
                 bb_std: float = 2.0):
        
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.bb_period = bb_period
        self.bb_std = bb_std
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取所有特徵
        """
        df = df.copy()
        
        # 確保時間欄位存在
        has_time = False
        if 'open_time' not in df.columns:
            if 'time' in df.columns:
                df['open_time'] = pd.to_datetime(df['time'])
                has_time = True
        else:
            df['open_time'] = pd.to_datetime(df['open_time'])
            has_time = True
        
        # ====== 1. 趨勢特徵 ======
        df['ema_short'] = ta.trend.ema_indicator(df['close'], window=self.ema_short)
        df['ema_long'] = ta.trend.ema_indicator(df['close'], window=self.ema_long)
        df['ema_diff'] = df['ema_short'] - df['ema_long']
        df['ema_diff_pct'] = df['ema_diff'] / df['close']
        df['ema_slope'] = df['ema_short'].diff(3) / df['close'] # 3根K線斜率
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        df['macd_diff_change'] = df['macd_diff'].diff()
        
        # ADX (趨勢強度)
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()
        
        # ====== 2. 動能特徵 ======
        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=self.rsi_period)
        df['rsi_change'] = df['rsi'].diff(3)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        df['stoch_cross'] = ((df['stoch_k'] > df['stoch_d']) & 
                             (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))).astype(int)
        
        # Williams %R
        df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
        
        # ====== 3. 波動率特徵 ======
        # ATR
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=self.atr_period)
        df['atr_ma'] = df['atr'].rolling(20).mean()
        df['atr_ratio'] = df['atr'] / df['atr_ma']
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'], window=self.bb_period, window_dev=self.bb_std)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Keltner Channel
        kc = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'])
        df['kc_upper'] = kc.keltner_channel_hband()
        df['kc_lower'] = kc.keltner_channel_lband()
        
        # ====== 4. 量價特徵 ======
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['volume_spike'] = (df['volume'] > df['volume_ma'] * 2).astype(int)
        
        # OBV (On-Balance Volume)
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        df['obv_ma'] = df['obv'].rolling(20).mean()
        df['obv_slope'] = df['obv'].diff(5)
        
        # 價量背離
        price_new_high = df['close'] > df['close'].shift(1).rolling(10).max()
        volume_decreasing = df['volume'] < df['volume'].shift(1)
        df['price_volume_divergence'] = (price_new_high & volume_decreasing).astype(int)
        
        # ====== 5. 市場結構特徵 ======
        # K線形態
        df['candle_body'] = abs(df['close'] - df['open'])
        df['candle_upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['candle_lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']
        df['candle_body_ratio'] = df['candle_body'] / (df['high'] - df['low'] + 1e-10)
        
        # K線顏色
        df['is_green'] = (df['close'] > df['open']).astype(int)
        df['is_red'] = (df['close'] < df['open']).astype(int)
        
        # 連續K線
        df['consecutive_green'] = 0
        df['consecutive_red'] = 0
        for i in range(1, 6):
            df['consecutive_green'] += (df['close'].shift(i) > df['open'].shift(i)).astype(int)
            df['consecutive_red'] += (df['close'].shift(i) < df['open'].shift(i)).astype(int)
        
        # 支撐阻力 (簡化版: 近50根的最高/最低)
        df['resistance_50'] = df['high'].rolling(50).max()
        df['support_50'] = df['low'].rolling(50).min()
        df['dist_to_resistance'] = (df['resistance_50'] - df['close']) / df['close']
        df['dist_to_support'] = (df['close'] - df['support_50']) / df['close']
        
        # ====== 6. 時間特徵 (可選) ======
        if has_time:
            df['hour'] = df['open_time'].dt.hour
            df['minute'] = df['open_time'].dt.minute
            df['is_whole_hour'] = (df['minute'] % 60 < 15).astype(int) # 整點前15分鐘
            
            # 交易時段 (UTC 時間)
            df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
            df['european_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
            df['american_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)
        
        return df
    
    def get_feature_columns(self) -> list:
        """
        返回特徵列名稱 (用於模型輸入)
        """
        features = [
            # 趨勢
            'ema_diff_pct', 'ema_slope', 'macd_diff', 'macd_diff_change',
            'adx', 'adx_pos', 'adx_neg',
            
            # 動能
            'rsi', 'rsi_change', 'rsi_overbought', 'rsi_oversold',
            'stoch_k', 'stoch_d', 'stoch_cross', 'williams_r',
            
            # 波動率
            'atr_ratio', 'bb_width', 'bb_position',
            
            # 量價
            'volume_ratio', 'volume_spike', 'obv_slope', 'price_volume_divergence',
            
            # 市場結構
            'candle_body_ratio', 'is_green', 'is_red',
            'consecutive_green', 'consecutive_red',
            'dist_to_resistance', 'dist_to_support'
        ]
        
        # 時間特徵 (如果存在)
        # 這些會在訓練時動態檢查
        return features
    
    def get_all_possible_features(self) -> list:
        """
        返回所有可能的特徵 (包含時間特徵)
        """
        base_features = self.get_feature_columns()
        time_features = ['hour', 'is_whole_hour', 'asian_session', 'european_session', 'american_session']
        return base_features + time_features

if __name__ == '__main__':
    print("Scalping Feature Extractor")
    print("請在App中使用")