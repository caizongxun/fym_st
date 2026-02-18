import pandas as pd
import numpy as np
import ta
from typing import Optional

class ScalpingFeatureExtractor:
    """
    15m 剝頭皮特徵提取器 (增強版)
    
    新增特徵:
    - 價格型態組合
    - 微觀結構指標
    - 動能組合特徵
    - 跟風/逆勢特徵
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
        df['ema_slope'] = df['ema_short'].diff(3) / df['close']
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        df['macd_diff_change'] = df['macd_diff'].diff()
        
        # ADX
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()
        df['adx_trend_strength'] = (df['adx_pos'] - df['adx_neg']).abs()
        
        # ====== 2. 動能特徵 ======
        df['rsi'] = ta.momentum.rsi(df['close'], window=self.rsi_period)
        df['rsi_change'] = df['rsi'].diff(3)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_neutral'] = ((df['rsi'] >= 40) & (df['rsi'] <= 60)).astype(int)
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        df['stoch_cross'] = ((df['stoch_k'] > df['stoch_d']) & 
                             (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))).astype(int)
        
        # Williams %R
        df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
        
        # ROC (Rate of Change)
        df['roc_3'] = df['close'].pct_change(3)
        df['roc_5'] = df['close'].pct_change(5)
        df['roc_10'] = df['close'].pct_change(10)
        
        # ====== 3. 波動率特徵 ======
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=self.atr_period)
        df['atr_ma'] = df['atr'].rolling(20).mean()
        df['atr_ratio'] = df['atr'] / df['atr_ma']
        df['atr_pct'] = df['atr'] / df['close']  # ATR佔價格百分比
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'], window=self.bb_period, window_dev=self.bb_std)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).mean()).astype(int)
        
        # Keltner Channel
        kc = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'])
        df['kc_upper'] = kc.keltner_channel_hband()
        df['kc_lower'] = kc.keltner_channel_lband()
        
        # ====== 4. 量價特徵 ======
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['volume_spike'] = (df['volume'] > df['volume_ma'] * 2).astype(int)
        df['volume_std'] = df['volume'].rolling(20).std()
        df['volume_cv'] = df['volume_std'] / df['volume_ma']  # 變異係數
        
        # OBV
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        df['obv_ma'] = df['obv'].rolling(20).mean()
        df['obv_slope'] = df['obv'].diff(5)
        
        # 價量背離
        price_new_high = df['close'] > df['close'].shift(1).rolling(10).max()
        volume_decreasing = df['volume'] < df['volume'].shift(1)
        df['price_volume_divergence'] = (price_new_high & volume_decreasing).astype(int)
        
        # ====== 5. 價格型態特徵 (增強) ======
        df['candle_body'] = abs(df['close'] - df['open'])
        df['candle_range'] = df['high'] - df['low']
        df['candle_upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['candle_lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']
        df['candle_body_ratio'] = df['candle_body'] / (df['candle_range'] + 1e-10)
        
        # K線顏色
        df['is_green'] = (df['close'] > df['open']).astype(int)
        df['is_red'] = (df['close'] < df['open']).astype(int)
        df['is_doji'] = (df['candle_body_ratio'] < 0.1).astype(int)
        
        # 連續K線
        df['consecutive_green'] = 0
        df['consecutive_red'] = 0
        for i in range(1, 6):
            df['consecutive_green'] += (df['close'].shift(i) > df['open'].shift(i)).astype(int)
            df['consecutive_red'] += (df['close'].shift(i) < df['open'].shift(i)).astype(int)
        
        # 型態特徵
        df['hammer'] = ((df['candle_lower_shadow'] > df['candle_body'] * 2) & 
                        (df['candle_upper_shadow'] < df['candle_body'] * 0.5)).astype(int)
        df['shooting_star'] = ((df['candle_upper_shadow'] > df['candle_body'] * 2) & 
                               (df['candle_lower_shadow'] < df['candle_body'] * 0.5)).astype(int)
        
        # ====== 6. 微觀結構特徵 (新增) ======
        # 短期價格動能
        df['price_acceleration'] = df['close'].diff(1) - df['close'].diff(2)
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        
        # Gap
        df['gap_up'] = ((df['open'] - df['close'].shift(1)) / df['close'].shift(1)).clip(lower=0)
        df['gap_down'] = ((df['close'].shift(1) - df['open']) / df['close'].shift(1)).clip(lower=0)
        
        # ====== 7. 組合特徵 (新增) ======
        # 趨勢+動能
        df['trend_momentum'] = df['ema_diff_pct'] * df['rsi'] / 100
        df['macd_rsi'] = df['macd_diff'] * df['rsi'] / 100
        
        # 波動率+量能
        df['vol_volume'] = df['atr_ratio'] * df['volume_ratio']
        
        # BB+RSI
        df['bb_rsi_combo'] = df['bb_position'] * (df['rsi'] - 50) / 50
        
        # ====== 8. 支撐阻力 ======
        df['resistance_50'] = df['high'].rolling(50).max()
        df['support_50'] = df['low'].rolling(50).min()
        df['dist_to_resistance'] = (df['resistance_50'] - df['close']) / df['close']
        df['dist_to_support'] = (df['close'] - df['support_50']) / df['close']
        
        # 短期支撐阻力
        df['resistance_10'] = df['high'].rolling(10).max()
        df['support_10'] = df['low'].rolling(10).min()
        df['near_resistance'] = (df['close'] >= df['resistance_10'] * 0.995).astype(int)
        df['near_support'] = (df['close'] <= df['support_10'] * 1.005).astype(int)
        
        # ====== 9. 時間特徵 (可選) ======
        if has_time:
            df['hour'] = df['open_time'].dt.hour
            df['minute'] = df['open_time'].dt.minute
            df['is_whole_hour'] = (df['minute'] % 60 < 15).astype(int)
            
            df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
            df['european_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
            df['american_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)
        
        return df
    
    def get_feature_columns(self) -> list:
        """
        返回核心特徵 (不含時間)
        """
        features = [
            # 趨勢
            'ema_diff_pct', 'ema_slope', 'macd_diff', 'macd_diff_change',
            'adx', 'adx_pos', 'adx_neg', 'adx_trend_strength',
            
            # 動能
            'rsi', 'rsi_change', 'rsi_overbought', 'rsi_oversold', 'rsi_neutral',
            'stoch_k', 'stoch_d', 'stoch_cross', 'williams_r',
            'roc_3', 'roc_5', 'roc_10',
            
            # 波動率
            'atr_ratio', 'atr_pct', 'bb_width', 'bb_position', 'bb_squeeze',
            
            # 量價
            'volume_ratio', 'volume_spike', 'volume_cv', 'obv_slope', 'price_volume_divergence',
            
            # 價格型態
            'candle_body_ratio', 'is_green', 'is_red', 'is_doji',
            'consecutive_green', 'consecutive_red', 'hammer', 'shooting_star',
            
            # 微觀結構
            'price_acceleration', 'high_low_ratio', 'close_position',
            'gap_up', 'gap_down',
            
            # 組合特徵
            'trend_momentum', 'macd_rsi', 'vol_volume', 'bb_rsi_combo',
            
            # 支撐阻力
            'dist_to_resistance', 'dist_to_support', 'near_resistance', 'near_support'
        ]
        
        return features
    
    def get_all_possible_features(self) -> list:
        base_features = self.get_feature_columns()
        time_features = ['hour', 'is_whole_hour', 'asian_session', 'european_session', 'american_session']
        return base_features + time_features

if __name__ == '__main__':
    print("Scalping Feature Extractor (Enhanced)")
    print("請在App中使用")