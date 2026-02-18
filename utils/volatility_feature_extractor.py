import pandas as pd
import numpy as np
import ta

class VolatilityFeatureExtractor:
    """
    波動率預測特徵提取器
    
    專注於波動率相關特徵:
    - 波動率壓縮/釋放
    - 成交量異常
    - 價格靖默期
    - 波動率轉換訊號
    """
    
    def __init__(self,
                 atr_period: int = 14,
                 bb_period: int = 20,
                 volume_period: int = 20):
        
        self.atr_period = atr_period
        self.bb_period = bb_period
        self.volume_period = volume_period
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # ====== 1. ATR 相關特徵 ======
        df['atr'] = ta.volatility.average_true_range(
            df['high'], df['low'], df['close'], window=self.atr_period
        )
        df['atr_ma'] = df['atr'].rolling(20).mean()
        df['atr_ratio'] = df['atr'] / df['atr_ma']
        df['atr_pct'] = df['atr'] / df['close']
        
        # ATR 變化率 (加速度)
        df['atr_change'] = df['atr'].pct_change(1)
        df['atr_acceleration'] = df['atr_change'].diff(1)
        
        # ATR 越勢
        df['atr_trend'] = (df['atr'] > df['atr'].shift(1)).astype(int)
        df['atr_expanding'] = (
            (df['atr'] > df['atr'].shift(1)) & 
            (df['atr'].shift(1) > df['atr'].shift(2))
        ).astype(int)
        
        # ====== 2. Bollinger Bands 壓縮 ======
        bb = ta.volatility.BollingerBands(
            df['close'], window=self.bb_period, window_dev=2.0
        )
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # BB 壓縮指標
        df['bb_width_ma'] = df['bb_width'].rolling(20).mean()
        df['bb_squeeze'] = (df['bb_width'] < df['bb_width_ma'] * 0.8).astype(int)
        df['bb_squeeze_intensity'] = df['bb_width'] / df['bb_width_ma']
        
        # 連續壓縮時間
        df['consecutive_squeeze'] = 0
        squeeze_count = 0
        for i in range(len(df)):
            if df.iloc[i]['bb_squeeze'] == 1:
                squeeze_count += 1
            else:
                squeeze_count = 0
            df.iloc[i, df.columns.get_loc('consecutive_squeeze')] = squeeze_count
        
        # ====== 3. 價格範圍特徵 ======
        df['candle_range'] = df['high'] - df['low']
        df['candle_range_pct'] = df['candle_range'] / df['close']
        df['candle_range_ma'] = df['candle_range'].rolling(20).mean()
        df['candle_range_ratio'] = df['candle_range'] / df['candle_range_ma']
        
        # 連續小振幅K線 (預示壓縮)
        df['small_range'] = (df['candle_range_ratio'] < 0.7).astype(int)
        df['consecutive_small_range'] = 0
        small_count = 0
        for i in range(len(df)):
            if df.iloc[i]['small_range'] == 1:
                small_count += 1
            else:
                small_count = 0
            df.iloc[i, df.columns.get_loc('consecutive_small_range')] = small_count
        
        # ====== 4. 成交量特徵 ======
        df['volume_ma'] = df['volume'].rolling(self.volume_period).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['volume_std'] = df['volume'].rolling(self.volume_period).std()
        df['volume_cv'] = df['volume_std'] / df['volume_ma']  # 變異係數
        
        # 成交量突增
        df['volume_spike'] = (df['volume'] > df['volume_ma'] * 2).astype(int)
        df['volume_dry_up'] = (df['volume'] < df['volume_ma'] * 0.5).astype(int)
        
        # 成交量趋勢
        df['volume_trend'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
        
        # ====== 5. 價格靖默期 ======
        # 計算連續低波動的時間
        df['low_volatility'] = (df['candle_range_pct'] < df['atr_pct'] * 0.5).astype(int)
        df['quiet_period'] = 0
        quiet_count = 0
        for i in range(len(df)):
            if df.iloc[i]['low_volatility'] == 1:
                quiet_count += 1
            else:
                quiet_count = 0
            df.iloc[i, df.columns.get_loc('quiet_period')] = quiet_count
        
        # ====== 6. 動能指標 ======
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['rsi_change'] = df['rsi'].diff(3)
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # ====== 7. 跟風/逆勢特徵 ======
        df['momentum_3'] = df['close'].pct_change(3)
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_10'] = df['close'].pct_change(10)
        
        # 跟風強度
        df['trend_strength'] = abs(df['momentum_10'])
        
        # ====== 8. 組合特徵 ======
        # BB壓縮 + 低成交量 = 波動率即將釋放
        df['squeeze_with_low_volume'] = (
            (df['bb_squeeze'] == 1) & 
            (df['volume_dry_up'] == 1)
        ).astype(int)
        
        # ATR上升 + 成交量增加
        df['vol_expansion'] = (
            (df['atr_expanding'] == 1) & 
            (df['volume_spike'] == 1)
        ).astype(int)
        
        # 長期靖默 + 小振幅
        df['extended_quiet'] = (
            (df['quiet_period'] >= 5) & 
            (df['consecutive_small_range'] >= 3)
        ).astype(int)
        
        # ====== 9. 時間特徵 (可選) ======
        if 'open_time' in df.columns or 'time' in df.columns:
            if 'open_time' not in df.columns:
                df['open_time'] = pd.to_datetime(df['time'])
            else:
                df['open_time'] = pd.to_datetime(df['open_time'])
            
            df['hour'] = df['open_time'].dt.hour
            df['is_asian'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
            df['is_european'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
            df['is_american'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)
        
        return df
    
    def get_feature_columns(self) -> list:
        """
        返回核心特徵
        """
        features = [
            # ATR
            'atr_ratio', 'atr_pct', 'atr_change', 'atr_acceleration',
            'atr_trend', 'atr_expanding',
            
            # BB
            'bb_width', 'bb_squeeze', 'bb_squeeze_intensity', 'consecutive_squeeze',
            
            # 價格範圍
            'candle_range_pct', 'candle_range_ratio', 'small_range', 'consecutive_small_range',
            
            # 成交量
            'volume_ratio', 'volume_cv', 'volume_spike', 'volume_dry_up', 'volume_trend',
            
            # 靖默期
            'low_volatility', 'quiet_period',
            
            # 動能
            'rsi', 'rsi_change', 'stoch_k', 'stoch_d',
            
            # 動量
            'momentum_3', 'momentum_5', 'momentum_10', 'trend_strength',
            
            # 組合
            'squeeze_with_low_volume', 'vol_expansion', 'extended_quiet'
        ]
        
        return features
    
    def get_all_possible_features(self) -> list:
        base_features = self.get_feature_columns()
        time_features = ['hour', 'is_asian', 'is_european', 'is_american']
        return base_features + time_features

if __name__ == '__main__':
    print("Volatility Feature Extractor")
    print("請在App中使用")