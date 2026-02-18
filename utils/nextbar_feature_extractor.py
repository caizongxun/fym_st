import pandas as pd
import numpy as np
import ta

class NextBarFeatureExtractor:
    """
    下一根K棒預測特徵提取器
    
    專注於預測下一根K棒的高低點
    """
    
    def __init__(self, 
                 lookback: int = 20,
                 atr_period: int = 14):
        
        self.lookback = lookback
        self.atr_period = atr_period
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # ====== 1. 歷史K棒統計特徵 ======
        df['range'] = df['high'] - df['low']
        df['range'] = df['range'].replace(0, 1e-10)  # 避免零除
        df['range_pct'] = df['range'] / df['close'].replace(0, 1e-10)
        
        for period in [5, 10, 20]:
            df[f'avg_range_{period}'] = df['range'].rolling(period).mean()
            df[f'avg_range_pct_{period}'] = df['range_pct'].rolling(period).mean()
            df[f'std_range_{period}'] = df['range'].rolling(period).std()
            
            # 避免零除
            avg_range = df[f'avg_range_{period}'].replace(0, 1e-10)
            df[f'range_ratio_{period}'] = df['range'] / avg_range
        
        # ====== 2. K棒內部結構 ======
        df['body'] = abs(df['close'] - df['open'])
        df['body_pct'] = df['body'] / df['close'].replace(0, 1e-10)
        df['upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']
        
        df['upper_shadow_pct'] = df['upper_shadow'] / df['range'].replace(0, 1e-10)
        df['lower_shadow_pct'] = df['lower_shadow'] / df['range'].replace(0, 1e-10)
        df['body_ratio'] = df['body'] / df['range'].replace(0, 1e-10)
        
        for period in [5, 10, 20]:
            df[f'avg_upper_shadow_{period}'] = df['upper_shadow'].rolling(period).mean()
            df[f'avg_lower_shadow_{period}'] = df['lower_shadow'].rolling(period).mean()
        
        # ====== 3. ATR 相關 ======
        df['atr'] = ta.volatility.average_true_range(
            df['high'], df['low'], df['close'], window=self.atr_period
        )
        df['atr_pct'] = df['atr'] / df['close'].replace(0, 1e-10)
        df['atr_ma_20'] = df['atr'].rolling(20).mean()
        df['atr_ratio'] = df['atr'] / df['atr_ma_20'].replace(0, 1e-10)
        
        df['atr_change'] = df['atr'].pct_change(1)
        df['atr_change_3'] = df['atr'].pct_change(3)
        
        # ====== 4. 價格位置特徵 ======
        for period in [10, 20, 50]:
            df[f'high_{period}'] = df['high'].rolling(period).max()
            df[f'low_{period}'] = df['low'].rolling(period).min()
            
            range_period = (df[f'high_{period}'] - df[f'low_{period}']).replace(0, 1e-10)
            df[f'price_position_{period}'] = (
                (df['close'] - df[f'low_{period}']) / range_period
            )
        
        df['dist_to_high_20'] = (df['high_20'] - df['close']) / df['close'].replace(0, 1e-10)
        df['dist_to_low_20'] = (df['close'] - df['low_20']) / df['close'].replace(0, 1e-10)
        
        # ====== 5. 成交量特徵 ======
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_20'].replace(0, 1e-10)
        df['volume_std_20'] = df['volume'].rolling(20).std()
        
        df['volume_range_ratio'] = df['volume_ratio'] * df['range_ratio_20']
        
        # ====== 6. 動量特徵 ======
        for period in [3, 5, 10]:
            df[f'momentum_{period}'] = df['close'].pct_change(period)
        
        df['momentum_acceleration'] = df['momentum_3'] - df['momentum_5']
        
        # ====== 7. 波動率狀態 ======
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2.0)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'].replace(0, 1e-10)
        
        bb_range = (df['bb_upper'] - df['bb_lower']).replace(0, 1e-10)
        df['bb_position'] = (df['close'] - df['bb_lower']) / bb_range
        
        df['bb_width_change'] = df['bb_width'].pct_change(1)
        
        # ====== 8. 指標特徵 ======
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['rsi_change'] = df['rsi'].diff(3)
        
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # ====== 9. 時間特徵 (可選) ======
        if 'open_time' in df.columns or 'time' in df.columns:
            if 'open_time' not in df.columns:
                df['open_time'] = pd.to_datetime(df['time'])
            else:
                df['open_time'] = pd.to_datetime(df['open_time'])
            
            df['hour'] = df['open_time'].dt.hour
            df['minute'] = df['open_time'].dt.minute
            df['is_whole_hour'] = (df['minute'] % 60 < 15).astype(int)
        
        # ====== 10. 近期K棒型態特徵 ======
        df['prev_range'] = df['range'].shift(1)
        df['prev_body_ratio'] = df['body_ratio'].shift(1)
        df['prev_direction'] = (df['close'].shift(1) > df['open'].shift(1)).astype(int)
        
        df['consecutive_up'] = 0
        df['consecutive_down'] = 0
        for i in range(1, 6):
            df['consecutive_up'] += (df['close'].shift(i) > df['open'].shift(i)).astype(int)
            df['consecutive_down'] += (df['close'].shift(i) < df['open'].shift(i)).astype(int)
        
        # ====== 11. 預測目標輔助特徵 ======
        df['historical_high_pct'] = (df['high'] - df['close']) / df['close'].replace(0, 1e-10)
        df['historical_low_pct'] = (df['low'] - df['close']) / df['close'].replace(0, 1e-10)
        
        for period in [5, 10, 20]:
            df[f'avg_high_pct_{period}'] = df['historical_high_pct'].rolling(period).mean()
            df[f'avg_low_pct_{period}'] = df['historical_low_pct'].rolling(period).mean()
        
        # ====== 清理異常值 ======
        # 替換 inf 和 -inf
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 填充 NaN (使用前向填充，然後零填充)
        df = df.fillna(method='ffill').fillna(0)
        
        # 限制極端值 (避免數值過大)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['open', 'high', 'low', 'close', 'volume']:
                # 限制在 -100 到 100 之間
                df[col] = df[col].clip(-100, 100)
        
        return df
    
    def get_feature_columns(self) -> list:
        """
        返回核心特徵
        """
        features = [
            # 振幅統計
            'avg_range_pct_5', 'avg_range_pct_10', 'avg_range_pct_20',
            'std_range_5', 'std_range_10', 'std_range_20',
            'range_ratio_5', 'range_ratio_10', 'range_ratio_20',
            
            # K棒內部結構
            'body_pct', 'upper_shadow_pct', 'lower_shadow_pct', 'body_ratio',
            'avg_upper_shadow_5', 'avg_upper_shadow_10', 'avg_upper_shadow_20',
            'avg_lower_shadow_5', 'avg_lower_shadow_10', 'avg_lower_shadow_20',
            
            # ATR
            'atr_pct', 'atr_ratio', 'atr_change', 'atr_change_3',
            
            # 價格位置
            'price_position_10', 'price_position_20', 'price_position_50',
            'dist_to_high_20', 'dist_to_low_20',
            
            # 成交量
            'volume_ratio', 'volume_range_ratio',
            
            # 動量
            'momentum_3', 'momentum_5', 'momentum_10', 'momentum_acceleration',
            
            # 波動率
            'bb_width', 'bb_position', 'bb_width_change',
            
            # 指標
            'rsi', 'rsi_change', 'macd_diff',
            
            # 型態
            'prev_range', 'prev_body_ratio', 'prev_direction',
            'consecutive_up', 'consecutive_down',
            
            # 歷史百分比
            'avg_high_pct_5', 'avg_high_pct_10', 'avg_high_pct_20',
            'avg_low_pct_5', 'avg_low_pct_10', 'avg_low_pct_20'
        ]
        
        return features
    
    def get_all_possible_features(self) -> list:
        base_features = self.get_feature_columns()
        time_features = ['hour', 'is_whole_hour']
        return base_features + time_features

if __name__ == '__main__':
    print("Next Bar Feature Extractor")
    print("請在App中使用")