import pandas as pd
import numpy as np
import ta
from typing import Tuple, Dict

class DualModelFeatureExtractor:
    """
    雙模型特徵提取器 - 用於剩頭皮交易
    
    模型A: 預測下一根K棒漲跌 (direction_pred)
    模型B: 預測下一根K棒最高/最低價 (high_pred, low_pred)
    
    特點:
    - 只使用已完全形成的K棒 (當下K棒不參與訓練)
    - 多維度技術指標 + 價量特徵
    - 考慮波動率、動量、趨勢等
    """
    
    def __init__(self, lookback_candles: int = 10):
        """
        Args:
            lookback_candles: 回望期數 (使用多少根歷史K棒預測)
        """
        self.lookback_candles = lookback_candles
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加技術指標"""
        df = df.copy()
        
        # ===== 價格特徵 =====
        # 報酬率
        df['returns'] = df['close'].pct_change()
        df['returns_1'] = df['returns'].shift(1)
        df['returns_2'] = df['returns'].shift(2)
        
        # 波動率
        df['volatility_5'] = df['returns'].rolling(5).std()
        df['volatility_10'] = df['returns'].rolling(10).std()
        
        # 高低價範圍
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        df['hl_range_ma5'] = df['hl_range'].rolling(5).mean()
        
        # K棒實體大小
        df['body_size'] = abs(df['close'] - df['open']) / df['open']
        df['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['open']
        df['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['open']
        
        # ===== 成交量特徵 =====
        df['volume_ma5'] = df['volume'].rolling(5).mean()
        df['volume_ma10'] = df['volume'].rolling(10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma10']
        df['volume_change'] = df['volume'].pct_change()
        
        # ===== 動量指標 =====
        # RSI
        df['rsi_7'] = ta.momentum.RSIIndicator(df['close'], window=7).rsi()
        df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # ===== 趨勢指標 =====
        # 移動平均
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        
        # 價格相對於MA的位置
        df['price_to_sma5'] = (df['close'] - df['sma_5']) / df['sma_5']
        df['price_to_sma10'] = (df['close'] - df['sma_10']) / df['sma_10']
        
        # MA斜率
        df['sma5_slope'] = df['sma_5'].pct_change()
        df['sma10_slope'] = df['sma_10'].pct_change()
        
        # ADX
        adx_indicator = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx_indicator.adx()
        df['adx_pos'] = adx_indicator.adx_pos()
        df['adx_neg'] = adx_indicator.adx_neg()
        
        # ===== Bollinger Bands =====
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ===== ATR =====
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        df['atr_pct'] = df['atr'] / df['close']
        
        # 填充缺失值
        df = df.ffill().bfill()
        
        return df
    
    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        創建Label
        - direction_label: 下一根K棒漲/跌 (1=漲, 0=跌)
        - next_high: 下一根K棒最高價
        - next_low: 下一根K棒最低價
        - next_high_pct: 下一根最高價相對於當前close的報酬率
        - next_low_pct: 下一根最低價相對於當前close的報酬率
        """
        df = df.copy()
        
        # 模型A: 漲跌預測
        df['next_close'] = df['close'].shift(-1)
        df['next_open'] = df['open'].shift(-1)
        df['direction_label'] = (df['next_close'] > df['next_open']).astype(int)
        
        # 模型B: 最高/最低價預測
        df['next_high'] = df['high'].shift(-1)
        df['next_low'] = df['low'].shift(-1)
        
        # 轉換為相對報酬率 (更容易學習)
        df['next_high_pct'] = (df['next_high'] - df['close']) / df['close'] * 100
        df['next_low_pct'] = (df['next_low'] - df['close']) / df['close'] * 100
        
        return df
    
    def get_feature_columns(self) -> list:
        """獲取特徵列名稱"""
        return [
            # 價格特徵
            'returns', 'returns_1', 'returns_2',
            'volatility_5', 'volatility_10',
            'hl_range', 'hl_range_ma5',
            'body_size', 'upper_shadow', 'lower_shadow',
            
            # 成交量特徵
            'volume_ma5', 'volume_ma10', 'volume_ratio', 'volume_change',
            
            # 動量指標
            'rsi_7', 'rsi_14',
            'macd', 'macd_signal', 'macd_diff',
            'stoch_k', 'stoch_d',
            
            # 趨勢指標
            'price_to_sma5', 'price_to_sma10',
            'sma5_slope', 'sma10_slope',
            'adx', 'adx_pos', 'adx_neg',
            
            # BB指標
            'bb_width', 'bb_position',
            
            # 波動率
            'atr_pct'
        ]
    
    def process(self, df: pd.DataFrame, create_labels: bool = True) -> pd.DataFrame:
        """
        處理完整流程
        
        Args:
            df: 原始K棒數據
            create_labels: 是否創建Label (訓練時True, 預測時False)
        
        Returns:
            處理後的DataFrame
        """
        # 添加技術指標
        df = self.add_technical_indicators(df)
        
        # 創建Label
        if create_labels:
            df = self.create_labels(df)
        
        # 移除缺失值行
        df = df.dropna()
        
        return df
    
    def get_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """
        獲取訓練數據
        
        Returns:
            X: 特徵矩陣
            y_dict: Label字典 {'direction': Series, 'high_pct': Series, 'low_pct': Series}
        """
        feature_cols = self.get_feature_columns()
        
        X = df[feature_cols].copy()
        
        y_dict = {
            'direction': df['direction_label'].copy(),
            'high_pct': df['next_high_pct'].copy(),
            'low_pct': df['next_low_pct'].copy(),
            'next_high': df['next_high'].copy(),
            'next_low': df['next_low'].copy()
        }
        
        return X, y_dict
    
    def calculate_profit_potential(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算每筆交易的潛在獲利空間"""
        df = df.copy()
        
        # 使用下一根open作為進場價
        df['entry_price'] = df['open'].shift(-1)
        
        # 做多潛在利潤
        df['long_potential'] = (df['next_high'] - df['entry_price']) / df['entry_price'] * 100
        df['long_risk'] = (df['entry_price'] - df['next_low']) / df['entry_price'] * 100
        df['long_rr'] = df['long_potential'] / df['long_risk'].replace(0, 0.01)
        
        # 做空潛在利潤
        df['short_potential'] = (df['entry_price'] - df['next_low']) / df['entry_price'] * 100
        df['short_risk'] = (df['next_high'] - df['entry_price']) / df['entry_price'] * 100
        df['short_rr'] = df['short_potential'] / df['short_risk'].replace(0, 0.01)
        
        return df


if __name__ == '__main__':
    print("雙模型特徵提取器測試")
    print("="*60)
    
    # 生成測試數據
    dates = pd.date_range('2024-01-01', periods=1000, freq='15min')
    np.random.seed(42)
    
    base_price = 50000
    prices = base_price + np.random.randn(1000).cumsum() * 100
    
    df = pd.DataFrame({
        'open_time': dates,
        'open': prices,
        'high': prices + np.random.rand(1000) * 50,
        'low': prices - np.random.rand(1000) * 50,
        'close': prices + np.random.randn(1000) * 20,
        'volume': np.random.randint(1000, 5000, 1000)
    })
    
    # 處理數據
    extractor = DualModelFeatureExtractor(lookback_candles=10)
    df_processed = extractor.process(df, create_labels=True)
    
    print(f"原始數據: {len(df)} 根K棒")
    print(f"處理後: {len(df_processed)} 根K棒")
    print(f"特徵數量: {len(extractor.get_feature_columns())}")
    
    # 獲取訓練數據
    X, y_dict = extractor.get_training_data(df_processed)
    
    print(f"\n特徵矩陣 X: {X.shape}")
    print(f"漲跌Label: {y_dict['direction'].value_counts().to_dict()}")
    print(f"最高價範圍: {y_dict['high_pct'].min():.3f}% ~ {y_dict['high_pct'].max():.3f}%")
    print(f"最低價範圏: {y_dict['low_pct'].min():.3f}% ~ {y_dict['low_pct'].max():.3f}%")
    
    # 計算獲利潛力
    df_profit = extractor.calculate_profit_potential(df_processed)
    print(f"\n平均做多潛力: {df_profit['long_potential'].mean():.3f}%")
    print(f"平均做多風險: {df_profit['long_risk'].mean():.3f}%")
    print(f"平均風報比: {df_profit['long_rr'].mean():.2f}")