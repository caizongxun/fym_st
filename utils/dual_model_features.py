import pandas as pd
import numpy as np
import ta
from typing import Tuple, Dict

class DualModelFeatureExtractor:
    """
    雙模型特徵提取器 - 用於剩頭皮交易
    
    模型A: 預測下一根K棒漨跌 (direction_pred)
    模型B: 預測下一根K棒最高/最低價 (high_pred, low_pred)
    
    v2: 修復無限值和NaN問題
    """
    
    def __init__(self, lookback_candles: int = 10):
        self.lookback_candles = lookback_candles
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # ===== 價格特徵 =====
        df['returns'] = df['close'].pct_change()
        df['returns_1'] = df['returns'].shift(1)
        df['returns_2'] = df['returns'].shift(2)
        
        df['volatility_5'] = df['returns'].rolling(5).std()
        df['volatility_10'] = df['returns'].rolling(10).std()
        
        df['hl_range'] = (df['high'] - df['low']) / df['close'].replace(0, np.nan)
        df['hl_range_ma5'] = df['hl_range'].rolling(5).mean()
        
        df['body_size'] = abs(df['close'] - df['open']) / df['open'].replace(0, np.nan)
        df['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['open'].replace(0, np.nan)
        df['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['open'].replace(0, np.nan)
        
        # ===== 成交量特徵 =====
        df['volume_ma5'] = df['volume'].rolling(5).mean()
        df['volume_ma10'] = df['volume'].rolling(10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma10'].replace(0, 1)
        df['volume_change'] = df['volume'].pct_change()
        
        # ===== 動量指標 =====
        df['rsi_7'] = ta.momentum.RSIIndicator(df['close'], window=7).rsi()
        df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # ===== 趨勢指標 =====
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        
        df['price_to_sma5'] = (df['close'] - df['sma_5']) / df['sma_5'].replace(0, np.nan)
        df['price_to_sma10'] = (df['close'] - df['sma_10']) / df['sma_10'].replace(0, np.nan)
        
        df['sma5_slope'] = df['sma_5'].pct_change()
        df['sma10_slope'] = df['sma_10'].pct_change()
        
        adx_indicator = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx_indicator.adx()
        df['adx_pos'] = adx_indicator.adx_pos()
        df['adx_neg'] = adx_indicator.adx_neg()
        
        # ===== Bollinger Bands =====
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'].replace(0, np.nan)
        
        bb_range = (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)
        df['bb_position'] = (df['close'] - df['bb_lower']) / bb_range
        
        # ===== ATR =====
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        df['atr_pct'] = df['atr'] / df['close'].replace(0, np.nan)
        
        # 填充缺失值
        df = df.ffill().bfill()
        
        # 關鍵: 替換無限值和NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 對每個數值列再次填充
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                # 用中位數填充剩餘的NaN
                median_val = df[col].median()
                if pd.isna(median_val):
                    df[col] = 0
                else:
                    df[col] = df[col].fillna(median_val)
        
        return df
    
    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # 模型A: 漨跌預測
        df['next_close'] = df['close'].shift(-1)
        df['next_open'] = df['open'].shift(-1)
        df['direction_label'] = (df['next_close'] > df['next_open']).astype(int)
        
        # 模型B: 最高/最低價預測
        df['next_high'] = df['high'].shift(-1)
        df['next_low'] = df['low'].shift(-1)
        
        # 轉換為相對報酬率
        df['next_high_pct'] = (df['next_high'] - df['close']) / df['close'].replace(0, np.nan) * 100
        df['next_low_pct'] = (df['next_low'] - df['close']) / df['close'].replace(0, np.nan) * 100
        
        # 限制異常值 (避免極端值)
        df['next_high_pct'] = df['next_high_pct'].clip(-10, 10)
        df['next_low_pct'] = df['next_low_pct'].clip(-10, 10)
        
        return df
    
    def get_feature_columns(self) -> list:
        return [
            'returns', 'returns_1', 'returns_2',
            'volatility_5', 'volatility_10',
            'hl_range', 'hl_range_ma5',
            'body_size', 'upper_shadow', 'lower_shadow',
            'volume_ma5', 'volume_ma10', 'volume_ratio', 'volume_change',
            'rsi_7', 'rsi_14',
            'macd', 'macd_signal', 'macd_diff',
            'stoch_k', 'stoch_d',
            'price_to_sma5', 'price_to_sma10',
            'sma5_slope', 'sma10_slope',
            'adx', 'adx_pos', 'adx_neg',
            'bb_width', 'bb_position',
            'atr_pct'
        ]
    
    def process(self, df: pd.DataFrame, create_labels: bool = True) -> pd.DataFrame:
        # 添加技術指標
        df = self.add_technical_indicators(df)
        
        # 創建Label
        if create_labels:
            df = self.create_labels(df)
        
        # 移除缺失值行
        df = df.dropna()
        
        # 最後檢查: 確保沒有無限值
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        return df
    
    def get_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        feature_cols = self.get_feature_columns()
        
        X = df[feature_cols].copy()
        
        # 再次確認沒有無限值
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)  # 用0填充任何剩餘的NaN
        
        y_dict = {
            'direction': df['direction_label'].copy(),
            'high_pct': df['next_high_pct'].copy(),
            'low_pct': df['next_low_pct'].copy(),
            'next_high': df['next_high'].copy(),
            'next_low': df['next_low'].copy()
        }
        
        # 確保y也沒有無限值
        for key in ['high_pct', 'low_pct']:
            y_dict[key] = y_dict[key].replace([np.inf, -np.inf], np.nan)
            y_dict[key] = y_dict[key].fillna(0)
        
        return X, y_dict
    
    def calculate_profit_potential(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['entry_price'] = df['open'].shift(-1)
        
        df['long_potential'] = (df['next_high'] - df['entry_price']) / df['entry_price'].replace(0, np.nan) * 100
        df['long_risk'] = (df['entry_price'] - df['next_low']) / df['entry_price'].replace(0, np.nan) * 100
        df['long_rr'] = df['long_potential'] / df['long_risk'].replace(0, 0.01)
        
        df['short_potential'] = (df['entry_price'] - df['next_low']) / df['entry_price'].replace(0, np.nan) * 100
        df['short_risk'] = (df['next_high'] - df['entry_price']) / df['entry_price'].replace(0, np.nan) * 100
        df['short_rr'] = df['short_potential'] / df['short_risk'].replace(0, 0.01)
        
        # 清理無限值
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        return df


if __name__ == '__main__':
    print("雙模型特徵提取器測試 (v2 - 修復版)")
    print("="*60)
    
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
    
    extractor = DualModelFeatureExtractor(lookback_candles=10)
    df_processed = extractor.process(df, create_labels=True)
    
    print(f"原始數據: {len(df)} 根K棒")
    print(f"處理後: {len(df_processed)} 根K棒")
    print(f"特徵數量: {len(extractor.get_feature_columns())}")
    
    X, y_dict = extractor.get_training_data(df_processed)
    
    # 檢查無限值
    print(f"\nX中的inf: {np.isinf(X).sum().sum()}")
    print(f"X中的NaN: {X.isna().sum().sum()}")
    print(f"\n特徵矩陣 X: {X.shape}")
    print(f"漨跌Label: {y_dict['direction'].value_counts().to_dict()}")
    print(f"最高價範圍: {y_dict['high_pct'].min():.3f}% ~ {y_dict['high_pct'].max():.3f}%")
    print(f"最低價範圍: {y_dict['low_pct'].min():.3f}% ~ {y_dict['low_pct'].max():.3f}%")