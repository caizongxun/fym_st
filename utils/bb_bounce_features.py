import pandas as pd
import numpy as np
from typing import Tuple, Dict

class BBBounceFeatureExtractor:
    """
    布林通道反彈特徵提取器
    
    核心功能:
    1. 檢測BB上軌/下軌觸碰
    2. 計算ADX和DI指標判斷趨勢強度
    3. 提取30根K線的反彈預測特徵
    4. 區分「強趨勢回調」vs「真反轉」
    """
    
    def __init__(self, 
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 adx_period: int = 14,
                 touch_threshold: float = 0.3):
        """
        Args:
            bb_period: BB週期
            bb_std: BB標準差倍數
            adx_period: ADX計算週期
            touch_threshold: 觸碰閾值(標準差倍數,0.3=距離軌道0.3σ內)
        """
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.adx_period = adx_period
        self.touch_threshold = touch_threshold
    
    def calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        計算布林通道
        """
        df = df.copy()
        
        # 中軌 = SMA
        df['bb_middle'] = df['close'].rolling(window=self.bb_period).mean()
        
        # 標準差
        df['bb_std'] = df['close'].rolling(window=self.bb_period).std()
        
        # 上軌和下軌
        df['bb_upper'] = df['bb_middle'] + (self.bb_std * df['bb_std'])
        df['bb_lower'] = df['bb_middle'] - (self.bb_std * df['bb_std'])
        
        # BB帶寬(波動度指標)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100
        
        # 價格在BB中的位置 (0=下軌, 0.5=中軌, 1=上軌)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    def calculate_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        計算ADX和方向指標(+DI, -DI)
        用於判斷趨勢強度
        """
        df = df.copy()
        
        # True Range
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = abs(df['high'] - df['close'].shift(1))
        df['low_close'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        
        # +DM和-DM
        df['up_move'] = df['high'] - df['high'].shift(1)
        df['down_move'] = df['low'].shift(1) - df['low']
        
        df['plus_dm'] = np.where(
            (df['up_move'] > df['down_move']) & (df['up_move'] > 0),
            df['up_move'],
            0
        )
        
        df['minus_dm'] = np.where(
            (df['down_move'] > df['up_move']) & (df['down_move'] > 0),
            df['down_move'],
            0
        )
        
        # 平滑處理
        period = self.adx_period
        df['atr'] = df['tr'].rolling(window=period).mean()
        df['plus_di'] = 100 * (df['plus_dm'].rolling(window=period).mean() / df['atr'])
        df['minus_di'] = 100 * (df['minus_dm'].rolling(window=period).mean() / df['atr'])
        
        # DX和ADX
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        df['adx'] = df['dx'].rolling(window=period).mean()
        
        # 清理臨時列
        df.drop(['high_low', 'high_close', 'low_close', 'tr', 'up_move', 'down_move',
                 'plus_dm', 'minus_dm', 'dx'], axis=1, inplace=True, errors='ignore')
        
        return df
    
    def detect_bb_touches(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        檢測觸碰BB上軌/下軌的點
        """
        df = df.copy()
        
        # 計算到上軌和下軌的距離(以標準差為單位)
        df['dist_to_upper'] = (df['bb_upper'] - df['close']) / df['bb_std']
        df['dist_to_lower'] = (df['close'] - df['bb_lower']) / df['bb_std']
        
        # 觸碰上軌: 距離 < threshold
        df['touch_upper'] = (df['dist_to_upper'] < self.touch_threshold).astype(int)
        
        # 觸碰下軌: 距離 < threshold  
        df['touch_lower'] = (df['dist_to_lower'] < self.touch_threshold).astype(int)
        
        # 穿越檢測(更強的信號)
        df['pierce_upper'] = ((df['high'] >= df['bb_upper']) & 
                              (df['close'] < df['bb_upper'])).astype(int)
        df['pierce_lower'] = ((df['low'] <= df['bb_lower']) & 
                              (df['close'] > df['bb_lower'])).astype(int)
        
        return df
    
    def classify_trend_state(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        分類趨勢狀態
        用於過濾強趨勢中的假信號
        """
        df = df.copy()
        
        # 趨勢強度分類
        conditions = [
            # 強上升趨勢: ADX>30, +DI > -DI by significant margin
            (df['adx'] > 30) & (df['plus_di'] > df['minus_di'] + 10),
            
            # 弱上升趨勢: ADX<30, +DI > -DI
            (df['adx'] <= 30) & (df['plus_di'] > df['minus_di']),
            
            # 盤整: ADX < 20
            df['adx'] < 20,
            
            # 弱下降趨勢: ADX<30, -DI > +DI
            (df['adx'] <= 30) & (df['minus_di'] > df['plus_di']),
            
            # 強下降趨勢: ADX>30, -DI > +DI by significant margin
            (df['adx'] > 30) & (df['minus_di'] > df['plus_di'] + 10),
        ]
        
        choices = ['strong_uptrend', 'weak_uptrend', 'ranging', 'weak_downtrend', 'strong_downtrend']
        
        df['trend_state'] = np.select(conditions, choices, default='ranging')
        
        # 數值編碼(用於模型訓練)
        trend_map = {
            'strong_downtrend': -2,
            'weak_downtrend': -1,
            'ranging': 0,
            'weak_uptrend': 1,
            'strong_uptrend': 2
        }
        df['trend_state_encoded'] = df['trend_state'].map(trend_map)
        
        return df
    
    def extract_bounce_features(self, df: pd.DataFrame, lookback: int = 30) -> pd.DataFrame:
        """
        提取用於反彈預測的特徵
        包含30根K線的滾動特徵
        """
        df = df.copy()
        
        # === BB相關特徵 ===
        df['bb_width_change'] = df['bb_width'].pct_change(self.bb_period)
        df['bb_width_rank'] = df['bb_width'].rolling(window=lookback).rank(pct=True)
        
        # === 動量特徵 ===
        for period in [3, 5, 10, 20]:
            df[f'return_{period}'] = df['close'].pct_change(period) * 100
            df[f'vol_ratio_{period}'] = df['volume'] / df['volume'].rolling(period).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # === 趨勢特徵 ===
        for period in [10, 20, 50]:
            df[f'slope_{period}'] = (df['close'] - df['close'].shift(period)) / period
        
        # 連續同向K線
        df['up_candles'] = (df['close'] > df['open']).astype(int)
        df['consecutive_up'] = df['up_candles'].groupby(
            (df['up_candles'] != df['up_candles'].shift()).cumsum()
        ).cumsum()
        
        df['down_candles'] = (df['close'] < df['open']).astype(int)
        df['consecutive_down'] = df['down_candles'].groupby(
            (df['down_candles'] != df['down_candles'].shift()).cumsum()
        ).cumsum()
        
        # === K線形態 ===
        df['body_size'] = abs(df['close'] - df['open']) / df['open'] * 100
        df['upper_wick'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['open'] * 100
        df['lower_wick'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['open'] * 100
        
        # === 歷史反彈統計 ===
        # 簡化版:過去N次觸碰的平均反彈幅度
        df['upper_touch_count'] = df['touch_upper'].rolling(window=100).sum()
        df['lower_touch_count'] = df['touch_lower'].rolling(window=100).sum()
        
        return df
    
    def create_labels(self, df: pd.DataFrame, forward_bars: int = 3, 
                     bounce_threshold: float = 0.5) -> pd.DataFrame:
        """
        創建訓練標籤
        
        觸碰上軌:
            Label=1: 未來forward_bars內反彈至少bounce_threshold%
            Label=0: 沒有明顯反彈或繼續上漲
            
        觸碰下軌:
            Label=1: 未來forward_bars內反彈至少bounce_threshold%
            Label=0: 沒有明顯反彈或繼續下跌
        """
        df = df.copy()
        
        # 初始化標籤
        df['upper_bounce_label'] = 0
        df['lower_bounce_label'] = 0
        
        # 計算未來最高/最低
        df['future_high'] = df['high'].rolling(window=forward_bars).max().shift(-forward_bars)
        df['future_low'] = df['low'].rolling(window=forward_bars).min().shift(-forward_bars)
        
        # 上軌反彈標籤(做空機會)
        upper_touches = df['touch_upper'] == 1
        bounce_down = (df['close'] - df['future_low']) / df['close'] * 100 >= bounce_threshold
        df.loc[upper_touches & bounce_down, 'upper_bounce_label'] = 1
        
        # 下軌反彈標籤(做多機會)
        lower_touches = df['touch_lower'] == 1
        bounce_up = (df['future_high'] - df['close']) / df['close'] * 100 >= bounce_threshold
        df.loc[lower_touches & bounce_up, 'lower_bounce_label'] = 1
        
        # 清理臨時列
        df.drop(['future_high', 'future_low'], axis=1, inplace=True, errors='ignore')
        
        return df
    
    def process(self, df: pd.DataFrame, create_labels: bool = False) -> pd.DataFrame:
        """
        完整處理流程
        """
        df = df.copy()
        
        # 1. 計算BB
        df = self.calculate_bollinger_bands(df)
        
        # 2. 計算ADX
        df = self.calculate_adx(df)
        
        # 3. 檢測觸碰
        df = self.detect_bb_touches(df)
        
        # 4. 分類趨勢狀態
        df = self.classify_trend_state(df)
        
        # 5. 提取特徵
        df = self.extract_bounce_features(df)
        
        # 6. 創建標籤(訓練時)
        if create_labels:
            df = self.create_labels(df)
        
        return df
    
    def get_feature_columns(self) -> list:
        """
        返回用於模型訓練的特徵列
        """
        bb_features = ['bb_width', 'bb_position', 'dist_to_upper', 'dist_to_lower',
                       'bb_width_change', 'bb_width_rank']
        
        trend_features = ['adx', 'plus_di', 'minus_di', 'trend_state_encoded']
        
        momentum_features = ['rsi', 'macd', 'macd_hist'] + \
                           [f'return_{p}' for p in [3, 5, 10, 20]] + \
                           [f'vol_ratio_{p}' for p in [3, 5, 10, 20]]
        
        slope_features = [f'slope_{p}' for p in [10, 20, 50]]
        
        pattern_features = ['consecutive_up', 'consecutive_down', 'body_size',
                           'upper_wick', 'lower_wick']
        
        history_features = ['upper_touch_count', 'lower_touch_count']
        
        return bb_features + trend_features + momentum_features + \
               slope_features + pattern_features + history_features