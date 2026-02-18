import pandas as pd
import numpy as np
import joblib
import os
from utils.bb_reversal_features import BBReversalFeatureExtractor
import ta

class BBReversalSignalGenerator:
    """
    BB反轉信號生成器 (增強版)
    
    加入嚴格過濾條件以提高勝率:
    1. K線顏色確認 (做多收紅不可進, 做空收綠不可進)
    2. RSI輔助判斷
    3. BB帶寬過濾
    """
    
    def __init__(self, model_path: str, 
                 bb_period: int = 20, bb_std: float = 2.0, 
                 touch_threshold: float = 0.001):
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
        self.model = joblib.load(model_path)
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.touch_threshold = touch_threshold
        
        # 初始化特徵提取器
        self.extractor = BBReversalFeatureExtractor(
            bb_period=bb_period,
            bb_std=bb_std,
            rsi_period=14
        )
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信號
        """
        df = df.copy()
        
        # 標準化時間列
        if 'time' in df.columns and 'open_time' not in df.columns:
            df['open_time'] = pd.to_datetime(df['time'])
        elif 'open_time' in df.columns:
            df['open_time'] = pd.to_datetime(df['open_time'])
            
        # 1. 計算特徵
        df_features = self.extractor.process(df, create_labels=False)
        
        # 恢復open_time
        if 'open_time' not in df_features.columns:
            df_features['open_time'] = df.loc[df_features.index, 'open_time']
        
        # 確保有BB數據
        if 'bb_upper' not in df_features.columns:
            bb = ta.volatility.BollingerBands(df_features['close'], window=self.bb_period, window_dev=self.bb_std)
            df_features['bb_upper'] = bb.bollinger_hband()
            df_features['bb_lower'] = bb.bollinger_lband()
            df_features['bb_width'] = (df_features['bb_upper'] - df_features['bb_lower']) / df_features['bb_middle']
        
        # 計算觸碰
        df_features['dist_to_upper_val'] = (df_features['bb_upper'] - df_features['high']) / df_features['bb_upper']
        df_features['dist_to_lower_val'] = (df_features['low'] - df_features['bb_lower']) / df_features['bb_lower']
        
        touch_upper = (df_features['dist_to_upper_val'] <= self.touch_threshold) | (df_features['close'] > df_features['bb_upper'])
        touch_lower = (df_features['dist_to_lower_val'] <= self.touch_threshold) | (df_features['close'] < df_features['bb_lower'])
        
        # 準備模型輸入
        feature_cols = self.extractor.get_feature_columns()
        X = df_features[feature_cols]
        
        # 初始化信號
        df_features['signal'] = 0
        df_features['reversal_prob'] = 0.0
        
        if len(X) > 0:
            y_pred = self.model.predict(X)
            y_prob = self.model.predict_proba(X)
            
            # ====== 過濾條件 ======
            
            # 1. K線顏色 (Close vs Open)
            is_green = df_features['close'] > df_features['open']
            is_red = df_features['close'] < df_features['open']
            
            # 2. RSI 過濾 (避免在強烈趨勢中逆勢)
            # RSI太高不宜做多(但也許適合做空)，RSI太低不宜做空
            rsi = df_features['rsi']
            
            # 3. BB寬度過濾 (過濾極度窄幅盤整，容易爆發突破而非反轉)
            # 簡單用百分位數或絕對值，這裡暫用 > 0.01 (1%)
            valid_width = df_features['bb_width'] > 0.01
            
            # ====== 生成信號 ======
            
            # SHORT信號: 
            # 1. 觸碰上軌 
            # 2. 模型預測做空(0) 
            # 3. 收陰線 (證明有壓力) OR 雖然收陽但留長上影線? 暫只用收陰
            # 4. RSI > 50 (確保不是在超賣區做空)
            short_mask = (
                touch_upper & 
                (y_pred == 0) & 
                (y_prob[:, 0] > 0.6) &  # 概率 > 60%
                is_red &                # 必須收陰線 (確認反轉開始)
                (rsi > 50) &            # RSI相對高位
                valid_width
            )
            
            df_features.loc[short_mask, 'signal'] = -1
            df_features.loc[short_mask, 'reversal_prob'] = y_prob[short_mask, 0]
            
            # LONG信號:
            # 1. 觸碰下軌
            # 2. 模型預測做多(1)
            # 3. 收陽線 (證明有支撐)
            # 4. RSI < 50 (確保不是在超買區做多)
            long_mask = (
                touch_lower & 
                (y_pred == 1) & 
                (y_prob[:, 1] > 0.6) &  # 概率 > 60%
                is_green &              # 必須收陽線 (確認反轉開始)
                (rsi < 50) &            # RSI相對低位
                valid_width
            )
            
            df_features.loc[long_mask, 'signal'] = 1
            df_features.loc[long_mask, 'reversal_prob'] = y_prob[long_mask, 1]
            
        return df_features

if __name__ == '__main__':
    print("BB反轉信號生成器 (增強版)")