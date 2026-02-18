import pandas as pd
import numpy as np
import joblib
import os
from utils.bb_reversal_features import BBReversalFeatureExtractor

class BBReversalSignalGenerator:
    """
    BB反轉信號生成器
    
    使用訓練好的LightGBM模型進行預測
    邏輯:
    1. 檢測BB觸碰 (上軌/下軌)
    2. 使用模型預測反轉類型 (0=做空, 1=做多)
    3. 規則過濾:
       - 觸碰上軌 AND 模型預測做空 -> SHORT
       - 觸碰下軌 AND 模型預測做多 -> LONG
       - 其他情況 -> 觀望 (過濾掉趨勢延續或矛盾信號)
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
        
        # 確保時間列名統一為 'open_time'
        if 'time' in df.columns and 'open_time' not in df.columns:
            df['open_time'] = pd.to_datetime(df['time'])
        elif 'open_time' in df.columns:
            df['open_time'] = pd.to_datetime(df['open_time'])
            
        # 1. 計算特徵 (與訓練時一致)
        # process會計算所有技術指標
        df_features = self.extractor.process(df, create_labels=False)
        
        # 確保open_time在結果中
        if 'open_time' not in df_features.columns:
            # 如果特徵提取過程中丟失了open_time，嘗試從原始df恢復
            # 注意: process可能會刪除NaN行，導致行數減少
            # 這裡我們重新賦值，假設索引是對齊的
            df_features['open_time'] = df.loc[df_features.index, 'open_time']
        
        # 2. 檢測觸碰候選點 (使用簡單規則)
        # 這裡我們需要重新計算BB，因為extractor.process可能過濾了數據
        # 但為了確保索引對齊，我們直接使用df_features
        
        # 確保有BB數據
        if 'bb_upper' not in df_features.columns:
            # 重新計算BB (理論上process已經計算了,但以防萬一)
            import ta
            bb = ta.volatility.BollingerBands(df_features['close'], window=self.bb_period, window_dev=self.bb_std)
            df_features['bb_upper'] = bb.bollinger_hband()
            df_features['bb_lower'] = bb.bollinger_lband()
        
        # 計算觸碰
        df_features['dist_to_upper_val'] = (df_features['bb_upper'] - df_features['high']) / df_features['bb_upper']
        df_features['dist_to_lower_val'] = (df_features['low'] - df_features['bb_lower']) / df_features['bb_lower']
        
        # 定義觸碰 (小於閾值或突破)
        # 注意: 這裡是為了找出"潛在"反轉點
        touch_upper = (df_features['dist_to_upper_val'] <= self.touch_threshold) | (df_features['close'] > df_features['bb_upper'])
        touch_lower = (df_features['dist_to_lower_val'] <= self.touch_threshold) | (df_features['close'] < df_features['bb_lower'])
        
        # 3. 準備模型輸入
        feature_cols = self.extractor.get_feature_columns()
        X = df_features[feature_cols]
        
        # 4. 模型預測
        # 我們對所有數據進行預測，然後過濾
        # LightGBM預測速度很快
        if len(X) > 0:
            y_pred = self.model.predict(X)
            y_prob = self.model.predict_proba(X)
            
            # 5. 生成信號
            df_features['signal'] = 0
            df_features['reversal_prob'] = 0.0
            
            # 邏輯匹配
            # 模型輸出: 0 = 做空 (Upper Reversal), 1 = 做多 (Lower Reversal)
            
            # SHORT信號: 觸碰上軌 + 模型預測做空(0)
            # 我們可以加一個概率閾值，例如 > 0.6 確信度
            short_mask = touch_upper & (y_pred == 0)
            df_features.loc[short_mask, 'signal'] = -1
            df_features.loc[short_mask, 'reversal_prob'] = y_prob[short_mask, 0]  # Class 0 prob
            
            # LONG信號: 觸碰下軌 + 模型預測做多(1)
            long_mask = touch_lower & (y_pred == 1)
            df_features.loc[long_mask, 'signal'] = 1
            df_features.loc[long_mask, 'reversal_prob'] = y_prob[long_mask, 1]  # Class 1 prob
        else:
            df_features['signal'] = 0
            df_features['reversal_prob'] = 0.0
        
        # 6. 設置止盈止損 (可選，這裡使用ATR)
        # BacktestEngine會處理，但我們可以在這裡提供建議值
        # 例如: TP = 中軌, SL = 軌道外一定距離
        # 這裡我們讓BacktestEngine使用ATR動態止盈止損，所以不強制寫入tp_price/sl_price
        
        return df_features

if __name__ == '__main__':
    print("BB反轉信號生成器")
    print("請在App中使用")