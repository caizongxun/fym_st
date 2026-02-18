import pandas as pd
import numpy as np
import joblib
import os
from utils.scalping_feature_extractor import ScalpingFeatureExtractor

class ScalpingSignalGenerator:
    """
    剝頭皮信號生成器
    
    特點:
    - 使用模型置信度過濾 (confidence > threshold)
    - 自動計算 Limit Order 進場價
    - 自動計算 TP/SL
    """
    
    def __init__(self, 
                 model_path: str,
                 confidence_threshold: float = 0.65,
                 entry_offset_pct: float = 0.001,  # 進場價偏移 0.1%
                 tp_pct: float = 0.003,            # 止盈 0.3%
                 sl_pct: float = 0.002):           # 止損 0.2%
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 載入模型包
        model_package = joblib.load(model_path)
        self.model = model_package['model']
        self.feature_columns = model_package['feature_columns']
        
        self.confidence_threshold = confidence_threshold
        self.entry_offset_pct = entry_offset_pct
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        
        # 特徵提取器
        self.extractor = ScalpingFeatureExtractor()
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信號
        
        Returns:
            df with columns:
            - signal: 1=LONG, -1=SHORT, 0=NEUTRAL
            - confidence: 模型置信度
            - limit_price: 限價進場價
            - tp_price: 止盈價
            - sl_price: 止損價
        """
        df = df.copy()
        
        # 確保有 open_time
        if 'time' in df.columns and 'open_time' not in df.columns:
            df['open_time'] = pd.to_datetime(df['time'])
        elif 'open_time' in df.columns:
            df['open_time'] = pd.to_datetime(df['open_time'])
        
        # 1. 提取特徵
        df_features = self.extractor.extract_features(df)
        
        # 恢復 open_time (同 BB版本的過程)
        if 'open_time' not in df_features.columns:
            if 'open_time' in df.columns:
                try:
                    df_features = df_features.join(df[['open_time']], how='left')
                except Exception:
                    if len(df_features) == len(df):
                        df_features['open_time'] = df['open_time'].values
        
        # 2. 準備模型輸入
        X = df_features[self.feature_columns].fillna(0)
        
        # 3. 預測
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)
        
        # 4. 生成信號
        df_features['signal'] = 0
        df_features['confidence'] = 0.0
        df_features['limit_price'] = 0.0
        df_features['tp_price'] = 0.0
        df_features['sl_price'] = 0.0
        
        for i in range(len(df_features)):
            pred_class = y_pred[i]
            prob = y_prob[i]
            
            # 獲取預測類別的置信度
            if pred_class == 1:  # LONG
                conf = prob[1]
                if conf >= self.confidence_threshold:
                    df_features.iloc[i, df_features.columns.get_loc('signal')] = 1
                    df_features.iloc[i, df_features.columns.get_loc('confidence')] = conf
                    
                    # 計算 Limit Order 價格 (等回調)
                    current_close = df_features.iloc[i]['close']
                    df_features.iloc[i, df_features.columns.get_loc('limit_price')] = current_close * (1 - self.entry_offset_pct)
                    df_features.iloc[i, df_features.columns.get_loc('tp_price')] = current_close * (1 + self.tp_pct)
                    df_features.iloc[i, df_features.columns.get_loc('sl_price')] = current_close * (1 - self.sl_pct)
            
            elif pred_class == 0:  # SHORT
                conf = prob[0]
                if conf >= self.confidence_threshold:
                    df_features.iloc[i, df_features.columns.get_loc('signal')] = -1
                    df_features.iloc[i, df_features.columns.get_loc('confidence')] = conf
                    
                    current_close = df_features.iloc[i]['close']
                    df_features.iloc[i, df_features.columns.get_loc('limit_price')] = current_close * (1 + self.entry_offset_pct)
                    df_features.iloc[i, df_features.columns.get_loc('tp_price')] = current_close * (1 - self.tp_pct)
                    df_features.iloc[i, df_features.columns.get_loc('sl_price')] = current_close * (1 + self.sl_pct)
        
        return df_features

if __name__ == '__main__':
    print("Scalping Signal Generator")
    print("請在App中使用")