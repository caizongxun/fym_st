import pandas as pd
import numpy as np
import joblib
import os

from utils.nextbar_feature_extractor import NextBarFeatureExtractor

class NextBarSignalGenerator:
    """
    下一根K棒信號生成器
    
    根據模型預測生成雙向掛單信號
    """
    
    def __init__(self, 
                 model_path: str,
                 entry_offset_pct: float = 0.001,
                 tp_buffer_pct: float = 0.0,
                 sl_buffer_pct: float = 0.002,
                 max_range_filter: float = 0.008,
                 min_range_filter: float = 0.002):
        
        self.model_path = model_path
        self.entry_offset_pct = entry_offset_pct  # 掛單偏移 (0.1%)
        self.tp_buffer_pct = tp_buffer_pct        # 止盈緩衝 (0%)
        self.sl_buffer_pct = sl_buffer_pct        # 止損緩衝 (0.2%)
        self.max_range_filter = max_range_filter  # 最大區間過濾 (0.8%)
        self.min_range_filter = min_range_filter  # 最小區間過濾 (0.2%)
        
        self.feature_extractor = NextBarFeatureExtractor()
        self.model_high = None
        self.model_low = None
        self.feature_columns = []
        
        self._load_model()
    
    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型不存在: {self.model_path}")
        
        model_package = joblib.load(self.model_path)
        self.model_high = model_package['model_high']
        self.model_low = model_package['model_low']
        self.feature_columns = model_package['feature_columns']
        
        print(f"模型已載入: {self.model_path}")
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信號
        
        Returns:
            df with columns:
            - signal: 1=做多, -1=做空, 0=觀望
            - pred_high: 預測高點
            - pred_low: 預測低點
            - pred_range: 預測區間
            - entry_long: 做多進場價
            - entry_short: 做空進場價
            - tp_long/sl_long: 做多止盈止損
            - tp_short/sl_short: 做空止盈止損
        """
        df = df.copy()
        
        # 1. 提取特徵
        print("提取特徵...")
        df = self.feature_extractor.extract_features(df)
        
        # 2. 提取需要的特徵
        X = df[self.feature_columns].fillna(0)
        
        # 3. 預測
        print("生成預測...")
        pred_high_pct = self.model_high.predict(X)
        pred_low_pct = self.model_low.predict(X)
        
        # 4. 計算預測價格
        df['pred_high_pct'] = pred_high_pct
        df['pred_low_pct'] = pred_low_pct
        df['pred_range_pct'] = pred_high_pct - pred_low_pct
        
        df['pred_high'] = df['close'] * (1 + df['pred_high_pct'])
        df['pred_low'] = df['close'] * (1 + df['pred_low_pct'])
        
        # 5. 過濾條件
        valid_signal = (
            (df['pred_range_pct'] > self.min_range_filter) &  # 區間不太小
            (df['pred_range_pct'] < self.max_range_filter)    # 區間不太大
        )
        
        # 6. 生成信號 (雙向掛單)
        df['signal'] = 0
        df.loc[valid_signal, 'signal'] = 2  # 2 表示雙向掛單
        
        # 7. 計算進場價 (在預測價位留一點空間)
        df['entry_long'] = df['pred_low'] * (1 - self.entry_offset_pct)
        df['entry_short'] = df['pred_high'] * (1 + self.entry_offset_pct)
        
        # 8. 計算止盈止損
        # 做多: 止盈=預測高點, 止損=預測低點-buffer
        df['tp_long'] = df['pred_high'] * (1 - self.tp_buffer_pct)
        df['sl_long'] = df['pred_low'] * (1 - self.sl_buffer_pct)
        
        # 做空: 止盈=預測低點, 止損=預測高點+buffer
        df['tp_short'] = df['pred_low'] * (1 + self.tp_buffer_pct)
        df['sl_short'] = df['pred_high'] * (1 + self.sl_buffer_pct)
        
        # 9. 計算預期風報比
        df['rr_long'] = (df['tp_long'] - df['entry_long']) / (df['entry_long'] - df['sl_long'] + 1e-10)
        df['rr_short'] = (df['entry_short'] - df['tp_short']) / (df['sl_short'] - df['entry_short'] + 1e-10)
        
        print(f"生成 {(df['signal'] == 2).sum()} 個雙向掛單信號")
        
        return df

if __name__ == '__main__':
    print("Next Bar Signal Generator")
    print("請在App中使用")