import pandas as pd
import numpy as np
from typing import Tuple

class VolatilityLabelGenerator:
    """
    波動率預測標籤生成器
    
    不預測方向，只預測波動率大小
    
    二分類:
    - 0: 低波動 (LOW_VOL) - 未來N根K線波動 < 閑值
    - 1: 高波動 (HIGH_VOL) - 未來N根K線波動 >= 閑值
    
    交易邏輯:
    當模型預測 HIGH_VOL 時，同時掛做多+做空限價單，
    利用波動吸收兩邊價差
    """
    
    def __init__(self,
                 vol_threshold: float = 0.005,  # 0.5% 波動閑值
                 lookforward: int = 3,           # 未來3根K線
                 use_dynamic_threshold: bool = True,
                 atr_multiplier: float = 1.5):
        
        self.vol_threshold = vol_threshold
        self.lookforward = lookforward
        self.use_dynamic_threshold = use_dynamic_threshold
        self.atr_multiplier = atr_multiplier
    
    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成波動率標籤
        """
        df = df.copy()
        
        # 計算動態閑值 (使用ATR)
        if self.use_dynamic_threshold and 'atr' in df.columns:
            df['vol_threshold'] = (df['atr'] / df['close']) * self.atr_multiplier
            df['vol_threshold'] = df['vol_threshold'].clip(
                lower=self.vol_threshold * 0.5,
                upper=self.vol_threshold * 2.0
            )
        else:
            df['vol_threshold'] = self.vol_threshold
        
        labels = []
        actual_volatility = []
        time_to_peak = []
        
        for i in range(len(df) - self.lookforward):
            current_close = df.iloc[i]['close']
            threshold = df.iloc[i]['vol_threshold']
            
            # 未來N根K線的最高/最低
            future_slice = df.iloc[i+1:i+1+self.lookforward]
            future_high = future_slice['high'].max()
            future_low = future_slice['low'].min()
            
            # 計算實際波動率 (最大振幅)
            volatility = (future_high - future_low) / current_close
            
            # 找到高點和低點的時間
            high_idx = future_slice['high'].idxmax()
            low_idx = future_slice['low'].idxmin()
            high_time = future_slice.index.get_loc(high_idx) + 1
            low_time = future_slice.index.get_loc(low_idx) + 1
            peak_time = min(high_time, low_time)  # 取先達到的
            
            # 標籤
            if volatility >= threshold:
                label = 1  # HIGH_VOL
            else:
                label = 0  # LOW_VOL
            
            labels.append(label)
            actual_volatility.append(volatility)
            time_to_peak.append(peak_time)
        
        # 未來N根無法預測，填充LOW_VOL
        labels.extend([0] * self.lookforward)
        actual_volatility.extend([0.0] * self.lookforward)
        time_to_peak.extend([0] * self.lookforward)
        
        df['label'] = labels
        df['actual_volatility'] = actual_volatility
        df['time_to_peak'] = time_to_peak
        
        return df
    
    def get_label_distribution(self, df: pd.DataFrame) -> dict:
        """
        獲取標籤分布
        """
        if 'label' not in df.columns:
            return {}
        
        total = len(df)
        high_vol = (df['label'] == 1).sum()
        low_vol = (df['label'] == 0).sum()
        
        stats = {
            'total_samples': total,
            'high_vol_signals': high_vol,
            'low_vol_signals': low_vol,
            'high_vol_pct': high_vol / total * 100 if total > 0 else 0,
            'avg_volatility': df['actual_volatility'].mean(),
            'avg_time_to_peak': df[df['label'] == 1]['time_to_peak'].mean() if high_vol > 0 else 0
        }
        
        return stats
    
    def filter_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        過濾數據 (波動率預測不需要過濾)
        """
        return df.copy()

if __name__ == '__main__':
    print("Volatility Label Generator")
    print("請在App中使用")