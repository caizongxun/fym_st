import pandas as pd
import numpy as np
from typing import Tuple

class ScalpingLabelGenerator:
    """
    剝頭皮標籤生成器
    
    三分類標籤:
    - 0: 做空 (SHORT) - 預期下跌 >= target_pct
    - 1: 做多 (LONG) - 預期上漨 >= target_pct
    - 2: 觀望 (NEUTRAL) - 不明確，過濾
    """
    
    def __init__(self,
                 target_pct: float = 0.003,  # 0.3% 目標利潤
                 lookforward: int = 5,        # 未來5根K線
                 risk_reward_ratio: float = 1.5):  # 風報比
        
        self.target_pct = target_pct
        self.lookforward = lookforward
        self.risk_reward_ratio = risk_reward_ratio
    
    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成標籤
        """
        df = df.copy()
        labels = []
        
        for i in range(len(df) - self.lookforward):
            current_close = df.iloc[i]['close']
            
            # 未來N根K線的最高/最低
            future_slice = df.iloc[i+1:i+1+self.lookforward]
            future_high = future_slice['high'].max()
            future_low = future_slice['low'].min()
            
            # 計算潛在漲跌幅
            upside = (future_high - current_close) / current_close
            downside = (current_close - future_low) / current_close
            
            # 標籤邏輯
            if upside >= self.target_pct and upside > downside * self.risk_reward_ratio:
                # 上漨空間明顯且風報比好
                label = 1  # LONG
            elif downside >= self.target_pct and downside > upside * self.risk_reward_ratio:
                # 下跌空間明顯且風報比好
                label = 0  # SHORT
            else:
                # 不明確或風報比不佳
                label = 2  # NEUTRAL
            
            labels.append(label)
        
        # 未來N根無法預測，填充NEUTRAL
        labels.extend([2] * self.lookforward)
        
        df['label'] = labels
        
        return df
    
    def get_label_distribution(self, df: pd.DataFrame) -> dict:
        """
        獲取標籤分布
        """
        if 'label' not in df.columns:
            return {}
        
        total = len(df[df['label'] != 2])  # 排除NEUTRAL
        
        stats = {
            'total_samples': len(df),
            'long_signals': (df['label'] == 1).sum(),
            'short_signals': (df['label'] == 0).sum(),
            'neutral_signals': (df['label'] == 2).sum(),
            'long_pct': (df['label'] == 1).sum() / total * 100 if total > 0 else 0,
            'short_pct': (df['label'] == 0).sum() / total * 100 if total > 0 else 0,
            'tradeable_pct': total / len(df) * 100 if len(df) > 0 else 0
        }
        
        return stats
    
    def filter_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        過濾掉NEUTRAL樣本，返回可訓練的數據
        """
        # 只保留LONG和SHORT樣本
        valid_mask = df['label'].isin([0, 1])
        df_train = df[valid_mask].copy()
        
        return df_train

if __name__ == '__main__':
    print("Scalping Label Generator")
    print("請在App中使用")