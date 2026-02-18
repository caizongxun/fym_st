import pandas as pd
import numpy as np
from typing import Tuple

class ScalpingLabelGenerator:
    """
    剝頭皮標籤生成器 (優化版)
    
    三分類標籤:
    - 0: 做空 (SHORT) - 預期下跌 >= target_pct
    - 1: 做多 (LONG) - 預期上漨 >= target_pct
    - 2: 觀望 (NEUTRAL) - 不明確，過濾
    
    優化:
    - 動態ATR調整目標
    - 增加標籤首達機制
    - 更嚴格的風報比過濾
    """
    
    def __init__(self,
                 target_pct: float = 0.003,
                 lookforward: int = 5,
                 risk_reward_ratio: float = 1.5,
                 use_dynamic_target: bool = True,
                 atr_multiplier: float = 1.0):
        
        self.target_pct = target_pct
        self.lookforward = lookforward
        self.risk_reward_ratio = risk_reward_ratio
        self.use_dynamic_target = use_dynamic_target
        self.atr_multiplier = atr_multiplier
    
    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成標籤 (優化版)
        """
        df = df.copy()
        
        # 計算ATR (用於動態目標)
        if self.use_dynamic_target and 'atr' in df.columns:
            # 使用ATR作為動態目標
            df['dynamic_target'] = (df['atr'] / df['close']) * self.atr_multiplier
            df['dynamic_target'] = df['dynamic_target'].clip(lower=self.target_pct * 0.5, 
                                                               upper=self.target_pct * 2.0)
        else:
            df['dynamic_target'] = self.target_pct
        
        labels = []
        label_info = []  # 記錄標籤資訊用於調試
        
        for i in range(len(df) - self.lookforward):
            current_close = df.iloc[i]['close']
            target_pct = df.iloc[i]['dynamic_target']
            
            # 未來N根K線的最高/最低
            future_slice = df.iloc[i+1:i+1+self.lookforward]
            future_high = future_slice['high'].max()
            future_low = future_slice['low'].min()
            
            # 計算潛在漲跌幅
            upside = (future_high - current_close) / current_close
            downside = (current_close - future_low) / current_close
            
            # 增加首達機制: 查看多久達到目標
            hit_target_up = False
            hit_target_down = False
            bars_to_hit_up = self.lookforward
            bars_to_hit_down = self.lookforward
            
            for j, (idx, row) in enumerate(future_slice.iterrows(), 1):
                if not hit_target_up and (row['high'] - current_close) / current_close >= target_pct:
                    hit_target_up = True
                    bars_to_hit_up = j
                if not hit_target_down and (current_close - row['low']) / current_close >= target_pct:
                    hit_target_down = True
                    bars_to_hit_down = j
                if hit_target_up and hit_target_down:
                    break
            
            # 標籤邏輯 (優先首達 + 風報比)
            if hit_target_up and upside >= target_pct:
                if hit_target_down:
                    # 兩個都達到，看誰先
                    if bars_to_hit_up < bars_to_hit_down and upside > downside * self.risk_reward_ratio:
                        label = 1  # LONG
                    elif bars_to_hit_down < bars_to_hit_up and downside > upside * self.risk_reward_ratio:
                        label = 0  # SHORT
                    else:
                        label = 2  # NEUTRAL (風報比不佳)
                else:
                    # 只有上漨達到
                    if upside > downside * self.risk_reward_ratio:
                        label = 1  # LONG
                    else:
                        label = 2  # NEUTRAL
            
            elif hit_target_down and downside >= target_pct:
                # 只有下跌達到
                if downside > upside * self.risk_reward_ratio:
                    label = 0  # SHORT
                else:
                    label = 2  # NEUTRAL
            
            else:
                # 兩個都沒達到
                label = 2  # NEUTRAL
            
            labels.append(label)
            
            # 記錄資訊
            label_info.append({
                'upside': upside,
                'downside': downside,
                'bars_to_hit_up': bars_to_hit_up if hit_target_up else None,
                'bars_to_hit_down': bars_to_hit_down if hit_target_down else None,
                'label': label
            })
        
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
        
        total = len(df[df['label'] != 2])
        
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
    
    def filter_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        過濾掉NEUTRAL樣本
        """
        valid_mask = df['label'].isin([0, 1])
        df_train = df[valid_mask].copy()
        
        return df_train

if __name__ == '__main__':
    print("Scalping Label Generator (Optimized)")
    print("請在App中使用")