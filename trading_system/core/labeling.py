import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TripleBarrierLabeling:
    def __init__(self, tp_multiplier: float = 2.5, sl_multiplier: float = 1.5, max_holding_bars: int = 24):
        self.tp_multiplier = tp_multiplier
        self.sl_multiplier = sl_multiplier
        self.max_holding_bars = max_holding_bars
    
    def apply_triple_barrier(self, df: pd.DataFrame, atr_column: str = 'atr') -> pd.DataFrame:
        logger.info(f"Applying triple barrier labeling with TP={self.tp_multiplier}x, SL={self.sl_multiplier}x, MaxHold={self.max_holding_bars}")
        
        result = df.copy()
        labels = []
        returns = []
        hit_times = []
        
        for i in range(len(df) - self.max_holding_bars):
            entry_price = df.iloc[i]['close']
            atr_value = df.iloc[i][atr_column]
            
            if pd.isna(atr_value) or atr_value <= 0:
                labels.append(np.nan)
                returns.append(np.nan)
                hit_times.append(np.nan)
                continue
            
            upper_barrier = entry_price + (self.tp_multiplier * atr_value)
            lower_barrier = entry_price - (self.sl_multiplier * atr_value)
            
            label = 0
            ret = 0
            hit_time = self.max_holding_bars
            
            for j in range(1, self.max_holding_bars + 1):
                if i + j >= len(df):
                    break
                
                current_price = df.iloc[i + j]['close']
                
                if current_price >= upper_barrier:
                    label = 1
                    ret = (current_price - entry_price) / entry_price
                    hit_time = j
                    break
                elif current_price <= lower_barrier:
                    label = 0
                    ret = (current_price - entry_price) / entry_price
                    hit_time = j
                    break
            
            if label == 0 and hit_time == self.max_holding_bars:
                final_price = df.iloc[i + self.max_holding_bars]['close'] if i + self.max_holding_bars < len(df) else entry_price
                ret = (final_price - entry_price) / entry_price
            
            labels.append(label)
            returns.append(ret)
            hit_times.append(hit_time)
        
        for _ in range(self.max_holding_bars):
            labels.append(np.nan)
            returns.append(np.nan)
            hit_times.append(np.nan)
        
        result['label'] = labels
        result['label_return'] = returns
        result['hit_time'] = hit_times
        
        result = result.dropna(subset=['label'])
        
        positive_count = (result['label'] == 1).sum()
        total_count = len(result)
        logger.info(f"Labeling complete: {positive_count}/{total_count} positive labels ({100*positive_count/total_count:.1f}%)")
        
        return result
    
    def calculate_sample_weights(self, df: pd.DataFrame) -> np.ndarray:
        abs_returns = np.abs(df['label_return'])
        atr_values = df['atr']
        normalized_returns = abs_returns / atr_values
        weights = np.log1p(normalized_returns)
        weights = weights / weights.sum()
        return weights.values