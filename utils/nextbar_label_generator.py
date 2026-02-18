import pandas as pd
import numpy as np
from typing import Tuple

class NextBarLabelGenerator:
    """
    下一根K棒高低點標籤生成器
    
    預測下一根K棒的:
    - high_pct: 最高價相對當前收盤價的百分比
    - low_pct: 最低價相對當前收盤價的百分比
    """
    
    def __init__(self, 
                 lookback: int = 20,
                 use_log_return: bool = False):
        
        self.lookback = lookback
        self.use_log_return = use_log_return
    
    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成標籤
        """
        df = df.copy()
        
        # 計算下一根K棒的high/low相對百分比
        if self.use_log_return:
            # 對數收益率
            df['next_high_pct'] = np.log((df['high'].shift(-1) + 1e-10) / (df['close'] + 1e-10))
            df['next_low_pct'] = np.log((df['low'].shift(-1) + 1e-10) / (df['close'] + 1e-10))
        else:
            # 簡單百分比變化
            df['next_high_pct'] = (df['high'].shift(-1) - df['close']) / (df['close'] + 1e-10)
            df['next_low_pct'] = (df['low'].shift(-1) - df['close']) / (df['close'] + 1e-10)
        
        # 計算預測區間寬度
        df['next_range_pct'] = df['next_high_pct'] - df['next_low_pct']
        
        # 計算實際振幅
        df['next_actual_range'] = df['high'].shift(-1) - df['low'].shift(-1)
        
        # 記錄下一根K棒的實際價格
        df['next_open'] = df['open'].shift(-1)
        df['next_high'] = df['high'].shift(-1)
        df['next_low'] = df['low'].shift(-1)
        df['next_close'] = df['close'].shift(-1)
        
        # 清理異常值
        df['next_high_pct'] = df['next_high_pct'].replace([np.inf, -np.inf], np.nan)
        df['next_low_pct'] = df['next_low_pct'].replace([np.inf, -np.inf], np.nan)
        df['next_range_pct'] = df['next_range_pct'].replace([np.inf, -np.inf], np.nan)
        
        # 限制極端值 (最多 10%)
        df['next_high_pct'] = df['next_high_pct'].clip(-0.10, 0.10)
        df['next_low_pct'] = df['next_low_pct'].clip(-0.10, 0.10)
        
        return df
    
    def get_label_statistics(self, df: pd.DataFrame) -> dict:
        """
        獲取標籤統計信息
        """
        if 'next_high_pct' not in df.columns:
            return {}
        
        valid_df = df[df['next_high_pct'].notna()].copy()
        
        if len(valid_df) == 0:
            return {
                'total_samples': 0,
                'avg_high_pct': 0,
                'avg_low_pct': 0,
                'avg_range_pct': 0,
                'std_high_pct': 0,
                'std_low_pct': 0,
                'std_range_pct': 0,
                'max_range_pct': 0,
                'min_range_pct': 0
            }
        
        stats = {
            'total_samples': len(valid_df),
            'avg_high_pct': valid_df['next_high_pct'].mean(),
            'avg_low_pct': valid_df['next_low_pct'].mean(),
            'avg_range_pct': valid_df['next_range_pct'].mean(),
            'std_high_pct': valid_df['next_high_pct'].std(),
            'std_low_pct': valid_df['next_low_pct'].std(),
            'std_range_pct': valid_df['next_range_pct'].std(),
            'max_range_pct': valid_df['next_range_pct'].max(),
            'min_range_pct': valid_df['next_range_pct'].min()
        }
        
        stats['high_pct_quantiles'] = {
            'p10': valid_df['next_high_pct'].quantile(0.1),
            'p50': valid_df['next_high_pct'].quantile(0.5),
            'p90': valid_df['next_high_pct'].quantile(0.9)
        }
        
        stats['low_pct_quantiles'] = {
            'p10': valid_df['next_low_pct'].quantile(0.1),
            'p50': valid_df['next_low_pct'].quantile(0.5),
            'p90': valid_df['next_low_pct'].quantile(0.9)
        }
        
        return stats
    
    def filter_training_data(self, df: pd.DataFrame, 
                            max_range_pct: float = 0.015) -> pd.DataFrame:
        """
        過濾異常數據
        """
        # 移除最後一根無效數據
        df_filtered = df[df['next_high_pct'].notna()].copy()
        
        if len(df_filtered) == 0:
            return df_filtered
        
        # 過濾異常大的波動
        df_filtered = df_filtered[
            df_filtered['next_range_pct'] < max_range_pct
        ].copy()
        
        # 過濾異常小的波動 (太小的波動難以預測)
        df_filtered = df_filtered[
            df_filtered['next_range_pct'] > 0.0001
        ].copy()
        
        return df_filtered

if __name__ == '__main__':
    print("Next Bar Label Generator")
    print("請在App中使用")