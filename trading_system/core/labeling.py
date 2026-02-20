import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TripleBarrierLabeling:
    def __init__(self, 
                 tp_multiplier: float = 2.5, 
                 sl_multiplier: float = 1.5, 
                 max_holding_bars: int = 24,
                 slippage: float = 0.001,
                 time_decay_lambda: float = 2.0,
                 quality_weight_alpha: float = 0.5,  # 降低預設值
                 use_quality_weight: bool = True):
        self.tp_multiplier = tp_multiplier
        self.sl_multiplier = sl_multiplier
        self.max_holding_bars = max_holding_bars
        self.slippage = slippage
        self.time_decay_lambda = time_decay_lambda
        self.quality_weight_alpha = quality_weight_alpha
        self.use_quality_weight = use_quality_weight
    
    def apply_triple_barrier(self, df: pd.DataFrame, atr_column: str = 'atr') -> pd.DataFrame:
        logger.info(f"應用三重屏障標記 TP={self.tp_multiplier}x, SL={self.sl_multiplier}x, MaxHold={self.max_holding_bars}")
        if self.use_quality_weight:
            logger.info(f"使用質量微調: time_decay_lambda={self.time_decay_lambda}, alpha={self.quality_weight_alpha}")
        else:
            logger.info("樣本權重: 全部 1.0 (禁用質量加權)")
        
        result = df.copy()
        labels = []
        returns = []
        hit_times = []
        exit_types = []
        sample_weights = []
        mae_values = []
        
        for i in range(len(df) - self.max_holding_bars - 1):
            if i + 1 >= len(df):
                break
                
            entry_price = df.iloc[i + 1]['open'] * (1 + self.slippage)
            atr_value = df.iloc[i][atr_column]
            
            if pd.isna(atr_value) or atr_value <= 0:
                labels.append(np.nan)
                returns.append(np.nan)
                hit_times.append(np.nan)
                exit_types.append(np.nan)
                sample_weights.append(np.nan)
                mae_values.append(np.nan)
                continue
            
            upper_barrier = entry_price + (self.tp_multiplier * atr_value)
            lower_barrier = entry_price - (self.sl_multiplier * atr_value)
            
            label = 0
            ret = 0
            hit_time = self.max_holding_bars
            exit_type = 'Timeout'
            max_adverse_excursion = 0
            
            for j in range(1, self.max_holding_bars + 1):
                if i + 1 + j >= len(df):
                    break
                
                current_high = df.iloc[i + 1 + j]['high']
                current_low = df.iloc[i + 1 + j]['low']
                current_close = df.iloc[i + 1 + j]['close']
                
                adverse_move = max(0, entry_price - current_low)
                max_adverse_excursion = max(max_adverse_excursion, adverse_move)
                
                if current_high >= upper_barrier:
                    label = 1
                    ret = (upper_barrier - entry_price) / entry_price
                    hit_time = j
                    exit_type = 'TP'
                    break
                elif current_low <= lower_barrier:
                    label = 0
                    ret = (lower_barrier - entry_price) / entry_price
                    hit_time = j
                    exit_type = 'SL'
                    break
            
            if exit_type == 'Timeout':
                if i + 1 + self.max_holding_bars < len(df):
                    final_price = df.iloc[i + 1 + self.max_holding_bars]['close']
                    ret = (final_price - entry_price) / entry_price
                    if ret > 0:
                        label = 1
                    adverse_move = max(0, entry_price - df.iloc[i + 1 + self.max_holding_bars]['low'])
                    max_adverse_excursion = max(max_adverse_excursion, adverse_move)
            
            weight = self._calculate_sample_weight(
                label, hit_time, max_adverse_excursion, 
                self.sl_multiplier * atr_value
            )
            
            labels.append(label)
            returns.append(ret)
            hit_times.append(hit_time)
            exit_types.append(exit_type)
            sample_weights.append(weight)
            mae_values.append(max_adverse_excursion / atr_value if atr_value > 0 else 0)
        
        for _ in range(self.max_holding_bars + 1):
            labels.append(np.nan)
            returns.append(np.nan)
            hit_times.append(np.nan)
            exit_types.append(np.nan)
            sample_weights.append(np.nan)
            mae_values.append(np.nan)
        
        result['label'] = labels
        result['label_return'] = returns
        result['hit_time'] = hit_times
        result['exit_type'] = exit_types
        result['sample_weight'] = sample_weights
        result['mae_ratio'] = mae_values
        
        result = result.dropna(subset=['label'])
        
        positive_count = (result['label'] == 1).sum()
        total_count = len(result)
        avg_weight_positive = result[result['label'] == 1]['sample_weight'].mean()
        avg_weight_negative = result[result['label'] == 0]['sample_weight'].mean()
        
        logger.info(f"標記完成: {positive_count}/{total_count} 正樣本 ({100*positive_count/total_count:.1f}%)")
        logger.info(f"平均樣本權重 - 正類: {avg_weight_positive:.2f}, 負類: {avg_weight_negative:.2f}")
        
        return result
    
    def _calculate_sample_weight(self, label: int, hit_time: int, mae: float, sl_distance: float) -> float:
        """
        計算樣本權重 - 修正版
        - 負類樣本: 固定權重 1.0
        - 正類樣本: 基礎權重 1.0 + 微量質量調整 (0-0.5)
        
        關鍵修正: 不再放大正類基礎權重,只用質量分數微調
        """
        # 所有樣本基礎權重為 1.0
        base_weight = 1.0
        
        # 如果禁用質量加權,所有樣本都是 1.0
        if not self.use_quality_weight:
            return base_weight
        
        # 負類固定為 1.0
        if label == 0:
            return base_weight
        
        # 正類: 計算質量分數作為微調
        time_factor = np.exp(-self.time_decay_lambda * hit_time / self.max_holding_bars)
        
        if sl_distance > 0:
            drawdown_factor = 1.0 - (mae / sl_distance)
            drawdown_factor = max(0.0, min(1.0, drawdown_factor))
        else:
            drawdown_factor = 1.0
        
        quality_score = time_factor * drawdown_factor
        
        # 關鍵: 只加上小量調整,不放大基礎權重
        # 範圍: 1.0 到 1.0 + alpha (alpha 預設 0.5)
        weight = base_weight + (self.quality_weight_alpha * quality_score)
        
        return weight
    
    def calculate_sample_weights(self, df: pd.DataFrame) -> np.ndarray:
        if 'sample_weight' in df.columns:
            return df['sample_weight'].values
        else:
            return np.ones(len(df))

class MetaLabeling:
    def __init__(self, primary_model):
        self.primary_model = primary_model
    
    def generate_meta_labels(self, df: pd.DataFrame, signals: pd.Series) -> pd.DataFrame:
        meta_df = df[signals == 1].copy()
        meta_df['meta_label'] = (meta_df['label_return'] > 0).astype(int)
        return meta_df