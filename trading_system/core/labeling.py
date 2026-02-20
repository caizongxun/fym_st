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
                 quality_weight_alpha: float = 2.0):
        self.tp_multiplier = tp_multiplier
        self.sl_multiplier = sl_multiplier
        self.max_holding_bars = max_holding_bars
        self.slippage = slippage
        self.time_decay_lambda = time_decay_lambda
        self.quality_weight_alpha = quality_weight_alpha
    
    def apply_triple_barrier(self, df: pd.DataFrame, atr_column: str = 'atr') -> pd.DataFrame:
        logger.info(f"應用三重屏障標記 TP={self.tp_multiplier}x, SL={self.sl_multiplier}x, MaxHold={self.max_holding_bars}")
        logger.info(f"使用質量評分: time_decay_lambda={self.time_decay_lambda}, alpha={self.quality_weight_alpha}")
        
        result = df.copy()
        labels = []
        returns = []
        hit_times = []
        exit_types = []
        sample_weights = []
        mae_values = []  # Maximum Adverse Excursion
        
        for i in range(len(df) - self.max_holding_bars - 1):  # -1 因為需要下一根K線
            # 第 i 根 K 線收盤時計算特徵和信號
            # 第 i+1 根 K 線開盤時入場
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
            
            # 從第 i+2 根開始觀察價格變動
            for j in range(1, self.max_holding_bars + 1):
                if i + 1 + j >= len(df):
                    break
                
                current_high = df.iloc[i + 1 + j]['high']
                current_low = df.iloc[i + 1 + j]['low']
                current_close = df.iloc[i + 1 + j]['close']
                
                # 計算 MAE (向下的最大不利偏移)
                adverse_move = max(0, entry_price - current_low)
                max_adverse_excursion = max(max_adverse_excursion, adverse_move)
                
                # 檢查是否觸及止盈
                if current_high >= upper_barrier:
                    label = 1
                    ret = (upper_barrier - entry_price) / entry_price
                    hit_time = j
                    exit_type = 'TP'
                    break
                # 檢查是否觸及止損
                elif current_low <= lower_barrier:
                    label = 0
                    ret = (lower_barrier - entry_price) / entry_price
                    hit_time = j
                    exit_type = 'SL'
                    break
            
            # 超時出場
            if exit_type == 'Timeout':
                if i + 1 + self.max_holding_bars < len(df):
                    final_price = df.iloc[i + 1 + self.max_holding_bars]['close']
                    ret = (final_price - entry_price) / entry_price
                    if ret > 0:
                        label = 1
                    adverse_move = max(0, entry_price - df.iloc[i + 1 + self.max_holding_bars]['low'])
                    max_adverse_excursion = max(max_adverse_excursion, adverse_move)
            
            # 計算樣本質量權重
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
        
        # 填充最後的樣本
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
        計算樣本權重
        - 負類樣本: 固定權重 1.0
        - 正類樣本: 根據持倉時間和MAE調整權重
        """
        if label == 0:
            return 1.0
        
        # 時間衰減因子: 快速止盈權重高
        time_factor = np.exp(-self.time_decay_lambda * hit_time / self.max_holding_bars)
        
        # MAE 懲罰因子: 低回撤權重高
        if sl_distance > 0:
            drawdown_factor = 1.0 - (mae / sl_distance)
            drawdown_factor = max(0.0, min(1.0, drawdown_factor))  # 限制在 [0, 1]
        else:
            drawdown_factor = 1.0
        
        # 綜合質量分數
        quality_score = time_factor * drawdown_factor
        
        # 最終權重
        weight = 1.0 + (self.quality_weight_alpha * quality_score)
        
        return weight
    
    def calculate_sample_weights(self, df: pd.DataFrame) -> np.ndarray:
        """返回已計算的樣本權重"""
        if 'sample_weight' in df.columns:
            return df['sample_weight'].values
        else:
            # 向下兼容: 如果沒有權重列,返回全1
            return np.ones(len(df))

class MetaLabeling:
    """
    Meta-Labeling: 在已有交易信號的基礎上,預測該信號的質量
    不是預測方向,而是預測「這個做多信號是否應該執行」
    """
    def __init__(self, primary_model):
        self.primary_model = primary_model
    
    def generate_meta_labels(self, df: pd.DataFrame, signals: pd.Series) -> pd.DataFrame:
        """
        df: 包含標籤的數據框
        signals: 主模型產生的交易信號 (1=做多, 0=不交易)
        
        返回: 只包含信號為1的樣本,標籤表示該信號是否盈利
        """
        meta_df = df[signals == 1].copy()
        meta_df['meta_label'] = (meta_df['label_return'] > 0).astype(int)
        return meta_df