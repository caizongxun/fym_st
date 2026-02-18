import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict
from utils.dual_model_features_v2 import EnhancedDualModelFeatureExtractor

class DualModelSignalGeneratorLGB:
    """
    雙模型信號生成器 - LightGBM版
    
    使用訓練好的LightGBM模型生成交易信號
    """
    
    def __init__(self,
                 model_dir: str = 'models/saved',
                 model_prefix: str = '',
                 min_confidence: float = 0.55,
                 tp_safety_factor: float = 0.90,
                 sl_cushion: float = 0.05,
                 min_reward_risk: float = 1.2):
        self.model_dir = model_dir
        self.model_prefix = model_prefix
        self.min_confidence = min_confidence
        self.tp_safety_factor = tp_safety_factor
        self.sl_cushion = sl_cushion
        self.min_reward_risk = min_reward_risk
        
        self.direction_model = None
        self.high_model = None
        self.low_model = None
        self.extractor = EnhancedDualModelFeatureExtractor()
        
        self._load_models()
    
    def _load_models(self):
        prefix = f"{self.model_prefix}_" if self.model_prefix else ""
        
        direction_path = os.path.join(self.model_dir, f"{prefix}dual_direction_lgb.pkl")
        high_path = os.path.join(self.model_dir, f"{prefix}dual_high_lgb.pkl")
        low_path = os.path.join(self.model_dir, f"{prefix}dual_low_lgb.pkl")
        
        try:
            with open(direction_path, 'rb') as f:
                self.direction_model = pickle.load(f)
            with open(high_path, 'rb') as f:
                self.high_model = pickle.load(f)
            with open(low_path, 'rb') as f:
                self.low_model = pickle.load(f)
            
            print(f"✅ 成功載入LightGBM模型: {self.model_prefix}")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"找不到模型檔案: {e}\n"
                f"請先訓練模型: {self.model_prefix}"
            )
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        # 處理特徵
        df_processed = self.extractor.process(df.copy(), create_labels=False)
        
        feature_cols = self.extractor.get_feature_columns()
        X = df_processed[feature_cols].copy()
        
        # 清理無限值和NaN
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # 模型A: 預測方向 + 信心度
        direction_proba = self.direction_model.predict_proba(X)
        df_processed['up_prob'] = direction_proba[:, 1]
        df_processed['down_prob'] = direction_proba[:, 0]
        
        # 模型B: 預測價格範圍
        df_processed['pred_high_pct'] = self.high_model.predict(X)
        df_processed['pred_low_pct'] = self.low_model.predict(X)
        
        # 計算絕對價格
        df_processed['pred_high_price'] = df_processed['close'] * (1 + df_processed['pred_high_pct'] / 100)
        df_processed['pred_low_price'] = df_processed['close'] * (1 + df_processed['pred_low_pct'] / 100)
        
        # 生成信號
        df_processed['signal'] = 0
        df_processed['tp_price'] = np.nan
        df_processed['sl_price'] = np.nan
        df_processed['reward_risk'] = 0.0
        
        for idx in df_processed.index:
            up_prob = df_processed.loc[idx, 'up_prob']
            down_prob = df_processed.loc[idx, 'down_prob']
            
            current_close = df_processed.loc[idx, 'close']
            pred_high = df_processed.loc[idx, 'pred_high_price']
            pred_low = df_processed.loc[idx, 'pred_low_price']
            
            # 做多信號
            if up_prob >= self.min_confidence:
                # TP: 預測最高價的安全系數
                tp = pred_high * self.tp_safety_factor
                
                # SL: 預測最低價 + 緩衝
                sl = pred_low * (1 + self.sl_cushion)
                
                # 確保SL < entry < TP
                if sl < current_close < tp:
                    potential_profit = tp - current_close
                    potential_loss = current_close - sl
                    
                    if potential_loss > 0:
                        rr = potential_profit / potential_loss
                        
                        if rr >= self.min_reward_risk:
                            df_processed.loc[idx, 'signal'] = 1
                            df_processed.loc[idx, 'tp_price'] = tp
                            df_processed.loc[idx, 'sl_price'] = sl
                            df_processed.loc[idx, 'reward_risk'] = rr
            
            # 做空信號
            elif down_prob >= self.min_confidence:
                # TP: 預測最低價的安全系數
                tp = pred_low * (2 - self.tp_safety_factor)
                
                # SL: 預測最高價 - 緩衝
                sl = pred_high * (1 - self.sl_cushion)
                
                # 確保TP < entry < SL
                if tp < current_close < sl:
                    potential_profit = current_close - tp
                    potential_loss = sl - current_close
                    
                    if potential_loss > 0:
                        rr = potential_profit / potential_loss
                        
                        if rr >= self.min_reward_risk:
                            df_processed.loc[idx, 'signal'] = -1
                            df_processed.loc[idx, 'tp_price'] = tp
                            df_processed.loc[idx, 'sl_price'] = sl
                            df_processed.loc[idx, 'reward_risk'] = rr
        
        return df_processed
    
    def get_signal_summary(self, df: pd.DataFrame) -> Dict:
        signals = df[df['signal'] != 0]
        
        return {
            'total_signals': len(signals),
            'long_signals': len(signals[signals['signal'] == 1]),
            'short_signals': len(signals[signals['signal'] == -1]),
            'avg_confidence': signals[['up_prob', 'down_prob']].max(axis=1).mean() if len(signals) > 0 else 0,
            'avg_reward_risk': signals['reward_risk'].mean() if len(signals) > 0 else 0,
            'min_reward_risk': signals['reward_risk'].min() if len(signals) > 0 else 0,
            'max_reward_risk': signals['reward_risk'].max() if len(signals) > 0 else 0
        }


if __name__ == '__main__':
    print("雙模型信號生成器 - LightGBM版")
    print("測試需要先訓練模型")