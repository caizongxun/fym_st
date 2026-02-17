import pandas as pd
import numpy as np
from typing import Dict, Tuple
import os

class DualModelSignalGenerator:
    """
    雙模型信號生成器
    
    策略邏輯:
    1. 模型A預測方向 (漲/跌)
    2. 模型B預測最高/最低價
    3. 如果預測漲 -> 下一根open做多
       - 止盈: 預測最高價 * safety_factor (0.85-0.95)
       - 止損: 預測最低價 * (1 + cushion)
    4. 如果預測跌 -> 下一根open做空
       - 止盈: 預測最低價 * safety_factor
       - 止損: 預測最高價 * (1 + cushion)
    
    優勢:
    - 動態止盈止損 (不是ATR固定倍數)
    - 風報比自適應
    - 高信心度信號節流
    """
    
    def __init__(self,
                 model_dir: str = 'models/saved',
                 model_prefix: str = '',
                 min_confidence: float = 0.55,
                 tp_safety_factor: float = 0.90,
                 sl_cushion: float = 0.05,
                 min_reward_risk: float = 1.2):
        """
        Args:
            model_dir: 模型目錄
            model_prefix: 模型前綴 (幣種)
            min_confidence: 最低信心度 (0.5-1.0)
            tp_safety_factor: 止盈安全系數 (0.85-0.95)
            sl_cushion: 止損緩衝 (0.02-0.1)
            min_reward_risk: 最低風報比
        """
        from models.train_dual_model import DualModelTrainer
        from utils.dual_model_features import DualModelFeatureExtractor
        
        self.model_dir = model_dir
        self.model_prefix = model_prefix
        self.min_confidence = min_confidence
        self.tp_safety_factor = tp_safety_factor
        self.sl_cushion = sl_cushion
        self.min_reward_risk = min_reward_risk
        
        # 載入模型
        self.trainer = DualModelTrainer(model_dir=model_dir)
        success = self.trainer.load_models(prefix=model_prefix)
        
        if not success:
            raise FileNotFoundError(
                f"無法載入模型: {model_prefix}\n"
                f"請先訓練模型!"
            )
        
        # 特徵提取器
        self.feature_extractor = DualModelFeatureExtractor()
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信號
        
        Returns:
            DataFrame with columns:
            - signal: 1=做多, -1=做空, 0=無信號
            - direction_pred: 方向預測
            - direction_proba: 方向信心度
            - high_pred: 預測最高價
            - low_pred: 預測最低價
            - tp_price: 止盈價
            - sl_price: 止損價
            - reward_risk: 風報比
        """
        # 處理特徵
        df_processed = self.feature_extractor.process(df, create_labels=False)
        
        # 獲取特徵
        feature_cols = self.feature_extractor.get_feature_columns()
        X = df_processed[feature_cols]
        
        # 預測
        predictions = self.trainer.predict(X)
        
        # 添加預測結果
        df_processed['direction_pred'] = predictions['direction']
        df_processed['direction_proba'] = predictions['direction_proba']
        df_processed['high_pred_pct'] = predictions['high_pct']
        df_processed['low_pred_pct'] = predictions['low_pct']
        
        # 計算預測價格 (絕對值)
        df_processed['high_pred'] = df_processed['close'] * (1 + df_processed['high_pred_pct'] / 100)
        df_processed['low_pred'] = df_processed['close'] * (1 + df_processed['low_pred_pct'] / 100)
        
        # 初始化信號和止盈止損
        df_processed['signal'] = 0
        df_processed['tp_price'] = 0.0
        df_processed['sl_price'] = 0.0
        df_processed['reward_risk'] = 0.0
        df_processed['entry_price'] = 0.0
        
        # 下一根open作為進場價
        df_processed['next_open'] = df_processed['open'].shift(-1)
        
        # 築選高信心度信號
        for idx in df_processed.index:
            if pd.isna(df_processed.loc[idx, 'next_open']):
                continue
            
            direction = df_processed.loc[idx, 'direction_pred']
            confidence = df_processed.loc[idx, 'direction_proba']
            
            # 信心度築選
            if direction == 1:  # 預測漸
                if confidence < self.min_confidence:
                    continue
            else:  # 預測跌
                if (1 - confidence) < self.min_confidence:
                    continue
            
            entry = df_processed.loc[idx, 'next_open']
            high_pred = df_processed.loc[idx, 'high_pred']
            low_pred = df_processed.loc[idx, 'low_pred']
            
            if direction == 1:  # 做多
                # 止盈: 預測最高價的 90%
                tp = high_pred * self.tp_safety_factor
                # 止損: 預測最低價再往下一點
                sl = low_pred * (1 - self.sl_cushion)
                
                # 確保止盈大於進場價
                if tp <= entry:
                    continue
                
                # 計算風報比
                potential_profit = tp - entry
                potential_loss = entry - sl
                
                if potential_loss <= 0:
                    continue
                
                rr = potential_profit / potential_loss
                
                # 風報比築選
                if rr < self.min_reward_risk:
                    continue
                
                df_processed.loc[idx, 'signal'] = 1
                df_processed.loc[idx, 'tp_price'] = tp
                df_processed.loc[idx, 'sl_price'] = sl
                df_processed.loc[idx, 'reward_risk'] = rr
                df_processed.loc[idx, 'entry_price'] = entry
                
            else:  # 做空
                # 止盈: 預測最低價的 90%
                tp = low_pred * (2 - self.tp_safety_factor)  # 等同於 low * 1.1 當 safety=0.9
                # 止損: 預測最高價再往上一點
                sl = high_pred * (1 + self.sl_cushion)
                
                # 確保止盈小於進場價
                if tp >= entry:
                    continue
                
                # 計算風報比
                potential_profit = entry - tp
                potential_loss = sl - entry
                
                if potential_loss <= 0:
                    continue
                
                rr = potential_profit / potential_loss
                
                # 風報比築選
                if rr < self.min_reward_risk:
                    continue
                
                df_processed.loc[idx, 'signal'] = -1
                df_processed.loc[idx, 'tp_price'] = tp
                df_processed.loc[idx, 'sl_price'] = sl
                df_processed.loc[idx, 'reward_risk'] = rr
                df_processed.loc[idx, 'entry_price'] = entry
        
        return df_processed
    
    def get_signal_summary(self, df: pd.DataFrame) -> Dict:
        """獲取信號統計"""
        signals = df[df['signal'] != 0]
        
        if len(signals) == 0:
            return {
                'total_signals': 0,
                'long_signals': 0,
                'short_signals': 0,
                'avg_confidence': 0,
                'avg_reward_risk': 0,
                'signal_frequency': 0
            }
        
        long_signals = signals[signals['signal'] == 1]
        short_signals = signals[signals['signal'] == -1]
        
        return {
            'total_signals': len(signals),
            'long_signals': len(long_signals),
            'short_signals': len(short_signals),
            'avg_confidence': signals['direction_proba'].mean(),
            'avg_reward_risk': signals['reward_risk'].mean(),
            'signal_frequency': len(signals) / len(df) * 100,
            'avg_long_rr': long_signals['reward_risk'].mean() if len(long_signals) > 0 else 0,
            'avg_short_rr': short_signals['reward_risk'].mean() if len(short_signals) > 0 else 0
        }


if __name__ == '__main__':
    from utils.dual_model_features import DualModelFeatureExtractor
    from models.train_dual_model import DualModelTrainer
    
    print("雙模型信號生成器測試")
    print("="*60)
    
    # 1. 生成測試數據
    dates = pd.date_range('2024-01-01', periods=2000, freq='15min')
    np.random.seed(42)
    
    base_price = 50000
    prices = base_price + np.random.randn(2000).cumsum() * 100
    
    df = pd.DataFrame({
        'open_time': dates,
        'open': prices,
        'high': prices + np.random.rand(2000) * 50,
        'low': prices - np.random.rand(2000) * 50,
        'close': prices + np.random.randn(2000) * 20,
        'volume': np.random.randint(1000, 5000, 2000)
    })
    
    # 2. 訓練模型
    print("步驟1: 訓練模型...")
    extractor = DualModelFeatureExtractor()
    df_processed = extractor.process(df, create_labels=True)
    X, y_dict = extractor.get_training_data(df_processed)
    
    trainer = DualModelTrainer()
    trainer.train_all_models(X, y_dict)
    trainer.save_models(prefix='test')
    
    # 3. 生成信號
    print("\n步驟2: 生成交易信號...")
    signal_gen = DualModelSignalGenerator(
        model_prefix='test',
        min_confidence=0.55,
        tp_safety_factor=0.90,
        min_reward_risk=1.2
    )
    
    df_signals = signal_gen.generate_signals(df)
    
    # 4. 顯示統計
    summary = signal_gen.get_signal_summary(df_signals)
    print("\n信號統計:")
    print(f"  總信號數: {summary['total_signals']}")
    print(f"  做多: {summary['long_signals']}, 做空: {summary['short_signals']}")
    print(f"  平均信心度: {summary['avg_confidence']:.3f}")
    print(f"  平均風報比: {summary['avg_reward_risk']:.2f}")
    print(f"  信號頻率: {summary['signal_frequency']:.2f}%")
    
    # 5. 顯示第一個信號
    if summary['total_signals'] > 0:
        first_signal = df_signals[df_signals['signal'] != 0].iloc[0]
        print(f"\n第一個信號:")
        print(f"  方向: {'LONG' if first_signal['signal'] == 1 else 'SHORT'}")
        print(f"  進場: {first_signal['entry_price']:.2f}")
        print(f"  止盈: {first_signal['tp_price']:.2f}")
        print(f"  止損: {first_signal['sl_price']:.2f}")
        print(f"  風報比: {first_signal['reward_risk']:.2f}")