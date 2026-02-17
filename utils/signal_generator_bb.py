import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, Optional

from utils.bb_bounce_features import BBBounceFeatureExtractor

class BBBounceSignalGenerator:
    """
    BB反彈信號生成器
    
    整合:
    1. BB反彈預測模型
    2. 原有反轉模型(雙重確認)
    3. ADX趨勢過濾器
    
    策略邏輯:
    - 做空: 觸碰上軌 + BB模型預測反彈 + 非強多頭 + 反轉確認 + RSI超買
    - 做多: 觸碰下軌 + BB模型預測反彈 + 非強空頭 + 反轉確認 + RSI超賣
    """
    
    def __init__(self,
                 bb_model_dir: str = 'models/saved',
                 reversal_model_path: Optional[str] = None,
                 bb_bounce_threshold: float = 0.60,
                 reversal_threshold: float = 0.50,
                 adx_strong_trend_threshold: float = 30):
        """
        Args:
            bb_model_dir: BB模型目錄
            reversal_model_path: 反轉模型路徑(可選)
            bb_bounce_threshold: BB反彈機率門檻
            reversal_threshold: 反轉機率門檻
            adx_strong_trend_threshold: 強趨勢ADX門檻
        """
        self.feature_extractor = BBBounceFeatureExtractor()
        self.bb_bounce_threshold = bb_bounce_threshold
        self.reversal_threshold = reversal_threshold
        self.adx_threshold = adx_strong_trend_threshold
        
        # 載入BB模型
        self.upper_model = self._load_model(
            os.path.join(bb_model_dir, 'bb_upper_bounce_model.pkl')
        )
        self.lower_model = self._load_model(
            os.path.join(bb_model_dir, 'bb_lower_bounce_model.pkl')
        )
        
        # 載入反轉模型(可選)
        self.reversal_model = None
        if reversal_model_path and os.path.exists(reversal_model_path):
            self.reversal_model = self._load_model(reversal_model_path)
    
    def _load_model(self, path: str):
        """載入模型"""
        if os.path.exists(path):
            return joblib.load(path)
        return None
    
    def generate_signals(self, df: pd.DataFrame, 
                        reversal_predictions: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        生成交易信號
        
        Args:
            df: 原始K線數據
            reversal_predictions: 反轉模型預測(可選)
        
        Returns:
            帶有信號的DataFrame
        """
        # 1. 處理特徵
        df = self.feature_extractor.process(df, create_labels=False)
        
        # 2. 初始化信號
        df['signal'] = 0
        df['bb_upper_bounce_prob'] = 0.0
        df['bb_lower_bounce_prob'] = 0.0
        df['signal_reason'] = ''
        
        # 3. 預測上軌反彈
        if self.upper_model is not None:
            upper_mask = df['touch_upper'] == 1
            if upper_mask.sum() > 0:
                X_upper = df.loc[upper_mask, self.feature_extractor.get_feature_columns()]
                upper_probs = self.upper_model.predict_proba(X_upper)[:, 1]
                df.loc[upper_mask, 'bb_upper_bounce_prob'] = upper_probs
        
        # 4. 預測下軌反彈
        if self.lower_model is not None:
            lower_mask = df['touch_lower'] == 1
            if lower_mask.sum() > 0:
                X_lower = df.loc[lower_mask, self.feature_extractor.get_feature_columns()]
                lower_probs = self.lower_model.predict_proba(X_lower)[:, 1]
                df.loc[lower_mask, 'bb_lower_bounce_prob'] = lower_probs
        
        # 5. 生成SHORT信號(觸碰上軌)
        short_conditions = [
            # BB觸碰和反彈預測
            (df['touch_upper'] == 1) | (df['pierce_upper'] == 1),
            df['bb_upper_bounce_prob'] >= self.bb_bounce_threshold,
            
            # 趨勢過濾: 非強多頭
            df['trend_state'] != 'strong_uptrend',
            
            # ADX過濾: ADX < threshold OR +DI 沒有明顯大於 -DI
            (df['adx'] < self.adx_threshold) | 
            ((df['plus_di'] - df['minus_di']) < 10),
            
            # RSI超買
            df['rsi'] > 60
        ]
        
        short_signal = pd.Series(True, index=df.index)
        for condition in short_conditions:
            short_signal = short_signal & condition
        
        # 6. 生成LONG信號(觸碰下軌)
        long_conditions = [
            # BB觸碰和反彈預測
            (df['touch_lower'] == 1) | (df['pierce_lower'] == 1),
            df['bb_lower_bounce_prob'] >= self.bb_bounce_threshold,
            
            # 趨勢過濾: 非強空頭
            df['trend_state'] != 'strong_downtrend',
            
            # ADX過濾
            (df['adx'] < self.adx_threshold) | 
            ((df['minus_di'] - df['plus_di']) < 10),
            
            # RSI超賣
            df['rsi'] < 40
        ]
        
        long_signal = pd.Series(True, index=df.index)
        for condition in long_conditions:
            long_signal = long_signal & condition
        
        # 7. 反轉模型雙重確認(可選)
        if reversal_predictions is not None:
            # 反轉機率需 > threshold
            reversal_confirm = reversal_predictions >= self.reversal_threshold
            short_signal = short_signal & reversal_confirm
            long_signal = long_signal & reversal_confirm
            
            df['reversal_prob'] = reversal_predictions
        
        # 8. 賭值信號
        df.loc[short_signal, 'signal'] = -1
        df.loc[long_signal, 'signal'] = 1
        
        # 9. 記錄信號原因
        df.loc[short_signal, 'signal_reason'] = df.loc[short_signal].apply(
            lambda x: f"SHORT: BB_upper({x['bb_upper_bounce_prob']:.2f}) ADX({x['adx']:.1f}) RSI({x['rsi']:.1f})",
            axis=1
        )
        df.loc[long_signal, 'signal_reason'] = df.loc[long_signal].apply(
            lambda x: f"LONG: BB_lower({x['bb_lower_bounce_prob']:.2f}) ADX({x['adx']:.1f}) RSI({x['rsi']:.1f})",
            axis=1
        )
        
        return df
    
    def add_signal_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加信號元數據"""
        df = df.copy()
        
        signal_map = {1: 'LONG', -1: 'SHORT', 0: 'HOLD'}
        df['signal_name'] = df['signal'].map(signal_map)
        
        # 信號強度 = BB反彈機率
        df['signal_strength'] = df.apply(
            lambda x: x['bb_upper_bounce_prob'] if x['signal'] == -1 
                     else (x['bb_lower_bounce_prob'] if x['signal'] == 1 else 0),
            axis=1
        )
        
        return df
    
    def get_trade_statistics(self, df: pd.DataFrame) -> Dict:
        """
        獲取交易統計
        """
        signals = df[df['signal'] != 0]
        
        stats = {
            'total_signals': len(signals),
            'long_signals': (signals['signal'] == 1).sum(),
            'short_signals': (signals['signal'] == -1).sum(),
            'avg_bb_upper_prob': df[df['signal'] == -1]['bb_upper_bounce_prob'].mean(),
            'avg_bb_lower_prob': df[df['signal'] == 1]['bb_lower_bounce_prob'].mean(),
            'trend_state_distribution': signals['trend_state'].value_counts().to_dict(),
            'avg_adx': signals['adx'].mean(),
            'avg_rsi_short': df[df['signal'] == -1]['rsi'].mean(),
            'avg_rsi_long': df[df['signal'] == 1]['rsi'].mean(),
        }
        
        return stats
    
    def print_signal_summary(self, df: pd.DataFrame):
        """
        列印信號摘要
        """
        stats = self.get_trade_statistics(df)
        
        print("\n" + "="*70)
        print("BB反彈信號生成摘要")
        print("="*70)
        
        print(f"\n總信號數: {stats['total_signals']}")
        print(f"  做多信號: {stats['long_signals']} ({stats['long_signals']/max(stats['total_signals'],1)*100:.1f}%)")
        print(f"  做空信號: {stats['short_signals']} ({stats['short_signals']/max(stats['total_signals'],1)*100:.1f}%)")
        
        print(f"\n平均BB反彈機率:")
        print(f"  上軌(做空): {stats['avg_bb_upper_prob']:.2%}")
        print(f"  下軌(做多): {stats['avg_bb_lower_prob']:.2%}")
        
        print(f"\n趨勢狀態分佈:")
        for trend, count in stats['trend_state_distribution'].items():
            print(f"  {trend}: {count} ({count/stats['total_signals']*100:.1f}%)")
        
        print(f"\n平均ADX: {stats['avg_adx']:.2f}")
        print(f"\n平均RSI:")
        print(f"  做空時: {stats['avg_rsi_short']:.2f}")
        print(f"  做多時: {stats['avg_rsi_long']:.2f}")
        
        print("\n" + "="*70)