import pandas as pd
import numpy as np
import ta
from utils.bb_reversal_detector import BBReversalDetector

class BBReversalFeatureExtractor:
    """
    BB反轉點特徵提取器
    
    基於有效BB反轉點訓練模型
    """
    
    def __init__(self,
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 rsi_period: int = 14,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9):
        
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        
        self.detector = BBReversalDetector(
            bb_period=bb_period,
            bb_std=bb_std,
            touch_threshold=0.001,
            reversal_confirm_candles=5,
            min_reversal_pct=0.005,
            trend_filter_enabled=True,
            trend_lookback=10,
            require_middle_return=True
        )
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # BB指標 (已在detector中計算)
        bb = ta.volatility.BollingerBands(
            df['close'], 
            window=self.bb_period, 
            window_dev=self.bb_std
        )
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # 價格與BB軌道的距離
        df['dist_to_upper_pct'] = (df['bb_upper'] - df['close']) / df['close'] * 100
        df['dist_to_middle_pct'] = (df['bb_middle'] - df['close']) / df['close'] * 100
        df['dist_to_lower_pct'] = (df['close'] - df['bb_lower']) / df['close'] * 100
        
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=self.rsi_period).rsi()
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        
        # MACD
        macd = ta.trend.MACD(
            df['close'],
            window_fast=self.macd_fast,
            window_slow=self.macd_slow,
            window_sign=self.macd_signal
        )
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        df['macd_cross_up'] = ((df['macd'] > df['macd_signal']) & 
                               (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
        df['macd_cross_down'] = ((df['macd'] < df['macd_signal']) & 
                                 (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)
        
        # 成交量
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['high_volume'] = (df['volume_ratio'] > 1.5).astype(int)
        
        # 波動率 (ATR)
        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14)
        df['atr'] = atr.average_true_range()
        df['atr_pct'] = df['atr'] / df['close'] * 100
        
        # K棒形態
        df['body_size'] = abs(df['close'] - df['open']) / df['open'] * 100
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open'] * 100
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open'] * 100
        df['is_green'] = (df['close'] > df['open']).astype(int)
        df['is_red'] = (df['close'] < df['open']).astype(int)
        
        # 連續漨跌
        df['consecutive_green'] = 0
        df['consecutive_red'] = 0
        for i in range(1, len(df)):
            if df.iloc[i]['is_green']:
                df.iloc[i, df.columns.get_loc('consecutive_green')] = df.iloc[i-1]['consecutive_green'] + 1
            if df.iloc[i]['is_red']:
                df.iloc[i, df.columns.get_loc('consecutive_red')] = df.iloc[i-1]['consecutive_red'] + 1
        
        # 價格動量
        df['price_change'] = df['close'].pct_change() * 100
        df['price_change_2'] = df['close'].pct_change(2) * 100
        df['price_change_3'] = df['close'].pct_change(3) * 100
        
        # EMA
        df['ema_5'] = ta.trend.EMAIndicator(df['close'], window=5).ema_indicator()
        df['ema_10'] = ta.trend.EMAIndicator(df['close'], window=10).ema_indicator()
        df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
        df['price_above_ema5'] = (df['close'] > df['ema_5']).astype(int)
        df['price_above_ema20'] = (df['close'] > df['ema_20']).astype(int)
        
        return df
    
    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用BB反轉檢測器創建標籤
        """
        df = df.copy()
        
        # 使用detector檢測反轉點
        df_detected = self.detector.detect_reversals(df)
        
        # 初始化標籤
        df_detected['signal'] = 0  # 0: 無信號, 1: 做多, -1: 做空
        df_detected['target_type'] = ''  # 'upper_reversal' 或 'lower_reversal'
        
        # 根據有效反轉點創建標籤
        for reversal in self.detector.reversals:
            idx = reversal['index']
            reversal_type = reversal['type']
            
            if reversal_type == 'upper':
                # 上軌反轉 -> 做空
                df_detected.iloc[idx, df_detected.columns.get_loc('signal')] = -1
                df_detected.iloc[idx, df_detected.columns.get_loc('target_type')] = 'upper_reversal'
            
            elif reversal_type == 'lower':
                # 下軌反轉 -> 做多
                df_detected.iloc[idx, df_detected.columns.get_loc('signal')] = 1
                df_detected.iloc[idx, df_detected.columns.get_loc('target_type')] = 'lower_reversal'
        
        return df_detected
    
    def process(self, df: pd.DataFrame, create_labels: bool = True) -> pd.DataFrame:
        """
        處理數據: 計算特徵 + 創建標籤
        """
        df = df.copy()
        
        # 計算技術指標
        df = self.calculate_technical_indicators(df)
        
        # 創建標籤
        if create_labels:
            df = self.create_labels(df)
        
        # 移除NaN
        df = df.dropna()
        
        return df
    
    def get_feature_columns(self) -> list:
        """
        獲取特徵列名稱
        """
        feature_cols = [
            # BB特徵
            'bb_width', 'bb_position',
            'dist_to_upper_pct', 'dist_to_middle_pct', 'dist_to_lower_pct',
            
            # RSI
            'rsi', 'rsi_overbought', 'rsi_oversold',
            
            # MACD
            'macd', 'macd_signal', 'macd_diff',
            'macd_cross_up', 'macd_cross_down',
            
            # 成交量
            'volume_ratio', 'high_volume',
            
            # 波動率
            'atr_pct',
            
            # K棒形態
            'body_size', 'upper_shadow', 'lower_shadow',
            'is_green', 'is_red',
            'consecutive_green', 'consecutive_red',
            
            # 價格動量
            'price_change', 'price_change_2', 'price_change_3',
            
            # EMA
            'price_above_ema5', 'price_above_ema20'
        ]
        
        return feature_cols
    
    def get_training_data(self, df: pd.DataFrame):
        """
        獲取訓練數據
        """
        feature_cols = self.get_feature_columns()
        
        # 只使用有信號的樣本 (不包括0)
        df_signals = df[df['signal'] != 0].copy()
        
        X = df_signals[feature_cols]
        y = df_signals['signal']
        
        # 轉換為二元分類: -1 -> 0, 1 -> 1
        y_binary = (y == 1).astype(int)
        
        return X, y_binary
    
    def get_reversal_statistics(self) -> dict:
        """
        獲取反轉點統計
        """
        if not hasattr(self.detector, 'reversals'):
            return {}
        
        stats = {
            'total_reversals': len(self.detector.reversals),
            'upper_reversals': len([r for r in self.detector.reversals if r['type'] == 'upper']),
            'lower_reversals': len([r for r in self.detector.reversals if r['type'] == 'lower']),
            'total_rejected': len(self.detector.rejected_touches)
        }
        
        # 拒絕原因統計
        rejection_reasons = {}
        for touch in self.detector.rejected_touches:
            reason = touch['reason']
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
        stats['rejection_reasons'] = rejection_reasons
        
        return stats


if __name__ == '__main__':
    print("BB反轉特徵提取器測試")
    print("="*60)
    
    # 需要實際數據測試
    print("請在App中使用")