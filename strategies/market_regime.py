"""
Market Regime Detector - 市場狀態識別器

功能:
識別當前市場狀態，為上層策略提供決策依據

輸出 4 種狀態:
1. BULLISH_TREND - 上升趨勢 (只做多)
2. BEARISH_TREND - 下降趨勢 (只做空)
3. RANGE_BOUND - 震盪整理 (網格策略)
4. HIGH_VOLATILITY - 高波動 (降低倉位或觀望)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class MarketRegimeDetector:
    """
    市場狀態識別器
    """
    
    REGIMES = {
        0: 'BULLISH_TREND',
        1: 'BEARISH_TREND', 
        2: 'RANGE_BOUND',
        3: 'HIGH_VOLATILITY'
    }
    
    def __init__(self):
        self.model = None
        self.trained = False
        if XGBOOST_AVAILABLE:
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
    
    def calculate_features(self, df_15m: pd.DataFrame, df_1h: pd.DataFrame, df_1d: pd.DataFrame) -> pd.DataFrame:
        """
        計算多時間框架特徵
        """
        features = pd.DataFrame(index=df_1h.index)
        
        # === 1h 中期特徵 ===
        # EMA 排列
        df_1h['ema8'] = df_1h['close'].ewm(span=8).mean()
        df_1h['ema20'] = df_1h['close'].ewm(span=20).mean()
        df_1h['ema50'] = df_1h['close'].ewm(span=50).mean()
        
        features['ema_alignment'] = 0
        # 多頭排列: EMA8 > EMA20 > EMA50
        features.loc[(df_1h['ema8'] > df_1h['ema20']) & (df_1h['ema20'] > df_1h['ema50']), 'ema_alignment'] = 1
        # 空頭排列: EMA8 < EMA20 < EMA50
        features.loc[(df_1h['ema8'] < df_1h['ema20']) & (df_1h['ema20'] < df_1h['ema50']), 'ema_alignment'] = -1
        
        # ADX (趨勢強度)
        df_1h = self._calculate_adx(df_1h)
        features['adx'] = df_1h['adx']
        
        # ATR (波動度)
        df_1h['tr'] = np.maximum(
            df_1h['high'] - df_1h['low'],
            np.maximum(
                abs(df_1h['high'] - df_1h['close'].shift(1)),
                abs(df_1h['low'] - df_1h['close'].shift(1))
            )
        )
        df_1h['atr'] = df_1h['tr'].rolling(14).mean()
        df_1h['atr_pct'] = (df_1h['atr'] / df_1h['close']) * 100
        features['atr_pct'] = df_1h['atr_pct']
        
        # 價格 vs 均線
        features['price_vs_ema20'] = (df_1h['close'] - df_1h['ema20']) / df_1h['ema20'] * 100
        
        # ROC (動量)
        features['roc_10'] = df_1h['close'].pct_change(10) * 100
        
        # === 1d 長期特徵 ===
        # 週線趨勢
        df_1d['ema20_daily'] = df_1d['close'].ewm(span=20).mean()
        features['daily_trend'] = (df_1d['close'] - df_1d['ema20_daily']) / df_1d['ema20_daily'] * 100
        
        # 高低點突破
        df_1d['high_20'] = df_1d['high'].rolling(20).max()
        df_1d['low_20'] = df_1d['low'].rolling(20).min()
        features['breakout_high'] = (df_1d['close'] >= df_1d['high_20'].shift(1)).astype(int)
        features['breakout_low'] = (df_1d['close'] <= df_1d['low_20'].shift(1)).astype(int)
        
        # === 15m 短期特徵 ===
        # RSI
        df_15m = self._calculate_rsi(df_15m)
        features['rsi_15m'] = df_15m['rsi']
        
        # MACD
        ema12 = df_15m['close'].ewm(span=12).mean()
        ema26 = df_15m['close'].ewm(span=26).mean()
        df_15m['macd'] = ema12 - ema26
        df_15m['macd_signal'] = df_15m['macd'].ewm(span=9).mean()
        df_15m['macd_hist'] = df_15m['macd'] - df_15m['macd_signal']
        features['macd_hist_15m'] = df_15m['macd_hist']
        
        # 成交量趨勢
        df_1h['volume_ma20'] = df_1h['volume'].rolling(20).mean()
        features['volume_ratio'] = df_1h['volume'] / df_1h['volume_ma20']
        
        features.fillna(0, inplace=True)
        return features
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """計算 ADX"""
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        tr = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        df['adx'] = dx.rolling(period).mean()
        
        return df
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """計算 RSI"""
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        df['rsi'] = 100 - (100 / (1 + rs))
        return df
    
    def label_regimes(self, features: pd.DataFrame) -> pd.Series:
        """
        根據特徵標註市場狀態 (用於訓練)
        """
        labels = pd.Series(2, index=features.index)  # 預設震盪
        
        # 規則 1: 上升趨勢
        bullish = (
            (features['ema_alignment'] == 1) &
            (features['adx'] > 25) &
            (features['daily_trend'] > 0) &
            (features['roc_10'] > 0)
        )
        labels[bullish] = 0
        
        # 規則 2: 下降趨勢
        bearish = (
            (features['ema_alignment'] == -1) &
            (features['adx'] > 25) &
            (features['daily_trend'] < 0) &
            (features['roc_10'] < 0)
        )
        labels[bearish] = 1
        
        # 規則 3: 高波動
        high_vol = (features['atr_pct'] > 5.0)
        labels[high_vol] = 3
        
        return labels
    
    def train(self, features: pd.DataFrame, labels: pd.Series):
        """
        訓練市場狀態識別模型
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost 未安裝")
        
        # 移除 NaN
        mask = ~(features.isna().any(axis=1) | labels.isna())
        X = features[mask]
        y = labels[mask]
        
        self.model.fit(X, y)
        self.trained = True
        return self.model
    
    def predict(self, features: pd.DataFrame) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
        """
        預測市場狀態
        返回: (狀態標籤, 機率分佈)
        """
        if self.model is None or not self.trained:
            # 如果沒有訓練模型，使用規則
            regime_codes = self.label_regimes(features)
            regime_names = regime_codes.map(self.REGIMES)
            return regime_names, None
        
        predictions = self.model.predict(features)
        probas = self.model.predict_proba(features)
        
        # 處理 XGBoost 可能輸出的類別數不匹配問題
        regime_names = pd.Series([self.REGIMES.get(p, 'RANGE_BOUND') for p in predictions], index=features.index)
        
        # 建立機率 DataFrame，填充缺失的類別
        n_classes = probas.shape[1]
        proba_dict = {}
        
        # 獲取 XGBoost 實際輸出的類別
        trained_classes = self.model.classes_
        
        # 為所有 4 種狀態建立機率欄位
        for i in range(4):
            regime_name = self.REGIMES[i]
            if i in trained_classes:
                # 找到對應的機率欄位
                class_idx = np.where(trained_classes == i)[0][0]
                proba_dict[regime_name] = probas[:, class_idx]
            else:
                # 缺失的類別填充0
                proba_dict[regime_name] = np.zeros(len(probas))
        
        proba_df = pd.DataFrame(proba_dict, index=features.index)
        
        return regime_names, proba_df
    
    def get_regime_name(self, regime_code: int) -> str:
        """獲取狀態名稱"""
        return self.REGIMES.get(regime_code, 'UNKNOWN')
    
    def get_regime_description(self, regime_name: str) -> dict:
        """
        獲取狀態描述與建議策略
        """
        descriptions = {
            'BULLISH_TREND': {
                'name': '上升趨勢',
                'emoji': '📈',
                'strategy': '只做多',
                'entry': 'EMA20 回調 + RSI<40',
                'tp': 'ATR * 3',
                'sl': 'ATR * 1.5'
            },
            'BEARISH_TREND': {
                'name': '下降趨勢',
                'emoji': '📉',
                'strategy': '只做空',
                'entry': 'EMA20 反彈 + RSI>60',
                'tp': 'ATR * 3',
                'sl': 'ATR * 1.5'
            },
            'RANGE_BOUND': {
                'name': '震盪整理',
                'emoji': '➡️',
                'strategy': '網格策略',
                'entry': 'BB上軌做空 / BB下軌做多',
                'tp': 'ATR * 1.5',
                'sl': 'ATR * 1.0'
            },
            'HIGH_VOLATILITY': {
                'name': '高波動',
                'emoji': '⚠️',
                'strategy': '降低倉位或觀望',
                'entry': '謹慎開倉',
                'tp': 'ATR * 2',
                'sl': 'ATR * 1.0'
            }
        }
        return descriptions.get(regime_name, {})
