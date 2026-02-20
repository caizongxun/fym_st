"""
Signal Generator - 信號生成器

功能:
根據市場狀態生成交易信號
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


class SignalGenerator:
    """
    信號生成器 - 根據市場狀態採用不同策略
    """
    
    def __init__(self):
        self.current_regime = None
    
    def generate_signals(
        self, 
        df: pd.DataFrame, 
        regime: str,
        regime_proba: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        生成交易信號
        
        Args:
            df: 1h K棒數據 (包含指標)
            regime: 市場狀態
            regime_proba: 狀態機率 (可選)
        
        Returns:
            包含信號欄位的 DataFrame
        """
        df = df.copy()
        self.current_regime = regime
        
        # 計算基礎指標
        df = self._calculate_indicators(df)
        
        # 根據狀態生成信號
        if regime == 'BULLISH_TREND':
            df = self._bullish_signals(df)
        elif regime == 'BEARISH_TREND':
            df = self._bearish_signals(df)
        elif regime == 'RANGE_BOUND':
            df = self._range_signals(df)
        else:  # HIGH_VOLATILITY
            df['signal'] = 0  # 觀望
            df['signal_strength'] = 0
        
        return df
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算技術指標"""
        # EMA
        df['ema8'] = df['close'].ewm(span=8).mean()
        df['ema20'] = df['close'].ewm(span=20).mean()
        df['ema50'] = df['close'].ewm(span=50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * bb_std
        df['bb_lower'] = df['bb_mid'] - 2 * bb_std
        df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        
        # ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(14).mean()
        
        # Volume
        df['volume_ma20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma20'] + 1e-8)
        
        return df
    
    def _bullish_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        上升趨勢信號 - 只做多
        
        進場條件:
        1. 價格回調到 EMA20 附近
        2. RSI < 40 (超賣)
        3. MACD 金叉
        4. 成交量放大
        """
        df['signal'] = 0
        df['signal_strength'] = 0.0
        
        # 條件 1: 價格接近 EMA20 (+-2%)
        price_near_ema20 = abs(df['close'] - df['ema20']) / df['ema20'] < 0.02
        
        # 條件 2: RSI 超賣
        rsi_oversold = df['rsi'] < 40
        
        # 條件 3: MACD 金叉
        macd_cross = (df['macd_hist'] > 0) & (df['macd_hist'].shift(1) <= 0)
        
        # 條件 4: 成交量確認
        volume_confirm = df['volume_ratio'] > 1.2
        
        # 組合信號
        long_signal = price_near_ema20 & rsi_oversold
        df.loc[long_signal, 'signal'] = 1
        
        # 計算信號強度
        strength = 0.0
        strength += price_near_ema20.astype(float) * 0.25
        strength += rsi_oversold.astype(float) * 0.25
        strength += macd_cross.astype(float) * 0.3
        strength += volume_confirm.astype(float) * 0.2
        df['signal_strength'] = strength
        
        return df
    
    def _bearish_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        下降趨勢信號 - 只做空
        
        進場條件:
        1. 價格反彈到 EMA20 附近
        2. RSI > 60 (超買)
        3. MACD 死叉
        4. 成交量萎縮
        """
        df['signal'] = 0
        df['signal_strength'] = 0.0
        
        # 條件 1: 價格接近 EMA20
        price_near_ema20 = abs(df['close'] - df['ema20']) / df['ema20'] < 0.02
        
        # 條件 2: RSI 超買
        rsi_overbought = df['rsi'] > 60
        
        # 條件 3: MACD 死叉
        macd_cross = (df['macd_hist'] < 0) & (df['macd_hist'].shift(1) >= 0)
        
        # 條件 4: 成交量萎縮
        volume_weak = df['volume_ratio'] < 0.8
        
        # 組合信號
        short_signal = price_near_ema20 & rsi_overbought
        df.loc[short_signal, 'signal'] = -1
        
        # 計算信號強度
        strength = 0.0
        strength += price_near_ema20.astype(float) * 0.25
        strength += rsi_overbought.astype(float) * 0.25
        strength += macd_cross.astype(float) * 0.3
        strength += volume_weak.astype(float) * 0.2
        df['signal_strength'] = strength
        
        return df
    
    def _range_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        震盪整理信號 - 網格策略
        
        進場條件:
        - BB上軌附近做空
        - BB下軌附近做多
        """
        df['signal'] = 0
        df['signal_strength'] = 0.0
        
        # BB 上軌做空
        at_upper = df['bb_pct'] > 0.9
        df.loc[at_upper, 'signal'] = -1
        df.loc[at_upper, 'signal_strength'] = 0.8
        
        # BB 下軌做多
        at_lower = df['bb_pct'] < 0.1
        df.loc[at_lower, 'signal'] = 1
        df.loc[at_lower, 'signal_strength'] = 0.8
        
        return df
    
    def calculate_exit_levels(self, df: pd.DataFrame, regime: str) -> Dict[str, pd.Series]:
        """
        計算出場位置 (TP/SL)
        
        Returns:
            {'tp_long', 'sl_long', 'tp_short', 'sl_short'}
        """
        atr = df['atr']
        close = df['close']
        
        if regime == 'BULLISH_TREND' or regime == 'BEARISH_TREND':
            # 趨勢市: 大目標小止損
            tp_multiplier = 3.0
            sl_multiplier = 1.5
        elif regime == 'RANGE_BOUND':
            # 震盪市: 小目標小止損
            tp_multiplier = 1.5
            sl_multiplier = 1.0
        else:  # HIGH_VOLATILITY
            # 高波動: 中等目標小止損
            tp_multiplier = 2.0
            sl_multiplier = 1.0
        
        return {
            'tp_long': close + atr * tp_multiplier,
            'sl_long': close - atr * sl_multiplier,
            'tp_short': close - atr * tp_multiplier,
            'sl_short': close + atr * sl_multiplier
        }
    
    def filter_signals(
        self, 
        df: pd.DataFrame, 
        min_strength: float = 0.5,
        multi_timeframe_confirm: bool = True
    ) -> pd.DataFrame:
        """
        過濾低質量信號
        
        Args:
            min_strength: 最小信號強度
            multi_timeframe_confirm: 是否需要多時間框架確認
        """
        df = df.copy()
        
        # 過濾 1: 信號強度
        weak_signals = df['signal_strength'] < min_strength
        df.loc[weak_signals, 'signal'] = 0
        
        # 過濾 2: 多時間框架確認 (如果有 15m 數據)
        if multi_timeframe_confirm and 'rsi_15m' in df.columns:
            # 做多時，15m RSI 也要超賣
            long_signals = df['signal'] == 1
            df.loc[long_signals & (df['rsi_15m'] > 50), 'signal'] = 0
            
            # 做空時，15m RSI 也要超買
            short_signals = df['signal'] == -1
            df.loc[short_signals & (df['rsi_15m'] < 50), 'signal'] = 0
        
        return df
    
    def get_signal_summary(self, df: pd.DataFrame) -> Dict:
        """
        獲取信號統計
        """
        return {
            'total_signals': (df['signal'] != 0).sum(),
            'long_signals': (df['signal'] == 1).sum(),
            'short_signals': (df['signal'] == -1).sum(),
            'avg_strength': df[df['signal'] != 0]['signal_strength'].mean() if (df['signal'] != 0).any() else 0
        }
