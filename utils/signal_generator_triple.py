import pandas as pd
import numpy as np
import ta
from typing import Dict, Optional

class TripleConfirmSignalGenerator:
    """
    BB + MACD + ADX 三重確認策略信號生成器
    
    核心邏輯 (修改版):
    做多: BB下軌 + MACD金叉 + ADX<30 + (RSI<50加分)
    做空: BB上軌 + MACD死叉 + ADX<30 + (RSI>50加分)
    
    RSI不再是必須條件,只用來計算信號強度
    """
    
    def __init__(self,
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 rsi_period: int = 14,
                 rsi_oversold: int = 30,
                 rsi_overbought: int = 70,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9,
                 adx_period: int = 14,
                 adx_threshold: int = 30,
                 use_strict_rsi: bool = False):
        
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.use_strict_rsi = use_strict_rsi
    
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加所有需要的技術指標"""
        df = df.copy()
        
        # Bollinger Bands
        bb_indicator = ta.volatility.BollingerBands(
            close=df['close'],
            window=self.bb_period,
            window_dev=self.bb_std
        )
        df['bb_upper'] = bb_indicator.bollinger_hband()
        df['bb_middle'] = bb_indicator.bollinger_mavg()
        df['bb_lower'] = bb_indicator.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(
            close=df['close'],
            window=self.rsi_period
        ).rsi()
        
        # MACD
        macd_indicator = ta.trend.MACD(
            close=df['close'],
            window_slow=self.macd_slow,
            window_fast=self.macd_fast,
            window_sign=self.macd_signal
        )
        df['macd'] = macd_indicator.macd()
        df['macd_signal'] = macd_indicator.macd_signal()
        df['macd_diff'] = macd_indicator.macd_diff()
        
        # ADX
        df['adx'] = ta.trend.ADXIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=self.adx_period
        ).adx()
        
        # 填充NaN
        df = df.ffill().bfill()
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成三重確認交易信號
        
        Returns:
            DataFrame with signals: 1=做多, -1=做空, 0=無信號
        """
        df = self.add_indicators(df)
        
        # 初始化信號
        df['signal'] = 0
        df['signal_strength'] = 0.0
        
        # 計算MACD交叉
        df['macd_cross_up'] = (
            (df['macd'] > df['macd_signal']) & 
            (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        )
        df['macd_cross_down'] = (
            (df['macd'] < df['macd_signal']) & 
            (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        )
        
        # 計算BB位置 (價格在BB中的相對位置 0-1)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_position'] = df['bb_position'].clip(0, 1)
        
        if self.use_strict_rsi:
            # 嚴格RSI模式 (原始版本)
            df['rsi_rising'] = df['rsi'] > df['rsi'].shift(1)
            df['rsi_falling'] = df['rsi'] < df['rsi'].shift(1)
            
            long_conditions = (
                (df['close'] <= df['bb_lower'] * 1.01) &
                (df['rsi'] < self.rsi_oversold) &
                (df['rsi_rising']) &
                (df['macd_cross_up']) &
                (df['adx'] < self.adx_threshold)
            )
            
            short_conditions = (
                (df['close'] >= df['bb_upper'] * 0.99) &
                (df['rsi'] > self.rsi_overbought) &
                (df['rsi_falling']) &
                (df['macd_cross_down']) &
                (df['adx'] < self.adx_threshold)
            )
        else:
            # 寬鬆模式 (預設) - RSI只用作加分,不是必須條件
            long_conditions = (
                # 1. 價格在BB下方 (BB position < 0.3)
                (df['bb_position'] < 0.3) &
                
                # 2. MACD金叉
                (df['macd_cross_up']) &
                
                # 3. ADX < 閾值 (非強趨勢)
                (df['adx'] < self.adx_threshold)
            )
            
            short_conditions = (
                # 1. 價格在BB上方 (BB position > 0.7)
                (df['bb_position'] > 0.7) &
                
                # 2. MACD死叉
                (df['macd_cross_down']) &
                
                # 3. ADX < 閾值 (非強趨勢)
                (df['adx'] < self.adx_threshold)
            )
        
        # 設置信號
        df.loc[long_conditions, 'signal'] = 1
        df.loc[short_conditions, 'signal'] = -1
        
        # 計算信號強度
        # 做多: BB位置越低 + RSI越低 + MACD差值越大 = 信號越強
        df.loc[long_conditions, 'signal_strength'] = (
            # BB位置越低越好 (0.3 - position) / 0.3
            (0.3 - df.loc[long_conditions, 'bb_position']) / 0.3 * 0.4 +
            # RSI越低越好 (加分項,不是必須)
            np.clip((50 - df.loc[long_conditions, 'rsi']) / 50, 0, 1) * 0.3 +
            # MACD正值越大越好
            np.clip(df.loc[long_conditions, 'macd_diff'] / df.loc[long_conditions, 'close'] * 1000, 0, 1) * 0.3
        )
        
        # 做空: BB位置越高 + RSI越高 + MACD差值越大 = 信號越強
        df.loc[short_conditions, 'signal_strength'] = (
            # BB位置越高越好 (position - 0.7) / 0.3
            (df.loc[short_conditions, 'bb_position'] - 0.7) / 0.3 * 0.4 +
            # RSI越高越好 (加分項,不是必須)
            np.clip((df.loc[short_conditions, 'rsi'] - 50) / 50, 0, 1) * 0.3 +
            # MACD負值越大越好
            np.clip(-df.loc[short_conditions, 'macd_diff'] / df.loc[short_conditions, 'close'] * 1000, 0, 1) * 0.3
        )
        
        # 確保信號強度在[0, 1]範圍
        df['signal_strength'] = np.clip(df['signal_strength'], 0, 1)
        
        return df
    
    def get_signal_description(self, row: pd.Series) -> str:
        """獲取信號描述"""
        if row['signal'] == 1:
            return f"LONG: Price={row['close']:.2f}, BB_pos={row.get('bb_position', 0):.2f}, RSI={row['rsi']:.1f}, MACD={row['macd']:.4f}, ADX={row['adx']:.1f}"
        elif row['signal'] == -1:
            return f"SHORT: Price={row['close']:.2f}, BB_pos={row.get('bb_position', 0):.2f}, RSI={row['rsi']:.1f}, MACD={row['macd']:.4f}, ADX={row['adx']:.1f}"
        else:
            return "NO SIGNAL"
    
    def get_signal_summary(self, df: pd.DataFrame) -> Dict:
        """獲取信號統計摘要"""
        signals = df[df['signal'] != 0]
        
        return {
            'total_signals': len(signals),
            'long_signals': len(signals[signals['signal'] == 1]),
            'short_signals': len(signals[signals['signal'] == -1]),
            'avg_signal_strength': signals['signal_strength'].mean() if len(signals) > 0 else 0,
            'signal_frequency': len(signals) / len(df) * 100,
        }


if __name__ == '__main__':
    print("三重確認信號生成器測試 (RSI寬鬆版)")
    print("="*50)
    
    dates = pd.date_range('2024-01-01', periods=1000, freq='15min')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'open_time': dates,
        'open': 50000 + np.random.randn(1000).cumsum() * 100,
        'high': 50000 + np.random.randn(1000).cumsum() * 100 + 50,
        'low': 50000 + np.random.randn(1000).cumsum() * 100 - 50,
        'close': 50000 + np.random.randn(1000).cumsum() * 100,
        'volume': np.random.randint(100, 1000, 1000)
    })
    
    generator = TripleConfirmSignalGenerator(use_strict_rsi=False)
    df_signals = generator.generate_signals(df)
    
    summary = generator.get_signal_summary(df_signals)
    print(f"\n信號統計:")
    print(f"  總信號數: {summary['total_signals']}")
    print(f"  做多信號: {summary['long_signals']}")
    print(f"  做空信號: {summary['short_signals']}")
    print(f"  平均強度: {summary['avg_signal_strength']:.3f}")
    print(f"  信號頻率: {summary['signal_frequency']:.2f}%")
    
    first_signal = df_signals[df_signals['signal'] != 0].iloc[0] if summary['total_signals'] > 0 else None
    if first_signal is not None:
        print(f"\n第一個信號:")
        print(f"  {generator.get_signal_description(first_signal)}")
        print(f"  信號強度: {first_signal['signal_strength']:.3f}")