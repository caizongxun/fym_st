import pandas as pd
import numpy as np
from typing import Dict, List

class SignalGenerator:
    """
    v5: PURE TECHNICAL BREAKOUT - NO AI PREDICTION
    
    Problem with v1-v4: Reversal prediction model has NO predictive power
    - Higher reversal prob → LOWER win rate (opposite!)
    - 50-55% prob: 44% win rate
    - >60% prob: 35% win rate
    
    New Approach: Classic Technical Analysis
    - EMA crossovers for trend confirmation
    - RSI for overbought/oversold
    - Volume confirmation
    - Support/resistance breakouts
    - Moving average confluence
    
    NO AI predictions, just pure price action
    """
    
    def __init__(self):
        self.ema_fast = 9
        self.ema_slow = 21
        self.rsi_period = 14
        self.rsi_oversold = 35
        self.rsi_overbought = 65
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals using ONLY technical indicators
        No ML models, no predictions
        """
        df = df.copy()
        df['signal'] = 0
        
        # Calculate technical indicators
        df = self._calculate_indicators(df)
        
        # === LONG SIGNALS ===
        # Condition 1: Price breaks above EMA_fast after being below
        bullish_cross = (df['close'] > df['ema_fast']) & (df['close'].shift(1) <= df['ema_fast'].shift(1))
        
        # Condition 2: EMA_fast trending up
        ema_fast_rising = df['ema_fast'] > df['ema_fast'].shift(2)
        
        # Condition 3: RSI oversold or neutral (not overbought)
        rsi_ok_long = df['rsi'] < self.rsi_overbought
        
        # Condition 4: Above longer-term EMA (trend filter)
        above_ema_slow = df['close'] > df['ema_slow']
        
        # Condition 5: Volume confirmation (above average)
        volume_confirm = df['volume'] > df['volume_ma']
        
        # LONG entry
        long_signal = (
            bullish_cross &
            ema_fast_rising &
            rsi_ok_long &
            above_ema_slow &
            volume_confirm
        )
        
        # === SHORT SIGNALS ===
        # Condition 1: Price breaks below EMA_fast after being above
        bearish_cross = (df['close'] < df['ema_fast']) & (df['close'].shift(1) >= df['ema_fast'].shift(1))
        
        # Condition 2: EMA_fast trending down
        ema_fast_falling = df['ema_fast'] < df['ema_fast'].shift(2)
        
        # Condition 3: RSI overbought or neutral (not oversold)
        rsi_ok_short = df['rsi'] > self.rsi_oversold
        
        # Condition 4: Below longer-term EMA (trend filter)
        below_ema_slow = df['close'] < df['ema_slow']
        
        # Condition 5: Volume confirmation
        volume_confirm_short = df['volume'] > df['volume_ma']
        
        # SHORT entry
        short_signal = (
            bearish_cross &
            ema_fast_falling &
            rsi_ok_short &
            below_ema_slow &
            volume_confirm_short
        )
        
        # === ADDITIONAL FILTER: MOMENTUM ===
        # Only take signals when there's clear momentum
        price_change_3 = df['close'].pct_change(3) * 100
        strong_momentum_up = price_change_3 > 0.3
        strong_momentum_down = price_change_3 < -0.3
        
        # Final signals
        df.loc[long_signal & strong_momentum_up, 'signal'] = 1
        df.loc[short_signal & strong_momentum_down, 'signal'] = -1
        
        # Add signal metadata
        df['signal_strategy'] = 'technical_breakout'
        df['signal_strength'] = df['rsi']  # Use RSI as signal strength
        
        return df
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators
        """
        # EMAs
        df['ema_fast'] = df['close'].ewm(span=self.ema_fast, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.ema_slow, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Volume MA
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        
        return df
    
    def add_signal_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add human-readable signal names
        """
        df = df.copy()
        
        signal_map = {1: 'LONG', -1: 'SHORT', 0: 'HOLD'}
        df['signal_name'] = df['signal'].map(signal_map)
        
        # Use RSI as confidence (distance from 50)
        df['signal_strength'] = abs(df.get('rsi', 50) - 50)
        
        return df