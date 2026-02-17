import pandas as pd
import numpy as np
from typing import Dict, List

class SignalGenerator:
    """
    v3: COMPLETE REDESIGN - Trade WITH Trend, Not Against It
    
    OLD PROBLEM:
    - Uptrend + reversal → SHORT ❌ (逆勢,101筆止損)
    - Downtrend + reversal → LONG ❌ (逆勢,74筆止損)
    - Result: 93% of losses from counter-trend trades
    
    NEW APPROACH: Momentum Breakout Strategy
    - Ranging + strong momentum → Enter in momentum direction
    - Weak trend + momentum acceleration → Enter in trend direction
    - Strong trend + pullback + reversal → Enter in trend direction (pullback entry)
    
    KEY: Always trade WITH the trend, never against it
    """
    
    def __init__(self, 
                 min_reversal_prob: float = 60.0,     # Minimum reversal confidence
                 ranging_momentum_threshold: float = 0.5):  # Momentum for ranging breakout
        self.min_reversal_prob = min_reversal_prob
        self.ranging_momentum_threshold = ranging_momentum_threshold
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals - WITH trend, not against
        
        Strategy:
        1. RANGING MARKET: Trade breakouts (momentum direction)
        2. TRENDING MARKET: Trade pullbacks (trend direction)
        3. NEVER counter-trend unless trend actually reverses
        """
        df = df.copy()
        df['signal'] = 0
        
        # Get trend info
        is_trending = df['is_trending']  # 0=ranging, 1=trending
        trend_direction = df['trend_direction']  # 1=up, -1=down, 0=ranging
        trend_strength = df.get('trend_strength_pred', 50)
        
        # Get reversal info (now used as "momentum shift" indicator)
        reversal_signal = df['reversal_signal']  # 0=no, 1=yes
        reversal_prob = df.get('reversal_prob_pred', 0)
        
        # Calculate momentum
        df['momentum_3'] = df['close'].pct_change(3) * 100
        df['momentum_5'] = df['close'].pct_change(5) * 100
        
        # Price vs EMA (for pullback detection)
        if '15m_ema_20' in df.columns:
            df['price_vs_ema20'] = (df['close'] - df['15m_ema_20']) / df['15m_ema_20'] * 100
        else:
            df['price_vs_ema20'] = 0
        
        # === STRATEGY 1: RANGING MARKET BREAKOUT ===
        # Wait for strong momentum in one direction
        ranging = (is_trending == 0)
        
        ranging_long = (
            ranging & 
            (df['momentum_5'] < -self.ranging_momentum_threshold) &  # Strong downward momentum
            (reversal_signal == 1) &  # Reversal detected (potential bounce)
            (reversal_prob >= self.min_reversal_prob)  # High confidence
        )
        
        ranging_short = (
            ranging & 
            (df['momentum_5'] > self.ranging_momentum_threshold) &  # Strong upward momentum
            (reversal_signal == 1) &  # Reversal detected (potential drop)
            (reversal_prob >= self.min_reversal_prob)
        )
        
        # === STRATEGY 2: TREND FOLLOWING (PULLBACK ENTRY) ===
        # In uptrend: wait for pullback, then enter LONG
        # In downtrend: wait for bounce, then enter SHORT
        
        trending = (is_trending == 1)
        
        # UPTREND: Wait for pullback (price < EMA), then reversal = buy signal
        uptrend_pullback_long = (
            trending &
            (trend_direction == 1) &  # Uptrend
            (df['price_vs_ema20'] < -0.3) &  # Price pulled back below EMA
            (reversal_signal == 1) &  # Reversal detected (bounce)
            (reversal_prob >= self.min_reversal_prob) &
            (df['momentum_3'] > 0.2)  # Starting to bounce
        )
        
        # DOWNTREND: Wait for bounce (price > EMA), then reversal = sell signal
        downtrend_bounce_short = (
            trending &
            (trend_direction == -1) &  # Downtrend
            (df['price_vs_ema20'] > 0.3) &  # Price bounced above EMA
            (reversal_signal == 1) &  # Reversal detected (resume down)
            (reversal_prob >= self.min_reversal_prob) &
            (df['momentum_3'] < -0.2)  # Starting to drop
        )
        
        # === STRATEGY 3: TREND REVERSAL (VERY SELECTIVE) ===
        # Only trade actual trend reversals with EXTREME confidence
        
        # Uptrend reversing to downtrend
        trend_reversal_short = (
            trending &
            (trend_direction == 1) &  # Was uptrend
            (trend_strength < 45) &  # Trend weakening
            (reversal_signal == 1) &
            (reversal_prob >= 75) &  # VERY high confidence
            (df['momentum_5'] < -1.0) &  # Strong opposite momentum
            (df['price_vs_ema20'] < -0.5)  # Price significantly below EMA
        )
        
        # Downtrend reversing to uptrend  
        trend_reversal_long = (
            trending &
            (trend_direction == -1) &  # Was downtrend
            (trend_strength < 45) &  # Trend weakening
            (reversal_signal == 1) &
            (reversal_prob >= 75) &  # VERY high confidence
            (df['momentum_5'] > 1.0) &  # Strong opposite momentum
            (df['price_vs_ema20'] > 0.5)  # Price significantly above EMA
        )
        
        # === ASSIGN SIGNALS ===
        all_long = (
            ranging_long | 
            uptrend_pullback_long | 
            trend_reversal_long
        )
        
        all_short = (
            ranging_short | 
            downtrend_bounce_short | 
            trend_reversal_short
        )
        
        df.loc[all_long, 'signal'] = 1
        df.loc[all_short, 'signal'] = -1
        
        # Store strategy type for analysis
        df['signal_strategy'] = 'none'
        df.loc[ranging_long | ranging_short, 'signal_strategy'] = 'ranging_breakout'
        df.loc[uptrend_pullback_long | downtrend_bounce_short, 'signal_strategy'] = 'trend_pullback'
        df.loc[trend_reversal_long | trend_reversal_short, 'signal_strategy'] = 'trend_reversal'
        
        # Cleanup
        df.drop(['momentum_3', 'momentum_5', 'price_vs_ema20'], axis=1, inplace=True)
        
        return df
    
    def add_signal_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        signal_map = {1: 'LONG', -1: 'SHORT', 0: 'HOLD'}
        df['signal_name'] = df['signal'].map(signal_map)
        df['signal_strength'] = df.get('reversal_prob_pred', 0)
        
        return df