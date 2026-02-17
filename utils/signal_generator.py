import pandas as pd
import numpy as np
from typing import Dict, List

class SignalGenerator:
    """
    OPTIMIZED: Trend + Binary Reversal Strategy with Trend Strength Filter
    
    NEW: Avoid counter-trend trades in STRONG trends
    - Strong uptrend + reversal → NO SHORT (wait for real reversal)
    - Strong downtrend + reversal → NO LONG (wait for real reversal)
    - Only trade pullbacks in WEAK trends or ranging markets
    
    Logic:
    1. Detect current trend + strength
    2. Wait for reversal signal
    3. Filter out false reversals in strong trends
    4. Only counter-trade in weak trends or ranging
    """
    
    def __init__(self, 
                 trend_strength_threshold: float = 60.0,  # Above = strong trend
                 reversal_prob_threshold: float = 50.0):   # Minimum reversal confidence
        self.trend_strength_threshold = trend_strength_threshold
        self.reversal_prob_threshold = reversal_prob_threshold
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals with trend strength filtering
        
        Strategy:
        1. Check trend direction + strength
        2. Check reversal signal + probability
        3. ONLY trade if:
           a) Ranging market (any reversal)
           b) WEAK trend (trend_strength < 60) + reversal
           c) STRONG reversal signal (reversal_prob > 60) regardless of trend
        """
        df = df.copy()
        df['signal'] = 0
        
        # Get trend info
        trend = df['trend_direction']  # 1=up, -1=down, 0=ranging
        is_trending = df['is_trending']  # 0=ranging, 1=trending
        trend_strength = df.get('trend_strength_pred', 50)  # Default 50 if missing
        
        # Get reversal info
        reversal = df['reversal_signal']  # 0=no, 1=yes
        reversal_prob = df.get('reversal_prob_pred', 0)  # 0-100
        
        # Calculate short-term momentum
        df['momentum_3'] = df['close'].pct_change(3) * 100
        df['momentum_5'] = df['close'].pct_change(5) * 100
        
        # === FILTER 1: Trend Strength ===
        # Strong trend = trend_strength > threshold AND is_trending = 1
        strong_trend = (is_trending == 1) & (trend_strength > self.trend_strength_threshold)
        weak_trend = (is_trending == 1) & (trend_strength <= self.trend_strength_threshold)
        ranging = (is_trending == 0)
        
        # === FILTER 2: Reversal Quality ===
        # Strong reversal = reversal_prob > 60
        strong_reversal = (reversal == 1) & (reversal_prob > 60)
        normal_reversal = (reversal == 1) & (reversal_prob >= self.reversal_prob_threshold) & (reversal_prob <= 60)
        
        # === SIGNAL GENERATION LOGIC ===
        
        # 1. RANGING MARKET: Trade any reversal with momentum confirmation
        ranging_long = (
            ranging & 
            (reversal == 1) & 
            (reversal_prob >= self.reversal_prob_threshold) &
            (df['momentum_3'] < -0.3)  # Downward momentum = buy low
        )
        
        ranging_short = (
            ranging & 
            (reversal == 1) & 
            (reversal_prob >= self.reversal_prob_threshold) &
            (df['momentum_3'] > 0.3)  # Upward momentum = sell high
        )
        
        # 2. WEAK TREND: Safe to counter-trade (original logic)
        weak_uptrend_short = (
            weak_trend & 
            (trend == 1) & 
            (reversal == 1) &
            (reversal_prob >= self.reversal_prob_threshold)
        )
        
        weak_downtrend_long = (
            weak_trend & 
            (trend == -1) & 
            (reversal == 1) &
            (reversal_prob >= self.reversal_prob_threshold)
        )
        
        # 3. STRONG TREND: ONLY trade if STRONG reversal signal
        #    (High confidence that trend is actually reversing)
        strong_uptrend_short = (
            strong_trend & 
            (trend == 1) & 
            strong_reversal &
            (df['momentum_5'] < -0.5)  # Need strong opposite momentum
        )
        
        strong_downtrend_long = (
            strong_trend & 
            (trend == -1) & 
            strong_reversal &
            (df['momentum_5'] > 0.5)  # Need strong opposite momentum
        )
        
        # === ASSIGN SIGNALS ===
        # LONG conditions
        all_long = (
            ranging_long | 
            weak_downtrend_long | 
            strong_downtrend_long
        )
        
        # SHORT conditions
        all_short = (
            ranging_short | 
            weak_uptrend_short | 
            strong_uptrend_short
        )
        
        df.loc[all_long, 'signal'] = 1
        df.loc[all_short, 'signal'] = -1
        
        # Store filter info for debugging
        df['trend_filter'] = 'none'
        df.loc[strong_trend, 'trend_filter'] = 'strong_trend'
        df.loc[weak_trend, 'trend_filter'] = 'weak_trend'
        df.loc[ranging, 'trend_filter'] = 'ranging'
        
        # Cleanup
        df.drop(['momentum_3', 'momentum_5'], axis=1, inplace=True)
        
        return df
    
    def add_signal_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        signal_map = {1: 'LONG', -1: 'SHORT', 0: 'HOLD'}
        df['signal_name'] = df['signal'].map(signal_map)
        df['signal_strength'] = df.get('reversal_prob_pred', 0)
        
        # Add reversal quality label
        df['reversal_quality'] = 'none'
        reversal_prob = df.get('reversal_prob_pred', 0)
        df.loc[(df['reversal_signal'] == 1) & (reversal_prob >= 50) & (reversal_prob < 60), 'reversal_quality'] = 'normal'
        df.loc[(df['reversal_signal'] == 1) & (reversal_prob >= 60) & (reversal_prob < 75), 'reversal_quality'] = 'good'
        df.loc[(df['reversal_signal'] == 1) & (reversal_prob >= 75), 'reversal_quality'] = 'excellent'
        
        return df