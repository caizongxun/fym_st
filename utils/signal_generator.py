import pandas as pd
import numpy as np
from typing import Dict, List

class SignalGenerator:
    """
    v4: Relaxed Filters - More Trades with Better Quality
    
    v3 problem: Too strict (1 trade only)
    v4 solution: Loosen conditions while keeping trend-following principle
    """
    
    def __init__(self, 
                 min_reversal_prob: float = 45.0,     # Lowered from 60
                 ranging_momentum_threshold: float = 0.3):  # Lowered from 0.5
        self.min_reversal_prob = min_reversal_prob
        self.ranging_momentum_threshold = ranging_momentum_threshold
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals - WITH trend, but more permissive
        """
        df = df.copy()
        df['signal'] = 0
        
        # Get trend info
        is_trending = df['is_trending']  # 0=ranging, 1=trending
        trend_direction = df['trend_direction']  # 1=up, -1=down, 0=ranging
        trend_strength = df.get('trend_strength_pred', 50)
        
        # Get reversal info
        reversal_signal = df['reversal_signal']  # 0=no, 1=yes
        reversal_prob = df.get('reversal_prob_pred', 0)
        
        # Calculate momentum
        df['momentum_3'] = df['close'].pct_change(3) * 100
        df['momentum_5'] = df['close'].pct_change(5) * 100
        
        # Price vs EMA
        if '15m_ema_20' in df.columns:
            df['price_vs_ema20'] = (df['close'] - df['15m_ema_20']) / df['15m_ema_20'] * 100
        else:
            df['price_vs_ema20'] = 0
        
        # === STRATEGY 1: RANGING MARKET ===
        # More permissive - any reversal signal
        ranging = (is_trending == 0)
        
        ranging_long = (
            ranging & 
            (reversal_signal == 1) &
            (reversal_prob >= self.min_reversal_prob) &
            (df['momentum_3'] < 0)  # Any downward momentum
        )
        
        ranging_short = (
            ranging & 
            (reversal_signal == 1) &
            (reversal_prob >= self.min_reversal_prob) &
            (df['momentum_3'] > 0)  # Any upward momentum
        )
        
        # === STRATEGY 2: TREND FOLLOWING ===
        trending = (is_trending == 1)
        
        # UPTREND: Enter on any reversal signal (less strict)
        uptrend_long = (
            trending &
            (trend_direction == 1) &  # Uptrend
            (reversal_signal == 1) &
            (reversal_prob >= self.min_reversal_prob) &
            (
                # Condition A: Pullback entry (preferred)
                (df['price_vs_ema20'] < 0) |
                # Condition B: Momentum continuation
                (df['momentum_3'] > 0.2)
            )
        )
        
        # DOWNTREND: Enter on any reversal signal
        downtrend_short = (
            trending &
            (trend_direction == -1) &  # Downtrend
            (reversal_signal == 1) &
            (reversal_prob >= self.min_reversal_prob) &
            (
                # Condition A: Bounce entry (preferred)
                (df['price_vs_ema20'] > 0) |
                # Condition B: Momentum continuation  
                (df['momentum_3'] < -0.2)
            )
        )
        
        # === STRATEGY 3: HIGH CONFIDENCE COUNTER-TREND ===
        # Only when VERY confident
        
        counter_uptrend_short = (
            trending &
            (trend_direction == 1) &  # Uptrend
            (reversal_signal == 1) &
            (reversal_prob >= 70) &  # High confidence
            (df['momentum_5'] < -0.8) &  # Strong opposite momentum
            (trend_strength < 50)  # Weakening trend
        )
        
        counter_downtrend_long = (
            trending &
            (trend_direction == -1) &  # Downtrend
            (reversal_signal == 1) &
            (reversal_prob >= 70) &  # High confidence
            (df['momentum_5'] > 0.8) &  # Strong opposite momentum
            (trend_strength < 50)  # Weakening trend
        )
        
        # === ASSIGN SIGNALS ===
        all_long = (
            ranging_long | 
            uptrend_long | 
            counter_downtrend_long
        )
        
        all_short = (
            ranging_short | 
            downtrend_short | 
            counter_uptrend_short
        )
        
        df.loc[all_long, 'signal'] = 1
        df.loc[all_short, 'signal'] = -1
        
        # Store strategy type
        df['signal_strategy'] = 'none'
        df.loc[ranging_long | ranging_short, 'signal_strategy'] = 'ranging'
        df.loc[uptrend_long | downtrend_short, 'signal_strategy'] = 'trend_follow'
        df.loc[counter_downtrend_long | counter_uptrend_short, 'signal_strategy'] = 'counter_trend'
        
        # Cleanup
        df.drop(['momentum_3', 'momentum_5', 'price_vs_ema20'], axis=1, inplace=True)
        
        return df
    
    def add_signal_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        signal_map = {1: 'LONG', -1: 'SHORT', 0: 'HOLD'}
        df['signal_name'] = df['signal'].map(signal_map)
        df['signal_strength'] = df.get('reversal_prob_pred', 0)
        
        return df