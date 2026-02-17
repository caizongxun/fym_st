import pandas as pd
import numpy as np
from typing import Tuple

class LabelGenerator:
    """
    Generate labels for training ML models
    Simplified trend classification: Bull (1), Range (0), Bear (-1)
    """
    
    def label_trend(self, df: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
        """
        Label trend direction and strength (SIMPLIFIED 3-CLASS VERSION)
        
        Args:
            df: DataFrame with OHLCV data
            horizon: Number of candles to look ahead
        
        Returns:
            DataFrame with trend_label and trend_strength columns
        """
        df = df.copy()
        
        # Calculate future return
        df['future_close'] = df['close'].shift(-horizon)
        df['net_move'] = (df['future_close'] - df['close']) / df['close'] * 100
        
        # Calculate ATR for normalization
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        atr = ranges.max(axis=1).rolling(window=14).mean()
        
        # Normalized move (in ATR terms)
        df['normalized_move'] = df['net_move'] / (atr / df['close'] * 100)
        
        # Simplified 3-class labeling with clearer boundaries
        # Bull: normalized_move > 0.8 ATR
        # Bear: normalized_move < -0.8 ATR
        # Range: everything in between
        
        conditions = [
            df['normalized_move'] > 0.8,   # Bullish
            df['normalized_move'] < -0.8,  # Bearish
        ]
        choices = [1, -1]
        
        df['trend_label'] = np.select(conditions, choices, default=0)
        
        # Calculate trend strength (0-100)
        # Based on absolute normalized move, capped at 3 ATR
        df['trend_strength'] = np.abs(df['normalized_move']).clip(0, 3) / 3 * 100
        
        # Calculate directional consistency (rolling window)
        df['price_momentum'] = df['close'].pct_change(5)
        df['momentum_direction'] = np.sign(df['price_momentum'])
        df['direction_consistency'] = df['momentum_direction'].rolling(10).mean().abs() * 100
        
        # Combine with volatility for final strength score
        df['trend_strength'] = (df['trend_strength'] * 0.7 + 
                                df['direction_consistency'] * 0.3).clip(0, 100)
        
        # Clean up temporary columns
        df.drop(['future_close', 'normalized_move', 'price_momentum', 
                'momentum_direction', 'direction_consistency'], axis=1, inplace=True)
        
        return df
    
    def label_volatility(self, df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
        """
        Label volatility regime and trend initiation probability
        """
        df = df.copy()
        
        # Calculate current and future ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        atr = ranges.max(axis=1).rolling(window=14).mean()
        
        # Future volatility change
        future_atr = atr.shift(-horizon)
        df['vol_change'] = (future_atr - atr) / atr * 100
        
        # Classify volatility regime based on percentiles
        atr_pct = atr.rolling(100).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        
        conditions = [
            atr_pct < 0.33,   # Low volatility
            atr_pct > 0.66,   # High volatility
        ]
        choices = [0, 2]
        df['volatility_regime'] = np.select(conditions, choices, default=1)  # Medium
        
        # Trend initiation probability
        # Based on volume spike and volatility expansion
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['vol_expanding'] = (df['vol_change'] > 10).astype(int)
        df['volume_spike'] = (df['volume_ratio'] > 1.5).astype(int)
        
        # Probability score (0-100)
        df['trend_initiation_prob'] = (
            df['vol_expanding'] * 40 +
            df['volume_spike'] * 40 +
            (atr_pct * 20)
        ).clip(0, 100)
        
        # Clean up
        df.drop(['volume_ratio', 'vol_expanding', 'volume_spike'], axis=1, inplace=True)
        
        return df
    
    def label_reversal(self, df: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
        """
        Label reversal points and support/resistance levels
        """
        df = df.copy()
        
        # Calculate swing highs and lows
        df['swing_high'] = df['high'].rolling(window=5, center=True).max()
        df['swing_low'] = df['low'].rolling(window=5, center=True).min()
        df['is_swing_high'] = (df['high'] == df['swing_high']).astype(int)
        df['is_swing_low'] = (df['low'] == df['swing_low']).astype(int)
        
        # Future price action
        df['future_high'] = df['high'].shift(-horizon).rolling(horizon).max()
        df['future_low'] = df['low'].shift(-horizon).rolling(horizon).min()
        
        # Calculate potential reversal magnitude
        df['potential_up_move'] = (df['future_high'] - df['close']) / df['close'] * 100
        df['potential_down_move'] = (df['close'] - df['future_low']) / df['close'] * 100
        
        # RSI for overbought/oversold
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Reversal direction
        # Bullish reversal: at swing low + oversold + future upside
        # Bearish reversal: at swing high + overbought + future downside
        
        bullish_conditions = (
            (df['is_swing_low'] == 1) | 
            (df['rsi'] < 30)
        ) & (df['potential_up_move'] > 2)
        
        bearish_conditions = (
            (df['is_swing_high'] == 1) | 
            (df['rsi'] > 70)
        ) & (df['potential_down_move'] > 2)
        
        df['reversal_direction'] = 0
        df.loc[bullish_conditions, 'reversal_direction'] = 1
        df.loc[bearish_conditions, 'reversal_direction'] = -1
        
        # Reversal probability (0-100)
        df['reversal_prob'] = 0.0
        
        # Bullish reversal probability
        bullish_score = (
            (df['rsi'] < 30).astype(int) * 30 +
            df['is_swing_low'] * 30 +
            (df['potential_up_move'] / 10 * 40).clip(0, 40)
        )
        df.loc[bullish_conditions, 'reversal_prob'] = bullish_score[bullish_conditions]
        
        # Bearish reversal probability
        bearish_score = (
            (df['rsi'] > 70).astype(int) * 30 +
            df['is_swing_high'] * 30 +
            (df['potential_down_move'] / 10 * 40).clip(0, 40)
        )
        df.loc[bearish_conditions, 'reversal_prob'] = bearish_score[bearish_conditions]
        
        # Support and resistance levels
        df['support_level'] = df['swing_low'].fillna(method='ffill')
        df['resistance_level'] = df['swing_high'].fillna(method='ffill')
        
        # Clean up
        df.drop(['swing_high', 'swing_low', 'is_swing_high', 'is_swing_low',
                'future_high', 'future_low', 'potential_up_move', 'potential_down_move',
                'rsi'], axis=1, inplace=True)
        
        return df
    
    def split_train_oos(self, df: pd.DataFrame, oos_size: int = 1500) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and out-of-sample sets
        
        Args:
            df: Full labeled dataset
            oos_size: Number of samples to reserve for OOS validation
        
        Returns:
            (train_df, oos_df)
        """
        if len(df) <= oos_size:
            return df, pd.DataFrame()
        
        split_idx = len(df) - oos_size
        train_df = df.iloc[:split_idx].copy()
        oos_df = df.iloc[split_idx:].copy()
        
        return train_df, oos_df
