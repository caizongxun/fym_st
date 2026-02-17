import pandas as pd
import numpy as np
from typing import Tuple

class LabelGenerator:
    """
    Generate labels for training ML models
    Binary trend classification: Trending (1) vs Ranging (0)
    Direction determined by technical indicators, not ML prediction
    """
    
    def label_trend(self, df: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
        """
        Label trend strength only (binary: trending or ranging)
        Direction will be determined by indicators, not prediction
        
        Args:
            df: DataFrame with OHLCV data
            horizon: Number of candles to look ahead
        
        Returns:
            DataFrame with trend_label (0=ranging, 1=trending) and trend_strength
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
        df['normalized_move'] = np.abs(df['net_move']) / (atr / df['close'] * 100)
        
        # Binary classification: Is there a trend?
        # Trending if abs(move) > 0.8 ATR
        df['trend_label'] = (df['normalized_move'] > 0.8).astype(int)
        
        # Calculate trend strength (0-100)
        df['trend_strength'] = df['normalized_move'].clip(0, 3) / 3 * 100
        
        # Calculate directional consistency
        df['price_momentum'] = df['close'].pct_change(5)
        df['momentum_direction'] = np.sign(df['price_momentum'])
        df['direction_consistency'] = df['momentum_direction'].rolling(10).mean().abs() * 100
        
        # Enhance strength with consistency
        df['trend_strength'] = (df['trend_strength'] * 0.7 + 
                                df['direction_consistency'] * 0.3).clip(0, 100)
        
        # Store actual direction for analysis (not for ML prediction)
        df['actual_direction'] = np.sign(df['net_move'])
        
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
            atr_pct < 0.33,
            atr_pct > 0.66,
        ]
        choices = [0, 2]
        df['volatility_regime'] = np.select(conditions, choices, default=1)
        
        # Trend initiation probability
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['vol_expanding'] = (df['vol_change'] > 10).astype(int)
        df['volume_spike'] = (df['volume_ratio'] > 1.5).astype(int)
        
        df['trend_initiation_prob'] = (
            df['vol_expanding'] * 40 +
            df['volume_spike'] * 40 +
            (atr_pct * 20)
        ).clip(0, 100)
        
        df.drop(['volume_ratio', 'vol_expanding', 'volume_spike'], axis=1, inplace=True)
        
        return df
    
    def label_reversal(self, df: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
        """
        Label reversal points with AGGRESSIVE criteria for high-frequency trading
        
        RELAXED CONDITIONS:
        - Lower price move threshold: 0.3% (was 2%)
        - Multiple momentum indicators (not just RSI)
        - MACD crossovers
        - Price action patterns
        - No strict swing point requirement
        """
        df = df.copy()
        
        # Calculate swing highs and lows (wider window for more signals)
        df['swing_high'] = df['high'].rolling(window=3, center=True).max()
        df['swing_low'] = df['low'].rolling(window=3, center=True).min()
        df['is_swing_high'] = (df['high'] == df['swing_high']).astype(int)
        df['is_swing_low'] = (df['low'] == df['swing_low']).astype(int)
        
        # Future price action (shorter horizon for 15m)
        df['future_high'] = df['high'].shift(-horizon).rolling(horizon).max()
        df['future_low'] = df['low'].shift(-horizon).rolling(horizon).min()
        
        # RELAXED: Only need 0.3% move (was 2%)
        df['potential_up_move'] = (df['future_high'] - df['close']) / df['close'] * 100
        df['potential_down_move'] = (df['close'] - df['future_low']) / df['close'] * 100
        
        # RSI (relaxed thresholds)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD for momentum
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_cross_up'] = ((df['macd'] > df['macd_signal']) & 
                               (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
        df['macd_cross_down'] = ((df['macd'] < df['macd_signal']) & 
                                 (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)
        
        # Price momentum (recent moves)
        df['momentum_3'] = df['close'].pct_change(3) * 100
        df['momentum_5'] = df['close'].pct_change(5) * 100
        
        # Volume confirmation
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volume_surge'] = (df['volume_ratio'] > 1.2).astype(int)
        
        # AGGRESSIVE BULLISH CONDITIONS (multiple signals)
        bullish_conditions = (
            # Price move potential (VERY RELAXED)
            (df['potential_up_move'] > 0.3) &
            (
                # Any of these momentum signals
                (df['rsi'] < 50) |  # RSI below midpoint
                (df['macd_cross_up'] == 1) |  # MACD golden cross
                (df['momentum_3'] < -0.5) |  # Recent pullback
                (df['is_swing_low'] == 1)  # Local low
            )
        )
        
        # AGGRESSIVE BEARISH CONDITIONS (multiple signals)
        bearish_conditions = (
            # Price move potential (VERY RELAXED)
            (df['potential_down_move'] > 0.3) &
            (
                # Any of these momentum signals
                (df['rsi'] > 50) |  # RSI above midpoint
                (df['macd_cross_down'] == 1) |  # MACD death cross
                (df['momentum_3'] > 0.5) |  # Recent rally
                (df['is_swing_high'] == 1)  # Local high
            )
        )
        
        df['reversal_direction'] = 0
        df.loc[bullish_conditions, 'reversal_direction'] = 1
        df.loc[bearish_conditions, 'reversal_direction'] = -1
        
        # Reversal probability (more generous scoring)
        df['reversal_prob'] = 0.0
        
        # Bullish score (easier to reach high probability)
        bullish_score = (
            (df['rsi'] < 50).astype(int) * 20 +  # Below midpoint = signal
            (df['rsi'] < 40).astype(int) * 10 +  # Oversold bonus
            df['is_swing_low'] * 15 +
            df['macd_cross_up'] * 25 +  # Strong signal
            (df['momentum_5'] < -1).astype(int) * 15 +  # Pullback
            df['volume_surge'] * 15 +
            (df['potential_up_move'] / 2 * 20).clip(0, 20)  # Move potential
        )
        df.loc[bullish_conditions, 'reversal_prob'] = bullish_score[bullish_conditions].clip(0, 100)
        
        # Bearish score (easier to reach high probability)
        bearish_score = (
            (df['rsi'] > 50).astype(int) * 20 +  # Above midpoint = signal
            (df['rsi'] > 60).astype(int) * 10 +  # Overbought bonus
            df['is_swing_high'] * 15 +
            df['macd_cross_down'] * 25 +  # Strong signal
            (df['momentum_5'] > 1).astype(int) * 15 +  # Rally
            df['volume_surge'] * 15 +
            (df['potential_down_move'] / 2 * 20).clip(0, 20)  # Move potential
        )
        df.loc[bearish_conditions, 'reversal_prob'] = bearish_score[bearish_conditions].clip(0, 100)
        
        # Support and resistance levels
        df['support_level'] = df['swing_low'].fillna(method='ffill')
        df['resistance_level'] = df['swing_high'].fillna(method='ffill')
        
        # Cleanup
        df.drop(['swing_high', 'swing_low', 'is_swing_high', 'is_swing_low',
                'future_high', 'future_low', 'potential_up_move', 'potential_down_move',
                'rsi', 'macd', 'macd_signal', 'macd_cross_up', 'macd_cross_down',
                'momentum_3', 'momentum_5', 'volume_ratio', 'volume_surge'], 
                axis=1, inplace=True)
        
        return df
    
    def split_train_oos(self, df: pd.DataFrame, oos_size: int = 1500) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and out-of-sample sets
        """
        if len(df) <= oos_size:
            return df, pd.DataFrame()
        
        split_idx = len(df) - oos_size
        train_df = df.iloc[:split_idx].copy()
        oos_df = df.iloc[split_idx:].copy()
        
        return train_df, oos_df