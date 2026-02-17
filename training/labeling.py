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
        BINARY REVERSAL DETECTION: Only detect IF reversal occurs (not direction)
        
        Logic:
        1. Detect momentum shifts (price direction change)
        2. Only label as reversal = 1, no direction
        3. Trading logic will use trend + reversal:
           - Uptrend + reversal → go SHORT
           - Downtrend + reversal → go LONG
        
        Returns:
            reversal_signal: 0 (no reversal) or 1 (has reversal)
            reversal_prob: 0-100 (strength of reversal)
        """
        df = df.copy()
        
        # Calculate momentum indicators
        df['momentum_3'] = df['close'].pct_change(3) * 100
        df['momentum_5'] = df['close'].pct_change(5) * 100
        df['momentum_10'] = df['close'].pct_change(10) * 100
        
        # Future momentum (to detect actual reversal)
        df['future_momentum_3'] = df['close'].shift(-3).pct_change(3) * 100
        df['future_momentum_5'] = df['close'].shift(-5).pct_change(5) * 100
        
        # RSI for extremes
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD for trend change
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Detect MACD histogram direction change (reversal signal)
        df['macd_hist_change'] = df['macd_hist'].diff()
        df['macd_reversal'] = (
            ((df['macd_hist'] > 0) & (df['macd_hist_change'] < 0)) |  # Peak
            ((df['macd_hist'] < 0) & (df['macd_hist_change'] > 0))     # Trough
        ).astype(int)
        
        # Price action: local extremes (3-bar patterns)
        df['local_high'] = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
        df['local_low'] = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))
        df['price_extreme'] = (df['local_high'] | df['local_low']).astype(int)
        
        # Volume confirmation
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volume_surge'] = (df['volume_ratio'] > 1.2).astype(int)
        
        # CORE LOGIC: Detect momentum reversal
        # Reversal = current momentum and future momentum have opposite signs
        df['momentum_reversal'] = (
            (np.sign(df['momentum_5']) != np.sign(df['future_momentum_5'])) &
            (np.abs(df['future_momentum_5']) > 0.3)  # Future move > 0.3%
        ).astype(int)
        
        # AGGRESSIVE CONDITIONS: Multiple reversal signals
        reversal_conditions = (
            # Core: momentum reversal detected
            (df['momentum_reversal'] == 1) |
            
            # MACD reversal (strong signal)
            (df['macd_reversal'] == 1) |
            
            # RSI extremes with price action
            ((df['rsi'] < 40) & (df['price_extreme'] == 1)) |
            ((df['rsi'] > 60) & (df['price_extreme'] == 1)) |
            
            # Strong momentum shift
            ((df['momentum_3'] > 1) & (df['future_momentum_3'] < -0.5)) |
            ((df['momentum_3'] < -1) & (df['future_momentum_3'] > 0.5))
        )
        
        # BINARY OUTPUT: 0 or 1 only
        df['reversal_signal'] = reversal_conditions.astype(int)
        
        # Reversal probability (strength)
        df['reversal_prob'] = 0.0
        
        reversal_score = (
            df['momentum_reversal'] * 30 +           # Core signal
            df['macd_reversal'] * 25 +                # Trend change
            df['price_extreme'] * 15 +                # Local extreme
            ((df['rsi'] < 40) | (df['rsi'] > 60)).astype(int) * 15 +  # RSI extreme
            df['volume_surge'] * 10 +                 # Volume confirmation
            (np.abs(df['momentum_5']) / 3 * 5).clip(0, 5)  # Momentum strength
        )
        
        df.loc[reversal_conditions, 'reversal_prob'] = reversal_score[reversal_conditions].clip(0, 100)
        
        # Support and resistance (for stop loss/take profit)
        df['swing_high'] = df['high'].rolling(window=5, center=True).max()
        df['swing_low'] = df['low'].rolling(window=5, center=True).min()
        df['support_level'] = df['swing_low'].fillna(method='ffill')
        df['resistance_level'] = df['swing_high'].fillna(method='ffill')
        
        # Cleanup temporary columns
        df.drop(['momentum_3', 'momentum_5', 'momentum_10', 'future_momentum_3', 
                'future_momentum_5', 'rsi', 'macd', 'macd_signal', 'macd_hist',
                'macd_hist_change', 'macd_reversal', 'local_high', 'local_low',
                'price_extreme', 'volume_ratio', 'volume_surge', 'momentum_reversal',
                'swing_high', 'swing_low'], axis=1, inplace=True)
        
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