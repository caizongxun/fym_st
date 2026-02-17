import pandas as pd
import numpy as np
from typing import Tuple

class LabelGenerator:
    """
    Generate labels for training the three models:
    1. Trend Model (1h): Classify trend strength and direction
    2. Volatility Model (15m): Predict volatility regime changes
    3. Reversal Model (15m): Identify reversal points
    """
    
    @staticmethod
    def label_trend(df: pd.DataFrame, horizon: int = 10, atr_col: str = '1h_atr') -> pd.DataFrame:
        """
        Label trend strength and direction for 1h data
        
        Args:
            df: DataFrame with 1h features
            horizon: Number of candles to look ahead
            atr_col: Column name for ATR
        
        Returns:
            DataFrame with trend labels
        """
        df = df.copy()
        
        # Calculate future price movement
        df['future_high'] = df['high'].rolling(horizon).max().shift(-horizon)
        df['future_low'] = df['low'].rolling(horizon).min().shift(-horizon)
        df['future_close'] = df['close'].shift(-horizon)
        
        # Calculate directional movement relative to ATR
        df['upside'] = (df['future_high'] - df['close']) / df[atr_col]
        df['downside'] = (df['close'] - df['future_low']) / df[atr_col]
        df['net_move'] = (df['future_close'] - df['close']) / df[atr_col]
        
        # Classify trend
        conditions = [
            (df['net_move'] > 2) & (df['upside'] > 2),      # Strong Bullish
            (df['net_move'] > 0.5) & (df['upside'] > 1),    # Weak Bullish
            (df['net_move'] < -2) & (df['downside'] > 2),   # Strong Bearish
            (df['net_move'] < -0.5) & (df['downside'] > 1), # Weak Bearish
        ]
        choices = [4, 3, 0, 1]  # 4=Strong Bull, 3=Weak Bull, 2=Range, 1=Weak Bear, 0=Strong Bear
        df['trend_label'] = np.select(conditions, choices, default=2)
        
        # Trend strength score (0-100)
        df['trend_strength'] = np.clip(abs(df['net_move']) * 20, 0, 100)
        
        # Clean up intermediate columns
        df = df.drop(columns=['future_high', 'future_low', 'future_close', 'upside', 'downside'])
        
        return df
    
    @staticmethod
    def label_volatility(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
        """
        Label volatility regime and trend initiation for 15m data
        
        Args:
            df: DataFrame with 15m features
            horizon: Number of candles to look ahead
        
        Returns:
            DataFrame with volatility labels
        """
        df = df.copy()
        
        # Calculate current and future ATR
        if '15m_atr' not in df.columns:
            df['15m_atr'] = df['high'] - df['low']  # Simplified if ATR not available
        
        df['current_atr'] = df['15m_atr']
        df['future_atr'] = df['15m_atr'].rolling(horizon).mean().shift(-horizon)
        
        # Volatility change
        df['vol_change'] = (df['future_atr'] - df['current_atr']) / df['current_atr']
        
        # Classify volatility regime
        conditions = [
            df['vol_change'] > 0.3,   # High volatility incoming
            df['vol_change'] < -0.3,  # Low volatility incoming
        ]
        choices = [2, 0]  # 2=High, 1=Medium, 0=Low
        df['volatility_regime'] = np.select(conditions, choices, default=1)
        
        # Detect trend initiation
        # Look for price breakout with expanding volatility
        df['price_range'] = df['high'] - df['low']
        df['future_range'] = df['price_range'].rolling(horizon).mean().shift(-horizon)
        df['range_expansion'] = df['future_range'] / df['price_range']
        
        # Trend initiation probability (0-100)
        df['trend_initiation_prob'] = np.clip(
            (df['vol_change'] + df['range_expansion'] - 1) * 100, 0, 100
        )
        
        # Clean up
        df = df.drop(columns=['current_atr', 'future_atr', 'price_range', 
                             'future_range', 'range_expansion'])
        
        return df
    
    @staticmethod
    def label_reversal(df: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
        """
        Label reversal points for 15m data
        
        Args:
            df: DataFrame with 15m features
            horizon: Number of candles to look ahead/behind
        
        Returns:
            DataFrame with reversal labels
        """
        df = df.copy()
        
        # Find local extrema
        # Bullish reversal: current low is the lowest in the window
        df['is_local_low'] = (
            (df['low'] == df['low'].rolling(window=horizon*2+1, center=True).min())
        ).astype(int)
        
        # Bearish reversal: current high is the highest in the window
        df['is_local_high'] = (
            (df['high'] == df['high'].rolling(window=horizon*2+1, center=True).max())
        ).astype(int)
        
        # Calculate reversal strength based on subsequent movement
        df['future_high'] = df['high'].rolling(horizon).max().shift(-horizon)
        df['future_low'] = df['low'].rolling(horizon).min().shift(-horizon)
        
        # Bullish reversal probability
        df['bullish_reversal_strength'] = (
            (df['future_high'] - df['close']) / df['close'] * 100
        ).clip(0, 100)
        
        # Bearish reversal probability  
        df['bearish_reversal_strength'] = (
            (df['close'] - df['future_low']) / df['close'] * 100
        ).clip(0, 100)
        
        # Combined reversal probability
        df['reversal_prob'] = 0.0
        df.loc[df['is_local_low'] == 1, 'reversal_prob'] = df['bullish_reversal_strength']
        df.loc[df['is_local_high'] == 1, 'reversal_prob'] = df['bearish_reversal_strength']
        
        # Reversal direction: 1=bullish, -1=bearish, 0=none
        df['reversal_direction'] = 0
        df.loc[df['is_local_low'] == 1, 'reversal_direction'] = 1
        df.loc[df['is_local_high'] == 1, 'reversal_direction'] = -1
        
        # Predicted support/resistance levels
        df['support_level'] = df['low'].rolling(horizon*2).min()
        df['resistance_level'] = df['high'].rolling(horizon*2).max()
        
        # Clean up
        df = df.drop(columns=['future_high', 'future_low', 'is_local_low', 'is_local_high',
                             'bullish_reversal_strength', 'bearish_reversal_strength'])
        
        return df
    
    @staticmethod
    def split_train_oos(df: pd.DataFrame, oos_size: int = 1500) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and out-of-sample sets
        
        Args:
            df: Full dataset
            oos_size: Number of candles to reserve for OOS validation
        
        Returns:
            Tuple of (training_df, oos_df)
        """
        if len(df) <= oos_size:
            print(f"Warning: Dataset too small ({len(df)} rows) for OOS split")
            return df, pd.DataFrame()
        
        split_idx = len(df) - oos_size
        train_df = df.iloc[:split_idx].copy()
        oos_df = df.iloc[split_idx:].copy()
        
        return train_df, oos_df