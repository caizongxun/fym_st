"""Range-Bound Strategy Signal Generator"""

import pandas as pd
import numpy as np
from models.range_bound_strategy import RangeBoundStrategy


class RangeBoundSignalGenerator:
    def __init__(
        self,
        adx_threshold: float = 25,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        volume_threshold: float = 0.8,
        use_atr_stops: bool = True,
        atr_multiplier: float = 2.0,
        fixed_stop_pct: float = 0.02,
        target_rr: float = 2.0
    ):
        self.strategy = RangeBoundStrategy(
            adx_threshold=adx_threshold,
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought,
            volume_threshold=volume_threshold,
            use_atr_stops=use_atr_stops,
            atr_multiplier=atr_multiplier,
            fixed_stop_pct=fixed_stop_pct,
            target_rr=target_rr
        )
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals"""
        df = df.copy()
        df = self.strategy.add_indicators(df)
        
        df['signal'] = 0
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan
        
        for i in range(50, len(df)):
            row = df.iloc[i]
            
            if pd.isna(row['adx']) or row['adx'] >= self.strategy.adx_threshold:
                continue
            
            if pd.isna(row['bb_width']) or row['bb_width'] < 0.01:
                continue
            
            current_price = row['close']
            atr = row['atr']
            volume_contracted = row['volume'] < (row['volume_ma'] * self.strategy.volume_threshold)
            
            if (row['close'] <= row['bb_lower'] and 
                row['rsi'] < self.strategy.rsi_oversold and 
                volume_contracted):
                
                if self.strategy.use_atr_stops:
                    stop_loss = current_price - (atr * self.strategy.atr_multiplier)
                else:
                    stop_loss = current_price * (1 - self.strategy.fixed_stop_pct)
                
                take_profit = row['bb_mid']
                risk = current_price - stop_loss
                reward = take_profit - current_price
                if reward / risk < 1.0:
                    take_profit = current_price + (risk * self.strategy.target_rr)
                
                df.loc[df.index[i], 'signal'] = 1
                df.loc[df.index[i], 'stop_loss'] = stop_loss
                df.loc[df.index[i], 'take_profit'] = take_profit
            
            elif (row['close'] >= row['bb_upper'] and 
                  row['rsi'] > self.strategy.rsi_overbought and 
                  volume_contracted):
                
                if self.strategy.use_atr_stops:
                    stop_loss = current_price + (atr * self.strategy.atr_multiplier)
                else:
                    stop_loss = current_price * (1 + self.strategy.fixed_stop_pct)
                
                take_profit = row['bb_mid']
                risk = stop_loss - current_price
                reward = current_price - take_profit
                if reward / risk < 1.0:
                    take_profit = current_price - (risk * self.strategy.target_rr)
                
                df.loc[df.index[i], 'signal'] = -1
                df.loc[df.index[i], 'stop_loss'] = stop_loss
                df.loc[df.index[i], 'take_profit'] = take_profit
        
        return df
