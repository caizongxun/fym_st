import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, 
                 initial_capital: float = 10000.0,
                 commission_rate: float = 0.001,
                 slippage: float = 0.0005):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
    
    def run_backtest(self, 
                     signals: pd.DataFrame,
                     price_column: str = 'close',
                     position_size_column: str = 'position_size',
                     tp_multiplier: float = 2.5,
                     sl_multiplier: float = 1.5) -> Dict:
        
        logger.info(f"Running backtest on {len(signals)} signals")
        
        trades = []
        capital = self.initial_capital
        
        for idx, row in signals.iterrows():
            entry_price = row[price_column] * (1 + self.slippage)
            position_size = row[position_size_column]
            atr = row['atr']
            
            position_value = capital * position_size
            
            commission = position_value * self.commission_rate
            
            tp_price = entry_price + (tp_multiplier * atr)
            sl_price = entry_price - (sl_multiplier * atr)
            
            future_data = signals.loc[idx:].iloc[1:25]
            
            if len(future_data) == 0:
                continue
            
            exit_price = None
            exit_reason = None
            exit_bars = 0
            
            for i, future_row in enumerate(future_data.iterrows()):
                _, frow = future_row
                if frow[price_column] >= tp_price:
                    exit_price = tp_price * (1 - self.slippage)
                    exit_reason = 'TP'
                    exit_bars = i + 1
                    break
                elif frow[price_column] <= sl_price:
                    exit_price = sl_price * (1 - self.slippage)
                    exit_reason = 'SL'
                    exit_bars = i + 1
                    break
            
            if exit_price is None:
                exit_price = future_data.iloc[-1][price_column] * (1 - self.slippage)
                exit_reason = 'Timeout'
                exit_bars = len(future_data)
            
            pnl_pct = (exit_price - entry_price) / entry_price
            pnl_dollar = position_value * pnl_pct - (2 * commission)
            
            capital += pnl_dollar
            
            trades.append({
                'entry_time': row['open_time'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'exit_bars': exit_bars,
                'position_size': position_size,
                'pnl_pct': pnl_pct,
                'pnl_dollar': pnl_dollar,
                'capital': capital
            })
        
        trades_df = pd.DataFrame(trades)
        
        if len(trades_df) == 0:
            logger.warning("No trades executed in backtest")
            return self._empty_stats()
        
        stats = self._calculate_statistics(trades_df)
        logger.info(f"Backtest complete: {len(trades_df)} trades, Final capital: ${stats['final_capital']:.2f}")
        
        return {
            'trades': trades_df,
            'statistics': stats
        }
    
    def _calculate_statistics(self, trades_df: pd.DataFrame) -> Dict:
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl_dollar'] > 0])
        losing_trades = len(trades_df[trades_df['pnl_dollar'] <= 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = trades_df['pnl_dollar'].sum()
        final_capital = trades_df.iloc[-1]['capital'] if len(trades_df) > 0 else self.initial_capital
        total_return = (final_capital - self.initial_capital) / self.initial_capital
        
        avg_win = trades_df[trades_df['pnl_dollar'] > 0]['pnl_dollar'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl_dollar'] <= 0]['pnl_dollar'].mean() if losing_trades > 0 else 0
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else np.inf
        
        trades_df['cumulative_return'] = (trades_df['capital'] - self.initial_capital) / self.initial_capital
        trades_df['drawdown'] = trades_df['cumulative_return'] - trades_df['cumulative_return'].cummax()
        max_drawdown = trades_df['drawdown'].min()
        
        sharpe_ratio = trades_df['pnl_pct'].mean() / trades_df['pnl_pct'].std() * np.sqrt(252) if trades_df['pnl_pct'].std() > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'final_capital': final_capital,
            'total_return': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }
    
    def _empty_stats(self) -> Dict:
        return {
            'trades': pd.DataFrame(),
            'statistics': {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'final_capital': self.initial_capital,
                'total_return': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
        }