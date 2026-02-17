import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class BacktestEngine:
    """
    Flip-flop reversal backtesting engine
    Automatically reverses position on each signal
    """
    
    def __init__(self, 
                 initial_capital: float = 10.0,
                 leverage: float = 10.0,
                 tp_atr_mult: float = None,  # Disabled for reversal strategy
                 sl_atr_mult: float = None,  # Disabled for reversal strategy
                 position_size_pct: float = 0.95,
                 max_positions: int = 1,
                 maker_fee: float = 0.0002,
                 taker_fee: float = 0.0006):
        
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.tp_atr_mult = tp_atr_mult  # Not used in flip-flop
        self.sl_atr_mult = sl_atr_mult  # Not used in flip-flop
        self.position_size_pct = position_size_pct
        self.max_positions = max_positions
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        
        self.equity = initial_capital
        self.peak_equity = initial_capital
        self.trades = []
        self.equity_curve = []
        self.open_positions = {}  # symbol -> position dict
        
    def calculate_position_size(self) -> float:
        """
        Calculate position size based on available capital
        """
        available_capital = self.equity - sum([p['margin'] for p in self.open_positions.values()])
        position_value = available_capital * self.position_size_pct
        return position_value * self.leverage
    
    def open_or_flip_position(self, symbol: str, direction: str, entry_price: float, 
                             timestamp: datetime, signal_data: dict) -> bool:
        """
        Open a new position or flip existing position
        
        Returns:
            True if action taken
        """
        # If position exists, close it first (will be opposite direction)
        if symbol in self.open_positions:
            old_pos = self.open_positions[symbol]
            # Only flip if direction is different
            if old_pos['direction'] != direction:
                self.close_position(symbol, entry_price, timestamp, 'FLIP')
            else:
                # Same direction signal, ignore
                return False
        
        # Open new position
        position_value = self.calculate_position_size()
        if position_value <= 0:
            return False
        
        quantity = position_value / entry_price
        margin = position_value / self.leverage
        entry_fee = position_value * self.taker_fee
        
        self.open_positions[symbol] = {
            'direction': direction,
            'entry_price': entry_price,
            'quantity': quantity,
            'position_value': position_value,
            'margin': margin,
            'entry_fee': entry_fee,
            'entry_time': timestamp,
            'signal_data': signal_data
        }
        
        self.equity -= entry_fee
        
        return True
    
    def close_position(self, symbol: str, exit_price: float, timestamp: datetime, 
                      exit_reason: str) -> Optional[Dict]:
        """
        Close an existing position
        """
        if symbol not in self.open_positions:
            return None
        
        pos = self.open_positions[symbol]
        
        exit_value = pos['quantity'] * exit_price
        exit_fee = exit_value * self.taker_fee
        
        if pos['direction'] == 'LONG':
            pnl = (exit_price - pos['entry_price']) * pos['quantity'] - pos['entry_fee'] - exit_fee
        else:
            pnl = (pos['entry_price'] - exit_price) * pos['quantity'] - pos['entry_fee'] - exit_fee
        
        self.equity += pnl + pos['margin']
        
        trade_record = {
            'symbol': symbol,
            'direction': pos['direction'],
            'entry_time': pos['entry_time'],
            'exit_time': timestamp,
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'quantity': pos['quantity'],
            'pnl': pnl,
            'pnl_pct': (pnl / pos['margin']) * 100,
            'exit_reason': exit_reason,
            'duration': (timestamp - pos['entry_time']).total_seconds() / 60,
            'signal_data': pos['signal_data']
        }
        
        self.trades.append(trade_record)
        del self.open_positions[symbol]
        
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        
        return trade_record
    
    def run_backtest(self, signals_dict: Dict[str, pd.DataFrame]) -> Dict:
        """
        Run flip-flop backtest on multiple symbols
        
        Args:
            signals_dict: Dictionary mapping symbol to DataFrame with signals
                         Required columns: open_time, close, signal (1=long, -1=short, 0=hold)
        
        Returns:
            Dictionary with backtest results
        """
        # Combine all dataframes
        all_data = []
        for symbol, df in signals_dict.items():
            df_copy = df.copy()
            df_copy['symbol'] = symbol
            all_data.append(df_copy)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values('open_time').reset_index(drop=True)
        
        for idx, row in combined_df.iterrows():
            timestamp = row['open_time']
            symbol = row['symbol']
            
            # Check for new signals
            if 'signal' in row and row['signal'] != 0:
                direction = 'LONG' if row['signal'] == 1 else 'SHORT'
                
                signal_data = {
                    'reversal_prob': row.get('reversal_prob_pred', 0),
                    'trend_direction': row.get('trend_direction', 0)
                }
                
                self.open_or_flip_position(symbol, direction, row['close'], timestamp, signal_data)
            
            # Record equity at each step
            unrealized_pnl = 0
            for sym, pos in self.open_positions.items():
                current_price = combined_df[
                    (combined_df['symbol'] == sym) & 
                    (combined_df['open_time'] == timestamp)
                ]['close'].values
                
                if len(current_price) > 0:
                    current_price = current_price[0]
                    if pos['direction'] == 'LONG':
                        unrealized_pnl += (current_price - pos['entry_price']) * pos['quantity']
                    else:
                        unrealized_pnl += (pos['entry_price'] - current_price) * pos['quantity']
            
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': self.equity + unrealized_pnl,
                'open_positions': len(self.open_positions)
            })
        
        # Close any remaining positions at last price
        for symbol in list(self.open_positions.keys()):
            last_price = combined_df[combined_df['symbol'] == symbol]['close'].iloc[-1]
            last_time = combined_df[combined_df['symbol'] == symbol]['open_time'].iloc[-1]
            self.close_position(symbol, last_price, last_time, 'END')
        
        return self.calculate_metrics()
    
    def calculate_metrics(self) -> Dict:
        """
        Calculate performance metrics
        """
        if not self.trades:
            return {
                'total_trades': 0,
                'final_equity': self.equity,
                'total_return_pct': (self.equity - self.initial_capital) / self.initial_capital * 100,
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown_pct': 0,
                'sharpe_ratio': 0
            }
        
        trades_df = pd.DataFrame(self.trades)
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        total_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        total_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        equity_series = pd.Series([e['equity'] for e in self.equity_curve])
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax * 100
        max_drawdown = abs(drawdown.min())
        
        returns = equity_series.pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252 * 96)) if returns.std() > 0 else 0  # 96 = 15m candles per day
        
        metrics = {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trades) * 100,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'profit_factor': profit_factor,
            'avg_win': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
            'largest_win': winning_trades['pnl'].max() if len(winning_trades) > 0 else 0,
            'largest_loss': losing_trades['pnl'].min() if len(losing_trades) > 0 else 0,
            'avg_duration_min': trades_df['duration'].mean(),
            'final_equity': self.equity,
            'total_return': self.equity - self.initial_capital,
            'total_return_pct': (self.equity - self.initial_capital) / self.initial_capital * 100,
            'max_drawdown_pct': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'trades_per_symbol': trades_df['symbol'].value_counts().to_dict()
        }
        
        return metrics
    
    def plot_equity_curve(self) -> go.Figure:
        """
        Plot equity curve with drawdown
        """
        if not self.equity_curve:
            return go.Figure()
        
        equity_df = pd.DataFrame(self.equity_curve)
        
        fig = make_subplots(rows=2, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.05,
                           row_heights=[0.7, 0.3],
                           subplot_titles=('Equity Curve (Flip-Flop Reversal Strategy)', 'Drawdown %'))
        
        fig.add_trace(
            go.Scatter(x=equity_df['timestamp'], y=equity_df['equity'],
                      mode='lines', name='Equity',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        cummax = equity_df['equity'].cummax()
        drawdown_pct = (equity_df['equity'] - cummax) / cummax * 100
        
        fig.add_trace(
            go.Scatter(x=equity_df['timestamp'], y=drawdown_pct,
                      mode='lines', name='Drawdown',
                      fill='tozeroy',
                      line=dict(color='red', width=1)),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text='Time', row=2, col=1)
        fig.update_yaxes(title_text='Equity (USDT)', row=1, col=1)
        fig.update_yaxes(title_text='Drawdown %', row=2, col=1)
        
        fig.update_layout(height=600, showlegend=True, hovermode='x unified')
        
        return fig
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """
        Return trades as DataFrame
        """
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trades)