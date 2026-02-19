"""Range-Bound Strategy Backtest Engine"""

import pandas as pd
import numpy as np
from typing import Dict
import plotly.graph_objects as go


class RangeBoundBacktestEngine:
    def __init__(
        self,
        initial_capital: float = 10000.0,
        leverage: float = 1.0,
        fee_rate: float = 0.0006
    ):
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.fee_rate = fee_rate
        self.trades = []
        self.equity_curve = []
        
    def run_backtest(self, signals_dict: Dict[str, pd.DataFrame]) -> Dict:
        """Run backtest"""
        for symbol, df in signals_dict.items():
            balance = self.initial_capital
            position = None
            
            for i in range(50, len(df)):
                row = df.iloc[i]
                current_price = row['close']
                current_time = row.get('open_time', i)
                
                current_equity = balance
                if position:
                    if position['type'] == 'long':
                        pnl = (current_price - position['entry']) * position['size']
                    else:
                        pnl = (position['entry'] - current_price) * position['size']
                    current_equity += pnl
                
                self.equity_curve.append({
                    'time': current_time,
                    'equity': current_equity,
                    'balance': balance
                })
                
                if position:
                    hit_sl = ((position['type'] == 'long' and row['low'] <= position['sl']) or 
                             (position['type'] == 'short' and row['high'] >= position['sl']))
                    
                    hit_tp = ((position['type'] == 'long' and row['high'] >= position['tp']) or 
                             (position['type'] == 'short' and row['low'] <= position['tp']))
                    
                    if hit_sl:
                        exit_price = position['sl']
                        exit_reason = 'SL'
                    elif hit_tp:
                        exit_price = position['tp']
                        exit_reason = 'TP'
                    else:
                        continue
                    
                    if position['type'] == 'long':
                        pnl = (exit_price - position['entry']) * position['size']
                    else:
                        pnl = (position['entry'] - exit_price) * position['size']
                    
                    fee = exit_price * position['size'] * self.fee_rate
                    pnl -= fee
                    balance += pnl
                    
                    duration_min = ((current_time - position['entry_time']).total_seconds() / 60 
                                   if hasattr(current_time - position['entry_time'], 'total_seconds') else 0)
                    
                    self.trades.append({
                        'symbol': symbol,
                        'Entry Time': position['entry_time'],
                        'Exit Time': current_time,
                        'Direction': 'Long' if position['type'] == 'long' else 'Short',
                        'Entry Price': position['entry'],
                        'Exit Price': exit_price,
                        'PnL (USDT)': pnl,
                        'PnL %': f"{(pnl / (position['entry'] * position['size'] / self.leverage)) * 100:.2f}%",
                        'Exit Reason': exit_reason,
                        'Hold Time (min)': duration_min
                    })
                    
                    position = None
                    continue
                
                if position is None:
                    signal = row.get('signal', 0)
                    
                    if signal == 1:
                        entry_price = current_price
                        position_value = balance * self.leverage
                        position_size = position_value / entry_price
                        
                        open_fee = entry_price * position_size * self.fee_rate
                        balance -= open_fee
                        
                        position = {
                            'type': 'long',
                            'entry': entry_price,
                            'size': position_size,
                            'sl': row.get('stop_loss', entry_price * 0.98),
                            'tp': row.get('take_profit', entry_price * 1.02),
                            'entry_time': current_time
                        }
                    
                    elif signal == -1:
                        entry_price = current_price
                        position_value = balance * self.leverage
                        position_size = position_value / entry_price
                        
                        open_fee = entry_price * position_size * self.fee_rate
                        balance -= open_fee
                        
                        position = {
                            'type': 'short',
                            'entry': entry_price,
                            'size': position_size,
                            'sl': row.get('stop_loss', entry_price * 1.02),
                            'tp': row.get('take_profit', entry_price * 0.98),
                            'entry_time': current_time
                        }
            
            if position:
                final_price = df.iloc[-1]['close']
                if position['type'] == 'long':
                    pnl = (final_price - position['entry']) * position['size']
                else:
                    pnl = (position['entry'] - final_price) * position['size']
                
                fee = final_price * position['size'] * self.fee_rate
                pnl -= fee
                balance += pnl
        
        return self._calculate_metrics()
    
    def _calculate_metrics(self) -> Dict:
        """Calculate performance metrics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'final_equity': self.initial_capital,
                'total_return_pct': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown_pct': 0,
                'avg_duration_min': 0
            }
        
        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)
        
        winning_trades = trades_df[trades_df['PnL (USDT)'] > 0]
        losing_trades = trades_df[trades_df['PnL (USDT)'] < 0]
        
        win_rate = len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0
        
        total_profit = winning_trades['PnL (USDT)'].sum() if len(winning_trades) > 0 else 0
        total_loss = abs(losing_trades['PnL (USDT)'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        final_equity = equity_df['equity'].iloc[-1]
        total_return_pct = (final_equity / self.initial_capital - 1) * 100
        
        equity_series = equity_df['equity']
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        max_drawdown_pct = drawdown.min() * 100
        
        returns = equity_series.pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(96 * 365)) if returns.std() > 0 else 0
        
        avg_duration = trades_df['Hold Time (min)'].mean()
        
        return {
            'total_trades': len(trades_df),
            'win_rate': win_rate,
            'final_equity': final_equity,
            'total_return_pct': total_return_pct,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown_pct,
            'avg_duration_min': avg_duration
        }
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """Get trade details"""
        return pd.DataFrame(self.trades)
    
    def plot_equity_curve(self) -> go.Figure:
        """Plot equity curve"""
        equity_df = pd.DataFrame(self.equity_curve)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_df['time'],
            y=equity_df['equity'],
            mode='lines',
            name='Equity',
            line=dict(color='green', width=2)
        ))
        
        fig.add_hline(
            y=self.initial_capital,
            line_dash='dash',
            line_color='gray',
            annotation_text='Initial Capital'
        )
        
        fig.update_layout(
            title='Strategy C Equity Curve',
            xaxis_title='Time',
            yaxis_title='Equity (USDT)',
            hovermode='x unified',
            height=400
        )
        
        return fig
