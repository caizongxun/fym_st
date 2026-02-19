"""Tick-Level Backtest Engine - Realistic Intra-Bar Price Movement Simulation"""

import pandas as pd
import numpy as np
from typing import Dict, List
import plotly.graph_objects as go


class TickLevelBacktestEngine:
    """Simulates tick-level price movement within each candle for realistic backtest"""
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        leverage: float = 3.0,
        fee_rate: float = 0.0006,
        slippage_pct: float = 0.02,
        ticks_per_candle: int = 100
    ):
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.fee_rate = fee_rate
        self.slippage_pct = slippage_pct
        self.ticks_per_candle = ticks_per_candle
        
        self.trades = []
        self.equity_curve = []
    
    def simulate_intrabar_ticks(self, row: pd.Series) -> List[float]:
        """Simulate realistic intra-bar price movement using OHLC"""
        open_price = row['open']
        high_price = row['high']
        low_price = row['low']
        close_price = row['close']
        
        if close_price >= open_price:
            if high_price - open_price > close_price - low_price:
                ticks = self._generate_pattern(open_price, high_price, low_price, close_price, 'high_first')
            else:
                ticks = self._generate_pattern(open_price, high_price, low_price, close_price, 'low_first')
        else:
            if open_price - low_price > high_price - close_price:
                ticks = self._generate_pattern(open_price, high_price, low_price, close_price, 'low_first')
            else:
                ticks = self._generate_pattern(open_price, high_price, low_price, close_price, 'high_first')
        
        return ticks
    
    def _generate_pattern(self, o, h, l, c, pattern='high_first') -> List[float]:
        """Generate tick sequence"""
        n = self.ticks_per_candle
        
        if pattern == 'high_first':
            segment1 = np.linspace(o, h, n//4)
            segment2 = np.linspace(h, l, n//2)
            segment3 = np.linspace(l, c, n//4)
            ticks = np.concatenate([segment1, segment2, segment3])
        else:
            segment1 = np.linspace(o, l, n//4)
            segment2 = np.linspace(l, h, n//2)
            segment3 = np.linspace(h, c, n//4)
            ticks = np.concatenate([segment1, segment2, segment3])
        
        noise = np.random.normal(0, (h - l) * 0.01, len(ticks))
        ticks = ticks + noise
        ticks = np.clip(ticks, l, h)
        
        return ticks.tolist()
    
    def run_backtest(self, df: pd.DataFrame, signals: pd.DataFrame) -> Dict:
        """Run tick-level backtest with dynamic position sizing"""
        balance = self.initial_capital
        position = None
        
        for i in range(50, len(df)):
            row = df.iloc[i]
            signal_row = signals.iloc[i]
            current_time = row.get('open_time', i)
            
            ticks = self.simulate_intrabar_ticks(row)
            
            for tick_idx, tick_price in enumerate(ticks):
                current_equity = balance
                if position:
                    if position['type'] == 'long':
                        unrealized_pnl = (tick_price - position['entry']) * position['size']
                    else:
                        unrealized_pnl = (position['entry'] - tick_price) * position['size']
                    current_equity += unrealized_pnl
                
                if tick_idx == 0:
                    self.equity_curve.append({
                        'time': current_time,
                        'equity': current_equity,
                        'balance': balance
                    })
                
                if position:
                    hit_sl = ((position['type'] == 'long' and tick_price <= position['sl']) or
                             (position['type'] == 'short' and tick_price >= position['sl']))
                    
                    hit_tp = ((position['type'] == 'long' and tick_price >= position['tp']) or
                             (position['type'] == 'short' and tick_price <= position['tp']))
                    
                    if hit_sl:
                        exit_price = position['sl'] * (1 - self.slippage_pct / 100)
                        exit_reason = 'SL'
                    elif hit_tp:
                        exit_price = position['tp'] * (1 - self.slippage_pct / 100)
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
                    
                    self.trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'direction': 'Long' if position['type'] == 'long' else 'Short',
                        'entry_price': position['entry'],
                        'exit_price': exit_price,
                        'pnl_usdt': pnl,
                        'pnl_pct': (pnl / (position['entry'] * position['size'] / self.leverage)) * 100,
                        'exit_reason': exit_reason
                    })
                    
                    position = None
                    break
            
            if position is None:
                signal = signal_row.get('signal', 0)
                long_proba = signal_row.get('long_proba', 0)
                short_proba = signal_row.get('short_proba', 0)
                
                # Get dynamic position size
                position_size_pct = signal_row.get('position_size', 1.0)
                
                if signal == 1 and long_proba > 0.5:
                    entry_price = row['close'] * (1 + self.slippage_pct / 100)
                    position_value = balance * self.leverage * position_size_pct
                    position_size = position_value / entry_price
                    
                    open_fee = entry_price * position_size * self.fee_rate
                    balance -= open_fee
                    
                    position = {
                        'type': 'long',
                        'entry': entry_price,
                        'size': position_size,
                        'sl': signal_row.get('stop_loss', entry_price * 0.98),
                        'tp': signal_row.get('take_profit', entry_price * 1.02),
                        'entry_time': current_time
                    }
                
                elif signal == -1 and short_proba > 0.5:
                    entry_price = row['close'] * (1 - self.slippage_pct / 100)
                    position_value = balance * self.leverage * position_size_pct
                    position_size = position_value / entry_price
                    
                    open_fee = entry_price * position_size * self.fee_rate
                    balance -= open_fee
                    
                    position = {
                        'type': 'short',
                        'entry': entry_price,
                        'size': position_size,
                        'sl': signal_row.get('stop_loss', entry_price * 1.02),
                        'tp': signal_row.get('take_profit', entry_price * 0.98),
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
                'avg_pnl_per_trade': 0
            }
        
        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)
        
        winning_trades = trades_df[trades_df['pnl_usdt'] > 0]
        losing_trades = trades_df[trades_df['pnl_usdt'] < 0]
        
        win_rate = len(winning_trades) / len(trades_df) * 100
        
        total_profit = winning_trades['pnl_usdt'].sum() if len(winning_trades) > 0 else 0
        total_loss = abs(losing_trades['pnl_usdt'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        final_equity = equity_df['equity'].iloc[-1]
        total_return_pct = (final_equity / self.initial_capital - 1) * 100
        
        equity_series = equity_df['equity']
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        max_drawdown_pct = drawdown.min() * 100
        
        returns = equity_series.pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(96 * 365)) if returns.std() > 0 else 0
        
        avg_pnl = trades_df['pnl_usdt'].mean()
        
        return {
            'total_trades': len(trades_df),
            'win_rate': win_rate,
            'final_equity': final_equity,
            'total_return_pct': total_return_pct,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown_pct,
            'avg_pnl_per_trade': avg_pnl
        }
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """Get detailed trade records"""
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
            line=dict(color='blue', width=2)
        ))
        
        fig.add_hline(
            y=self.initial_capital,
            line_dash='dash',
            line_color='gray',
            annotation_text='Initial Capital'
        )
        
        fig.update_layout(
            title='Equity Curve (Tick-Level Simulation)',
            xaxis_title='Time',
            yaxis_title='Equity (USDT)',
            hovermode='x unified',
            height=500
        )
        
        return fig
