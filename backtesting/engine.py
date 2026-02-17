import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timezone, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Taipei timezone (UTC+8)
TAIPEI_TZ = timezone(timedelta(hours=8))

class BacktestEngine:
    """
    ATR-based flip-flop reversal backtesting
    Supports both FIXED and COMPOUND position sizing modes
    """
    
    def __init__(self, 
                 initial_capital: float = 100.0,
                 leverage: float = 10.0,
                 tp_atr_mult: float = 2.0,
                 sl_atr_mult: float = 1.5,
                 position_size_pct: float = 0.1,
                 position_mode: str = 'fixed',  # 'fixed' or 'compound'
                 max_positions: int = 1,
                 maker_fee: float = 0.0002,
                 taker_fee: float = 0.0006):
        """
        Args:
            position_mode: 
                - 'fixed': 固定使用初始資金的比例 (例: 100U * 20% = 20U)
                - 'compound': 複利模式,使用當前權益的比例 (例: 120U * 20% = 24U)
        """
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.tp_atr_mult = tp_atr_mult
        self.sl_atr_mult = sl_atr_mult
        self.position_size_pct = position_size_pct
        self.position_mode = position_mode
        self.max_positions = max_positions
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        
        self.equity = initial_capital
        self.peak_equity = initial_capital
        self.trades = []
        self.equity_curve = []
        self.open_positions = {}
        
    def calculate_position_size(self) -> float:
        """
        Calculate position size based on selected mode
        
        FIXED mode: Always use percentage of INITIAL capital (no compound)
        COMPOUND mode: Use percentage of CURRENT equity (with compound)
        """
        if self.position_mode == 'fixed':
            # 固定模式: 永遠使用初始資金的比例
            base_capital = self.initial_capital * self.position_size_pct
        else:
            # 複利模式: 使用當前權益的比例
            base_capital = self.equity * self.position_size_pct
        
        position_value = base_capital * self.leverage
        return position_value
    
    def open_or_flip_position(self, symbol: str, direction: str, entry_price: float, 
                             timestamp: datetime, signal_data: dict, atr: float, 
                             trend_direction: int) -> bool:
        # Check if equity is positive
        if self.equity <= 0:
            return False
            
        # Close existing opposite position
        if symbol in self.open_positions:
            old_pos = self.open_positions[symbol]
            if old_pos['direction'] != direction:
                self.close_position(symbol, entry_price, timestamp, 'REVERSAL_FLIP', trend_direction)
            else:
                return False
        
        # Calculate position size based on mode
        position_value = self.calculate_position_size()
        
        # Safety check: don't open if required margin exceeds equity
        required_margin = position_value / self.leverage
        if required_margin > self.equity * 0.95:  # Max 95% of equity as margin
            return False
        
        quantity = position_value / entry_price
        margin = position_value / self.leverage
        entry_fee = position_value * self.taker_fee
        
        # Calculate TP/SL
        if direction == 'LONG':
            tp_price = entry_price + (atr * self.tp_atr_mult)
            sl_price = entry_price - (atr * self.sl_atr_mult)
        else:
            tp_price = entry_price - (atr * self.tp_atr_mult)
            sl_price = entry_price + (atr * self.sl_atr_mult)
        
        self.open_positions[symbol] = {
            'direction': direction,
            'entry_price': entry_price,
            'quantity': quantity,
            'position_value': position_value,
            'margin': margin,
            'entry_fee': entry_fee,
            'entry_time': timestamp,
            'tp_price': tp_price,
            'sl_price': sl_price,
            'atr': atr,
            'signal_data': signal_data,
            'entry_trend': trend_direction
        }
        
        self.equity -= entry_fee
        return True
    
    def check_tp_sl(self, symbol: str, current_price: float, timestamp: datetime) -> Optional[str]:
        if symbol not in self.open_positions:
            return None
        
        pos = self.open_positions[symbol]
        
        if pos['direction'] == 'LONG':
            if current_price >= pos['tp_price']:
                return 'TP'
            elif current_price <= pos['sl_price']:
                return 'SL'
        else:
            if current_price <= pos['tp_price']:
                return 'TP'
            elif current_price >= pos['sl_price']:
                return 'SL'
        
        return None
    
    def close_position(self, symbol: str, exit_price: float, timestamp: datetime, 
                      exit_reason: str, trend_direction: int) -> Optional[Dict]:
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
        
        # Map exit reason to Chinese
        exit_reason_map = {
            'TP_HIT': '止盈',
            'SL_HIT': '止損',
            'REVERSAL_FLIP': '反轉信號',
            'END': '回測結束'
        }
        
        # Map trend to Chinese
        trend_map = {
            1: '多頭',
            -1: '空頭',
            0: '盤整'
        }
        
        # Map direction to Chinese
        direction_map = {
            'LONG': '做多',
            'SHORT': '做空'
        }
        
        # Convert to Taipei time
        entry_time_taipei = pos['entry_time'].astimezone(TAIPEI_TZ) if pos['entry_time'].tzinfo else pos['entry_time'].replace(tzinfo=timezone.utc).astimezone(TAIPEI_TZ)
        exit_time_taipei = timestamp.astimezone(TAIPEI_TZ) if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc).astimezone(TAIPEI_TZ)
        
        trade_record = {
            'symbol': symbol,
            'direction': pos['direction'],
            '方向': direction_map.get(pos['direction'], pos['direction']),
            'entry_time': pos['entry_time'],
            '進場時間': entry_time_taipei.strftime('%Y-%m-%d %H:%M:%S'),
            'exit_time': timestamp,
            '離場時間': exit_time_taipei.strftime('%Y-%m-%d %H:%M:%S'),
            'entry_price': pos['entry_price'],
            '進場價格': pos['entry_price'],
            'exit_price': exit_price,
            '離場價格': exit_price,
            'tp_price': pos['tp_price'],
            'sl_price': pos['sl_price'],
            'quantity': pos['quantity'],
            'position_value': pos['position_value'],
            '手續費': pos['entry_fee'] + exit_fee,
            'pnl': pnl,
            '損益(USDT)': pnl,  # NEW: Display PnL in USDT
            'pnl_pct': (pnl / pos['margin']) * 100,
            '損益率': f"{(pnl / pos['margin']) * 100:.2f}%",
            'exit_reason': exit_reason,
            '離場原因': exit_reason_map.get(exit_reason, exit_reason),
            'duration': (timestamp - pos['entry_time']).total_seconds() / 60,
            '持倉時長(分)': int((timestamp - pos['entry_time']).total_seconds() / 60),
            'signal_data': pos['signal_data'],
            '進場趨勢': trend_map.get(pos.get('entry_trend', 0), '未知'),
            '離場趨勢': trend_map.get(trend_direction, '未知')
        }
        
        self.trades.append(trade_record)
        del self.open_positions[symbol]
        
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        
        return trade_record
    
    def run_backtest(self, signals_dict: Dict[str, pd.DataFrame]) -> Dict:
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
            current_price = row['close']
            atr = row.get('15m_atr', row.get('atr', 0))
            trend_direction = int(row.get('trend_direction', 0))
            
            # Check TP/SL first
            if symbol in self.open_positions:
                tp_sl_result = self.check_tp_sl(symbol, current_price, timestamp)
                
                if tp_sl_result == 'TP':
                    self.close_position(symbol, current_price, timestamp, 'TP_HIT', trend_direction)
                    continue
                
                elif tp_sl_result == 'SL':
                    self.close_position(symbol, current_price, timestamp, 'SL_HIT', trend_direction)
                    continue
            
            # Check reversal signals
            if 'signal' in row and row['signal'] != 0:
                direction = 'LONG' if row['signal'] == 1 else 'SHORT'
                
                if symbol in self.open_positions:
                    pos = self.open_positions[symbol]
                    if pos['direction'] != direction:
                        self.close_position(symbol, current_price, timestamp, 'REVERSAL_FLIP', trend_direction)
                        self.open_or_flip_position(symbol, direction, current_price, timestamp, 
                                                   {'reversal_prob': row.get('reversal_prob_pred', 0)}, atr, trend_direction)
                else:
                    signal_data = {'reversal_prob': row.get('reversal_prob_pred', 0)}
                    self.open_or_flip_position(symbol, direction, current_price, timestamp, signal_data, atr, trend_direction)
            
            # Record equity
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': self.equity,
                'open_positions': len(self.open_positions)
            })
        
        # Close remaining positions
        for symbol in list(self.open_positions.keys()):
            last_row = combined_df[combined_df['symbol'] == symbol].iloc[-1]
            last_price = last_row['close']
            last_time = last_row['open_time']
            last_trend = int(last_row.get('trend_direction', 0))
            self.close_position(symbol, last_price, last_time, 'END', last_trend)
        
        return self.calculate_metrics()
    
    def calculate_metrics(self) -> Dict:
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
        sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252 * 96)) if returns.std() > 0 else 0
        
        exit_reasons = trades_df['離場原因'].value_counts().to_dict()
        
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
            'exit_reasons': exit_reasons,
            'trades_per_symbol': trades_df['symbol'].value_counts().to_dict()
        }
        
        return metrics
    
    def plot_equity_curve(self) -> go.Figure:
        if not self.equity_curve:
            return go.Figure()
        
        equity_df = pd.DataFrame(self.equity_curve)
        
        mode_text = '固定倉位' if self.position_mode == 'fixed' else '複利模式'
        
        fig = make_subplots(rows=2, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.05,
                           row_heights=[0.7, 0.3],
                           subplot_titles=(f'權益曲線 ({mode_text})', '回撤 %'))
        
        fig.add_trace(
            go.Scatter(x=equity_df['timestamp'], y=equity_df['equity'],
                      mode='lines', name='權益',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        cummax = equity_df['equity'].cummax()
        drawdown_pct = (equity_df['equity'] - cummax) / cummax * 100
        
        fig.add_trace(
            go.Scatter(x=equity_df['timestamp'], y=drawdown_pct,
                      mode='lines', name='回撤',
                      fill='tozeroy',
                      line=dict(color='red', width=1)),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text='時間', row=2, col=1)
        fig.update_yaxes(title_text='權益 (USDT)', row=1, col=1)
        fig.update_yaxes(title_text='回撤 %', row=2, col=1)
        
        fig.update_layout(height=600, showlegend=True, hovermode='x unified')
        
        return fig
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trades)