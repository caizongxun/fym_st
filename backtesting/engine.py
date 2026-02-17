import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timezone, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TAIPEI_TZ = timezone(timedelta(hours=8))

class BacktestEngine:
    """
    v2: FIXED Look-Ahead Bias
    - Signal detected at candle N close
    - Entry at candle N+1 open (realistic)
    - TP/SL checked using high/low (intra-candle)
    """
    
    def __init__(self, 
                 initial_capital: float = 100.0,
                 leverage: float = 10.0,
                 tp_atr_mult: float = 2.0,
                 sl_atr_mult: float = 2.0,
                 position_size_pct: float = 0.1,
                 position_mode: str = 'fixed',
                 max_positions: int = 1,
                 maker_fee: float = 0.0002,
                 taker_fee: float = 0.0006,
                 debug: bool = False):
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.tp_atr_mult = tp_atr_mult
        self.sl_atr_mult = sl_atr_mult
        self.position_size_pct = position_size_pct
        self.position_mode = position_mode
        self.max_positions = max_positions
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.debug = debug
        
        self.equity = initial_capital
        self.peak_equity = initial_capital
        self.trades = []
        self.equity_curve = []
        self.open_positions = {}
        self.pending_signals = {}  # NEW: Store signals for next candle entry
        
    def calculate_position_size(self) -> float:
        if self.position_mode == 'fixed':
            base_capital = self.initial_capital * self.position_size_pct
        else:
            base_capital = self.equity * self.position_size_pct
        position_value = base_capital * self.leverage
        return position_value
    
    def open_or_flip_position(self, symbol: str, direction: str, entry_price: float, 
                             timestamp: datetime, signal_data: dict, atr: float, 
                             trend_direction: int, trend_strength: float,
                             reversal_prob: float, trend_filter: str) -> bool:
        if self.equity <= 0:
            if self.debug:
                print(f"[SKIP] Equity <= 0: {self.equity}")
            return False
            
        if symbol in self.open_positions:
            old_pos = self.open_positions[symbol]
            if old_pos['direction'] != direction:
                self.close_position(symbol, entry_price, timestamp, 'REVERSAL_FLIP', 
                                   trend_direction, trend_strength, reversal_prob, trend_filter)
            else:
                if self.debug:
                    print(f"[SKIP] Position already exists: {old_pos['direction']}")
                return False
        
        position_value = self.calculate_position_size()
        required_margin = position_value / self.leverage
        if required_margin > self.equity * 0.95:
            if self.debug:
                print(f"[SKIP] Insufficient margin: required={required_margin:.2f}, equity={self.equity:.2f}")
            return False
        
        quantity = position_value / entry_price
        margin = position_value / self.leverage
        entry_fee = position_value * self.taker_fee
        
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
            'entry_trend': trend_direction,
            'entry_trend_strength': trend_strength,
            'entry_reversal_prob': reversal_prob,
            'entry_trend_filter': trend_filter
        }
        
        self.equity -= entry_fee
        
        if self.debug:
            print(f"[OPEN] {direction} @ {entry_price:.2f}, TP={tp_price:.2f}, SL={sl_price:.2f}, ATR={atr:.2f}")
        
        return True
    
    def check_tp_sl_intrabar(self, symbol: str, high: float, low: float, close: float, timestamp: datetime) -> Optional[Tuple[str, float]]:
        """
        Check TP/SL using high/low (more realistic)
        Returns: (exit_reason, exit_price) or None
        """
        if symbol not in self.open_positions:
            return None
        
        pos = self.open_positions[symbol]
        
        if pos['direction'] == 'LONG':
            # Check SL first (conservative)
            if low <= pos['sl_price']:
                return ('SL', pos['sl_price'])
            # Then check TP
            elif high >= pos['tp_price']:
                return ('TP', pos['tp_price'])
        else:  # SHORT
            # Check SL first
            if high >= pos['sl_price']:
                return ('SL', pos['sl_price'])
            # Then check TP
            elif low <= pos['tp_price']:
                return ('TP', pos['tp_price'])
        
        return None
    
    def close_position(self, symbol: str, exit_price: float, timestamp: datetime, 
                      exit_reason: str, trend_direction: int, trend_strength: float,
                      reversal_prob: float, trend_filter: str) -> Optional[Dict]:
        if symbol not in self.open_positions:
            return None
        
        pos = self.open_positions[symbol]
        
        exit_value = pos['quantity'] * exit_price
        exit_fee = exit_value * self.taker_fee
        
        if pos['direction'] == 'LONG':
            pnl = (exit_price - pos['entry_price']) * pos['quantity'] - pos['entry_fee'] - exit_fee
        else:
            pnl = (pos['entry_price'] - exit_price) * pos['quantity'] - pos['entry_fee'] - exit_fee
        
        self.equity += pnl
        
        if self.debug:
            print(f"[CLOSE] {pos['direction']} @ {exit_price:.2f}, PNL={pnl:.2f}, Reason={exit_reason}")
        
        exit_reason_map = {
            'TP': '止盈',
            'SL': '止損',
            'REVERSAL_FLIP': '反轉信號',
            'END': '回測結束'
        }
        
        trend_map = {1: '多頭', -1: '空頭', 0: '盤整'}
        direction_map = {'LONG': '做多', 'SHORT': '做空'}
        
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
            '損益(USDT)': pnl,
            'pnl_pct': (pnl / pos['margin']) * 100,
            '損益率': f"{(pnl / pos['margin']) * 100:.2f}%",
            'exit_reason': exit_reason,
            '離場原因': exit_reason_map.get(exit_reason, exit_reason),
            'duration': (timestamp - pos['entry_time']).total_seconds() / 60,
            '持倉時長(分)': int((timestamp - pos['entry_time']).total_seconds() / 60),
            'signal_data': pos['signal_data'],
            '進場趨勢': trend_map.get(pos.get('entry_trend', 0), '未知'),
            '離場趨勢': trend_map.get(trend_direction, '未知'),
            '進場趨勢強度': pos.get('entry_trend_strength', 0),
            '離場趨勢強度': trend_strength,
            '進場反轉機率': pos.get('entry_reversal_prob', 0),
            '離場反轉機率': reversal_prob,
            '進場過濾器': pos.get('entry_trend_filter', 'unknown'),
            '離場過濾器': trend_filter
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
        
        if self.debug:
            print(f"\n=== BACKTEST START ===")
            print(f"Total rows: {len(combined_df)}")
            print(f"Signals detected: {(combined_df['signal'] != 0).sum()}")
            print(f"First signal at index: {combined_df[combined_df['signal'] != 0].index[0] if (combined_df['signal'] != 0).sum() > 0 else 'None'}")
        
        # Add next candle's open price
        for symbol in signals_dict.keys():
            symbol_mask = combined_df['symbol'] == symbol
            combined_df.loc[symbol_mask, 'next_open'] = combined_df.loc[symbol_mask, 'open'].shift(-1)
        
        signal_count = 0
        entry_count = 0
        
        for idx, row in combined_df.iterrows():
            timestamp = row['open_time']
            symbol = row['symbol']
            current_open = row['open']
            current_high = row['high']
            current_low = row['low']
            current_close = row['close']
            next_open = row.get('next_open', np.nan)
            
            atr = row.get('15m_atr', row.get('atr', 0))
            trend_direction = int(row.get('trend_direction', 0))
            trend_strength = row.get('trend_strength_pred', 50)
            reversal_prob = row.get('reversal_prob_pred', 0)
            trend_filter = row.get('trend_filter', 'unknown')
            
            # STEP 1: Check if we have pending signal from previous candle
            if symbol in self.pending_signals:
                pending = self.pending_signals[symbol]
                direction = pending['direction']
                signal_atr = pending['atr']
                signal_data = pending['signal_data']
                signal_trend = pending['trend_direction']
                signal_strength = pending['trend_strength']
                signal_reversal = pending['reversal_prob']
                signal_filter = pending['trend_filter']
                
                # Enter at current candle's OPEN price
                result = self.open_or_flip_position(symbol, direction, current_open, timestamp, 
                                          signal_data, signal_atr, signal_trend, 
                                          signal_strength, signal_reversal, signal_filter)
                if result:
                    entry_count += 1
                
                del self.pending_signals[symbol]
            
            # STEP 2: Check TP/SL for existing positions using high/low
            if symbol in self.open_positions:
                tp_sl_result = self.check_tp_sl_intrabar(symbol, current_high, current_low, current_close, timestamp)
                
                if tp_sl_result:
                    exit_reason, exit_price = tp_sl_result
                    self.close_position(symbol, exit_price, timestamp, exit_reason, 
                                       trend_direction, trend_strength, reversal_prob, trend_filter)
                    # If stopped out, don't process new signal this candle
                    if symbol in self.pending_signals:
                        del self.pending_signals[symbol]
                    continue
            
            # STEP 3: Detect new signal at candle close
            if 'signal' in row and row['signal'] != 0 and not pd.isna(next_open):
                signal_count += 1
                direction = 'LONG' if row['signal'] == 1 else 'SHORT'
                
                if self.debug and signal_count <= 3:
                    print(f"\n[SIGNAL #{signal_count}] idx={idx}, time={timestamp}, direction={direction}")
                    print(f"  ATR={atr:.2f}, next_open={next_open:.2f}")
                
                if symbol in self.open_positions:
                    pos = self.open_positions[symbol]
                    if pos['direction'] != direction:
                        # Close at current close, prepare to reverse at next open
                        self.close_position(symbol, current_close, timestamp, 'REVERSAL_FLIP', 
                                           trend_direction, trend_strength, reversal_prob, trend_filter)
                        self.pending_signals[symbol] = {
                            'direction': direction,
                            'atr': atr,
                            'signal_data': {'reversal_prob': reversal_prob},
                            'trend_direction': trend_direction,
                            'trend_strength': trend_strength,
                            'reversal_prob': reversal_prob,
                            'trend_filter': trend_filter
                        }
                        if self.debug:
                            print(f"  -> Pending reversal to {direction}")
                else:
                    # No position, prepare new entry at next open
                    self.pending_signals[symbol] = {
                        'direction': direction,
                        'atr': atr,
                        'signal_data': {'reversal_prob': reversal_prob},
                        'trend_direction': trend_direction,
                        'trend_strength': trend_strength,
                        'reversal_prob': reversal_prob,
                        'trend_filter': trend_filter
                    }
                    if self.debug and signal_count <= 3:
                        print(f"  -> Pending entry {direction}")
            
            # Record equity
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': self.equity,
                'open_positions': len(self.open_positions)
            })
        
        if self.debug:
            print(f"\n=== BACKTEST END ===")
            print(f"Total signals detected: {signal_count}")
            print(f"Total entries executed: {entry_count}")
            print(f"Total trades completed: {len(self.trades)}")
        
        # Close remaining positions
        for symbol in list(self.open_positions.keys()):
            last_row = combined_df[combined_df['symbol'] == symbol].iloc[-1]
            last_price = last_row['close']
            last_time = last_row['open_time']
            last_trend = int(last_row.get('trend_direction', 0))
            last_strength = last_row.get('trend_strength_pred', 50)
            last_reversal = last_row.get('reversal_prob_pred', 0)
            last_filter = last_row.get('trend_filter', 'unknown')
            self.close_position(symbol, last_price, last_time, 'END', last_trend, 
                               last_strength, last_reversal, last_filter)
        
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
        
        stop_loss_trades = trades_df[trades_df['離場原因'] == '止損']
        fast_stops = stop_loss_trades[stop_loss_trades['持倉時長(分)'] < 60]
        
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
            'stop_loss_count': len(stop_loss_trades),
            'fast_stop_count': len(fast_stops),
            'fast_stop_pct': len(fast_stops) / len(stop_loss_trades) * 100 if len(stop_loss_trades) > 0 else 0,
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