import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime, timezone, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TAIPEI_TZ = timezone(timedelta(hours=8))

class LimitOrderBacktestEngine:
    """
    限價單回測引擎 (Limit Order Backtest)
    
    模擬真實的掛單交易:
    - 進場: Limit Order (等價格回調才成交)
    - 止盈: Limit Order (Maker 費率)
    - 止損: Stop Market Order (Taker 費率)
    """
    
    def __init__(self,
                 initial_capital: float = 1000.0,
                 leverage: float = 5.0,
                 position_size_pct: float = 0.2,
                 maker_fee: float = 0.0002,
                 taker_fee: float = 0.0006,
                 limit_fill_threshold: float = 0.0001):  # 限價單成交閾值
        
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.position_size_pct = position_size_pct
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.limit_fill_threshold = limit_fill_threshold
        
        self.equity = initial_capital
        self.peak_equity = initial_capital
        self.trades = []
        self.equity_curve = []
        self.open_positions = {}
        self.pending_orders = {}  # 掛單簿
    
    def can_limit_order_fill(self, limit_price: float, candle_high: float, 
                            candle_low: float, direction: str) -> bool:
        """
        判斷限價單是否成交
        
        邏輯:
        - LONG: 價格必須下跌到 limit_price 以下
        - SHORT: 價格必須上漨到 limit_price 以上
        - 並且要穿透一點點 (排隊成交機制)
        """
        if direction == 'LONG':
            # 做多: 希望價格下跌到 limit_price 時買入
            # 成交條件: candle_low <= limit_price * (1 + threshold)
            return candle_low <= limit_price * (1 + self.limit_fill_threshold)
        else:  # SHORT
            # 做空: 希望價格上漨到 limit_price 時賣出
            # 成交條件: candle_high >= limit_price * (1 - threshold)
            return candle_high >= limit_price * (1 - self.limit_fill_threshold)
    
    def place_limit_order(self, symbol: str, direction: str, limit_price: float,
                         tp_price: float, sl_price: float, timestamp: datetime,
                         signal_data: dict):
        """
        下限價單
        """
        self.pending_orders[symbol] = {
            'direction': direction,
            'limit_price': limit_price,
            'tp_price': tp_price,
            'sl_price': sl_price,
            'timestamp': timestamp,
            'signal_data': signal_data
        }
    
    def execute_limit_order(self, symbol: str, fill_price: float, timestamp: datetime) -> bool:
        """
        執行限價單成交
        """
        if symbol not in self.pending_orders:
            return False
        
        order = self.pending_orders[symbol]
        
        position_value = self.equity * self.position_size_pct * self.leverage
        quantity = position_value / fill_price
        margin = position_value / self.leverage
        
        # Maker 費率
        entry_fee = position_value * self.maker_fee
        
        if entry_fee > self.equity:
            del self.pending_orders[symbol]
            return False
        
        self.open_positions[symbol] = {
            'direction': order['direction'],
            'entry_price': fill_price,
            'quantity': quantity,
            'position_value': position_value,
            'margin': margin,
            'entry_fee': entry_fee,
            'entry_time': timestamp,
            'tp_price': order['tp_price'],
            'sl_price': order['sl_price'],
            'signal_data': order['signal_data']
        }
        
        self.equity -= entry_fee
        del self.pending_orders[symbol]
        
        return True
    
    def check_tp_sl(self, symbol: str, high: float, low: float) -> Optional[Tuple[str, float]]:
        """
        檢查止盈止損
        """
        if symbol not in self.open_positions:
            return None
        
        pos = self.open_positions[symbol]
        
        if pos['direction'] == 'LONG':
            if low <= pos['sl_price']:
                return ('SL', pos['sl_price'])
            elif high >= pos['tp_price']:
                return ('TP', pos['tp_price'])
        else:  # SHORT
            if high >= pos['sl_price']:
                return ('SL', pos['sl_price'])
            elif low <= pos['tp_price']:
                return ('TP', pos['tp_price'])
        
        return None
    
    def close_position(self, symbol: str, exit_price: float, timestamp: datetime,
                      exit_reason: str) -> Optional[Dict]:
        """
        平倉
        """
        if symbol not in self.open_positions:
            return None
        
        pos = self.open_positions[symbol]
        
        # 止盈用Maker, 止損用Taker
        fee_rate = self.maker_fee if exit_reason == 'TP' else self.taker_fee
        
        exit_value = pos['quantity'] * exit_price
        exit_fee = exit_value * fee_rate
        
        if pos['direction'] == 'LONG':
            pnl = (exit_price - pos['entry_price']) * pos['quantity'] - pos['entry_fee'] - exit_fee
        else:
            pnl = (pos['entry_price'] - exit_price) * pos['quantity'] - pos['entry_fee'] - exit_fee
        
        self.equity += pnl
        
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        
        exit_reason_map = {'TP': '止盈', 'SL': '止損', 'END': '回測結束'}
        direction_map = {'LONG': '做多', 'SHORT': '做空'}
        
        entry_time_taipei = pos['entry_time'].astimezone(TAIPEI_TZ) if pos['entry_time'].tzinfo else pos['entry_time'].replace(tzinfo=timezone.utc).astimezone(TAIPEI_TZ)
        exit_time_taipei = timestamp.astimezone(TAIPEI_TZ) if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc).astimezone(TAIPEI_TZ)
        
        trade_record = {
            'symbol': symbol,
            'direction': pos['direction'],
            '方向': direction_map[pos['direction']],
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
            '持倉時長(分)': int((timestamp - pos['entry_time']).total_seconds() / 60)
        }
        
        self.trades.append(trade_record)
        del self.open_positions[symbol]
        
        return trade_record
    
    def run_backtest(self, signals_dict: Dict[str, pd.DataFrame]) -> Dict:
        """
        執行回測
        """
        all_data = []
        for symbol, df in signals_dict.items():
            df_copy = df.copy()
            df_copy['symbol'] = symbol
            all_data.append(df_copy)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        if 'open_time' not in combined_df.columns and 'time' in combined_df.columns:
            combined_df['open_time'] = pd.to_datetime(combined_df['time'])
        
        combined_df = combined_df.sort_values('open_time').reset_index(drop=True)
        
        for idx, row in combined_df.iterrows():
            timestamp = row['open_time']
            symbol = row['symbol']
            current_high = row['high']
            current_low = row['low']
            current_close = row['close']
            
            # 1. 檢查掛單是否成交
            if symbol in self.pending_orders:
                order = self.pending_orders[symbol]
                if self.can_limit_order_fill(order['limit_price'], current_high, 
                                            current_low, order['direction']):
                    self.execute_limit_order(symbol, order['limit_price'], timestamp)
            
            # 2. 檢查止盈止損
            if symbol in self.open_positions:
                tp_sl_result = self.check_tp_sl(symbol, current_high, current_low)
                if tp_sl_result:
                    exit_reason, exit_price = tp_sl_result
                    self.close_position(symbol, exit_price, timestamp, exit_reason)
                    # 清除掛單
                    if symbol in self.pending_orders:
                        del self.pending_orders[symbol]
                    continue
            
            # 3. 處理新信號
            if 'signal' in row and row['signal'] != 0:
                direction = 'LONG' if row['signal'] == 1 else 'SHORT'
                limit_price = row.get('limit_price', current_close)
                tp_price = row.get('tp_price', 0)
                sl_price = row.get('sl_price', 0)
                confidence = row.get('confidence', 0)
                
                # 如果已有掛單或持倉，先取消/平倉
                if symbol in self.pending_orders:
                    del self.pending_orders[symbol]
                
                if symbol in self.open_positions:
                    self.close_position(symbol, current_close, timestamp, 'REVERSAL')
                
                # 下新掛單
                self.place_limit_order(
                    symbol, direction, limit_price, tp_price, sl_price,
                    timestamp, {'confidence': confidence}
                )
            
            # 4. 記錄權益
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': self.equity,
                'open_positions': len(self.open_positions),
                'pending_orders': len(self.pending_orders)
            })
        
        # 回測結束，平所有倉位
        for symbol in list(self.open_positions.keys()):
            last_row = combined_df[combined_df['symbol'] == symbol].iloc[-1]
            self.close_position(symbol, last_row['close'], last_row['open_time'], 'END')
        
        return self.calculate_metrics()
    
    def calculate_metrics(self) -> Dict:
        """
        計算績效指標
        """
        if not self.trades:
            return {
                'total_trades': 0,
                'final_equity': self.equity,
                'total_return_pct': (self.equity - self.initial_capital) / self.initial_capital * 100
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
            'avg_duration_min': trades_df['duration'].mean(),
            'final_equity': self.equity,
            'total_return': self.equity - self.initial_capital,
            'total_return_pct': (self.equity - self.initial_capital) / self.initial_capital * 100,
            'max_drawdown_pct': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }
        
        return metrics
    
    def plot_equity_curve(self) -> go.Figure:
        """
        繪製權益曲線
        """
        if not self.equity_curve:
            return go.Figure()
        
        equity_df = pd.DataFrame(self.equity_curve)
        
        fig = make_subplots(rows=2, cols=1,
                           shared_xaxes=True,
                           vertical_spacing=0.05,
                           row_heights=[0.7, 0.3],
                           subplot_titles=('權益曲線 (Limit Order)', '回撤 %'))
        
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

if __name__ == '__main__':
    print("Limit Order Backtest Engine")
    print("請在App中使用")