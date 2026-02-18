import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class NextBarBacktestEngine:
    """
    下一根K棒雙向掛單回測引擎
    
    模擬雙向限價單交易:
    - 在預測低點掛做多限價單
    - 在預測高點掛做空限價單
    - 成交後設定止盈止損
    """
    
    def __init__(self,
                 initial_capital: float = 10000.0,
                 maker_fee: float = 0.0002,
                 taker_fee: float = 0.0004,
                 leverage: int = 1,
                 position_size_pct: float = 1.0):
        
        self.initial_capital = initial_capital
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.leverage = leverage
        self.position_size_pct = position_size_pct
        
        self.trades = []
        self.equity_curve = []
    
    def run_backtest(self, df: pd.DataFrame) -> Dict:
        """
        執行回測
        
        Args:
            df: 帶有 signal, entry_long, entry_short, tp, sl 的 DataFrame
        
        Returns:
            回測結果字典
        """
        capital = self.initial_capital
        position = None  # {'type': 'long'/'short', 'entry_price': float, 'tp': float, 'sl': float}
        
        for i in range(len(df) - 1):
            current = df.iloc[i]
            next_bar = df.iloc[i + 1]
            
            # 如果有持倉，檢查止盈止損
            if position is not None:
                position, capital = self._check_exit(
                    position, next_bar, capital
                )
            
            # 如果沒有持倉且有信號，嘗試進場
            if position is None and current['signal'] == 2:
                position, capital = self._try_entry(
                    current, next_bar, capital
                )
            
            # 記錄權益
            self.equity_curve.append({
                'time': next_bar.get('open_time', i+1),
                'equity': capital
            })
        
        # 強制平倉
        if position is not None:
            final_price = df.iloc[-1]['close']
            capital = self._close_position(position, final_price, capital, force=True)
        
        return self._calculate_metrics(capital)
    
    def _try_entry(self, current: pd.Series, next_bar: pd.Series, capital: float) -> Tuple:
        """
        嘗試雙向掛單進場
        
        檢查下一根K棒是否觸碰做多或做空進場價
        """
        entry_long = current['entry_long']
        entry_short = current['entry_short']
        
        next_high = next_bar['high']
        next_low = next_bar['low']
        next_open = next_bar['open']
        
        # 檢查做多進場
        long_triggered = next_low <= entry_long
        # 檢查做空進場
        short_triggered = next_high >= entry_short
        
        position = None
        
        # 情況 1: 只有做多成交
        if long_triggered and not short_triggered:
            position = {
                'type': 'long',
                'entry_price': entry_long,
                'tp': current['tp_long'],
                'sl': current['sl_long'],
                'entry_time': next_bar.get('open_time', 'unknown')
            }
            # 扣除 Maker 手續費
            capital *= (1 - self.maker_fee)
        
        # 情況 2: 只有做空成交
        elif short_triggered and not long_triggered:
            position = {
                'type': 'short',
                'entry_price': entry_short,
                'tp': current['tp_short'],
                'sl': current['sl_short'],
                'entry_time': next_bar.get('open_time', 'unknown')
            }
            capital *= (1 - self.maker_fee)
        
        # 情況 3: 雙邊都成交 (先成交哪個就持有哪個)
        elif long_triggered and short_triggered:
            # 使用 open 價判斷先觸碰哪個
            if abs(next_open - entry_long) < abs(next_open - entry_short):
                # 先觸碰做多
                position = {
                    'type': 'long',
                    'entry_price': entry_long,
                    'tp': current['tp_long'],
                    'sl': current['sl_long'],
                    'entry_time': next_bar.get('open_time', 'unknown')
                }
            else:
                # 先觸碰做空
                position = {
                    'type': 'short',
                    'entry_price': entry_short,
                    'tp': current['tp_short'],
                    'sl': current['sl_short'],
                    'entry_time': next_bar.get('open_time', 'unknown')
                }
            capital *= (1 - self.maker_fee)
        
        return position, capital
    
    def _check_exit(self, position: Dict, next_bar: pd.Series, capital: float) -> Tuple:
        """
        檢查止盈止損
        """
        next_high = next_bar['high']
        next_low = next_bar['low']
        
        if position['type'] == 'long':
            # 檢查止盈
            if next_high >= position['tp']:
                capital = self._close_position(position, position['tp'], capital, reason='TP')
                return None, capital
            # 檢查止損
            elif next_low <= position['sl']:
                capital = self._close_position(position, position['sl'], capital, reason='SL')
                return None, capital
        
        elif position['type'] == 'short':
            # 檢查止盈
            if next_low <= position['tp']:
                capital = self._close_position(position, position['tp'], capital, reason='TP')
                return None, capital
            # 檢查止損
            elif next_high >= position['sl']:
                capital = self._close_position(position, position['sl'], capital, reason='SL')
                return None, capital
        
        return position, capital
    
    def _close_position(self, position: Dict, exit_price: float, capital: float, 
                       reason: str = 'CLOSE', force: bool = False) -> float:
        """
        平倉
        """
        entry_price = position['entry_price']
        
        if position['type'] == 'long':
            pnl_pct = (exit_price - entry_price) / entry_price
        else:  # short
            pnl_pct = (entry_price - exit_price) / entry_price
        
        # 扣除出場手續費
        if reason == 'TP' or reason == 'SL':
            fee = self.maker_fee  # 限價單止盈止損
        else:
            fee = self.taker_fee  # 市價平倉
        
        capital *= (1 + pnl_pct - fee)
        
        # 記錄交易
        self.trades.append({
            'type': position['type'],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'entry_time': position.get('entry_time', 'unknown')
        })
        
        return capital
    
    def _calculate_metrics(self, final_capital: float) -> Dict:
        """
        計算回測指標
        """
        if len(self.trades) == 0:
            return {
                'total_trades': 0,
                'final_capital': final_capital,
                'total_return_pct': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_drawdown_pct': 0
            }
        
        trades_df = pd.DataFrame(self.trades)
        
        # 基本指標
        total_trades = len(trades_df)
        winning_trades = trades_df[trades_df['pnl_pct'] > 0]
        losing_trades = trades_df[trades_df['pnl_pct'] < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        avg_win = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0
        
        total_profit = winning_trades['pnl_pct'].sum() if len(winning_trades) > 0 else 0
        total_loss = abs(losing_trades['pnl_pct'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # 最大回撤
        equity_df = pd.DataFrame(self.equity_curve)
        if len(equity_df) > 0:
            equity_df['cummax'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
            max_drawdown_pct = equity_df['drawdown'].min()
        else:
            max_drawdown_pct = 0
        
        total_return_pct = (final_capital - self.initial_capital) / self.initial_capital
        
        return {
            'total_trades': total_trades,
            'final_capital': final_capital,
            'total_return_pct': total_return_pct,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown_pct': max_drawdown_pct,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'trades_df': trades_df,
            'equity_df': equity_df if len(self.equity_curve) > 0 else pd.DataFrame()
        }

if __name__ == '__main__':
    print("Next Bar Backtest Engine")
    print("請在App中使用")