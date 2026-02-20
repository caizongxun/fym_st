import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, 
                 initial_capital: float = 10000.0,
                 commission_rate: float = 0.001,
                 slippage: float = 0.0005,
                 risk_per_trade: float = 0.01):
        """
        Args:
            initial_capital: 初始資金
            commission_rate: 手續費率
            slippage: 滑點
            risk_per_trade: 每筆交易風險比例 (0.01 = 1%)
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.risk_per_trade = risk_per_trade
        logger.info(f"Backtester initialized: Capital=${initial_capital}, Risk={risk_per_trade*100}%, Commission={commission_rate*100}%, Slippage={slippage*100}%")
    
    def run_backtest(self, 
                     signals: pd.DataFrame,
                     price_column: str = 'close',
                     tp_multiplier: float = 2.5,
                     sl_multiplier: float = 1.5,
                     backtest_days: Optional[int] = None) -> Dict:
        """
        執行回測,使用 ATR 基礎的風險管理
        
        倉位計算:
        - 每筆風險金額 = 當前資金 × risk_per_trade
        - 每筆最大虧損 = sl_multiplier × ATR
        - 倉位大小 = 風險金額 / 最大虧損
        - 倉位金額 = 倉位大小 × 進場價
        """
        
        if backtest_days is not None:
            cutoff_time = signals['open_time'].max() - pd.Timedelta(days=backtest_days)
            signals = signals[signals['open_time'] >= cutoff_time].copy()
            logger.info(f"Backtesting last {backtest_days} days: {len(signals)} signals from {signals['open_time'].min()} to {signals['open_time'].max()}")
        else:
            logger.info(f"Backtesting all data: {len(signals)} signals")
        
        if len(signals) == 0:
            logger.warning("No signals in selected backtest period")
            return self._empty_stats()
        
        trades = []
        capital = self.initial_capital
        peak_capital = self.initial_capital
        
        for idx, row in signals.iterrows():
            entry_price = row[price_column] * (1 + self.slippage)
            atr = row['atr']
            
            if pd.isna(atr) or atr <= 0:
                continue
            
            # 關鍵: 根據 ATR 和風險比例計算倉位
            risk_amount = capital * self.risk_per_trade
            max_loss_per_unit = sl_multiplier * atr
            
            if max_loss_per_unit <= 0:
                continue
            
            # 計算能買多少單位 (例如 BTC 數量)
            position_units = risk_amount / max_loss_per_unit
            
            # 倉位總值
            position_value = position_units * entry_price
            
            # 限制最大倉位 (不超過當前資金的80%)
            max_position_value = capital * 0.8
            if position_value > max_position_value:
                position_value = max_position_value
                position_units = position_value / entry_price
            
            # 進場手續費
            entry_commission = position_value * self.commission_rate
            
            # 止盈止損價格
            tp_price = entry_price + (tp_multiplier * atr)
            sl_price = entry_price - (sl_multiplier * atr)
            
            # 尋找未來數據
            future_data = signals.loc[idx:].iloc[1:25]
            
            if len(future_data) == 0:
                continue
            
            exit_price = None
            exit_reason = None
            exit_bars = 0
            
            # 模擬交易執行
            for i, future_row in enumerate(future_data.iterrows()):
                _, frow = future_row
                current_high = frow['high']
                current_low = frow['low']
                current_close = frow[price_column]
                
                # 檢查止盈
                if current_high >= tp_price:
                    exit_price = tp_price * (1 - self.slippage)
                    exit_reason = 'TP'
                    exit_bars = i + 1
                    break
                # 檢查止損
                elif current_low <= sl_price:
                    exit_price = sl_price * (1 - self.slippage)
                    exit_reason = 'SL'
                    exit_bars = i + 1
                    break
            
            # 超時出場
            if exit_price is None:
                exit_price = future_data.iloc[-1][price_column] * (1 - self.slippage)
                exit_reason = 'Timeout'
                exit_bars = len(future_data)
            
            # 出場手續費
            exit_value = position_units * exit_price
            exit_commission = exit_value * self.commission_rate
            
            # 損益計算
            pnl_before_fees = (exit_price - entry_price) * position_units
            pnl_dollar = pnl_before_fees - entry_commission - exit_commission
            pnl_pct = pnl_dollar / position_value if position_value > 0 else 0
            
            # 更新資金
            capital += pnl_dollar
            
            # 記錄峰值
            if capital > peak_capital:
                peak_capital = capital
            
            # 計算回撤
            drawdown = (capital - peak_capital) / peak_capital
            
            trades.append({
                'entry_time': row['open_time'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'exit_bars': exit_bars,
                'position_units': position_units,
                'position_value': position_value,
                'risk_amount': risk_amount,
                'atr': atr,
                'pnl_pct': pnl_pct,
                'pnl_dollar': pnl_dollar,
                'pnl_before_fees': pnl_before_fees,
                'entry_commission': entry_commission,
                'exit_commission': exit_commission,
                'total_commission': entry_commission + exit_commission,
                'capital': capital,
                'peak_capital': peak_capital,
                'drawdown': drawdown
            })
        
        trades_df = pd.DataFrame(trades)
        
        if len(trades_df) == 0:
            logger.warning("No trades executed in backtest")
            return self._empty_stats()
        
        stats = self._calculate_statistics(trades_df)
        logger.info(f"Backtest complete: {len(trades_df)} trades, Final capital: ${stats['final_capital']:.2f}, Return: {stats['total_return']*100:.2f}%")
        
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
        total_commission = trades_df['total_commission'].sum()
        
        final_capital = trades_df.iloc[-1]['capital'] if len(trades_df) > 0 else self.initial_capital
        total_return = (final_capital - self.initial_capital) / self.initial_capital
        
        avg_win = trades_df[trades_df['pnl_dollar'] > 0]['pnl_dollar'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl_dollar'] <= 0]['pnl_dollar'].mean() if losing_trades > 0 else 0
        
        total_win = trades_df[trades_df['pnl_dollar'] > 0]['pnl_dollar'].sum() if winning_trades > 0 else 0
        total_loss = abs(trades_df[trades_df['pnl_dollar'] <= 0]['pnl_dollar'].sum()) if losing_trades > 0 else 0
        profit_factor = total_win / total_loss if total_loss > 0 else np.inf
        
        trades_df['cumulative_pnl'] = trades_df['pnl_dollar'].cumsum()
        trades_df['cumulative_return'] = (trades_df['capital'] - self.initial_capital) / self.initial_capital
        
        peak = self.initial_capital
        drawdowns = []
        for capital in trades_df['capital']:
            if capital > peak:
                peak = capital
            dd = (capital - peak) / peak
            drawdowns.append(dd)
        
        trades_df['drawdown_pct'] = drawdowns
        max_drawdown = min(drawdowns) if len(drawdowns) > 0 else 0
        
        returns = trades_df['pnl_dollar'] / trades_df['position_value']
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        avg_trade_duration = trades_df['exit_bars'].mean()
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_commission': total_commission,
            'net_pnl': total_pnl,
            'final_capital': final_capital,
            'total_return': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_trade_duration': avg_trade_duration,
            'total_win': total_win,
            'total_loss': total_loss
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
                'total_commission': 0,
                'net_pnl': 0,
                'final_capital': self.initial_capital,
                'total_return': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'avg_trade_duration': 0,
                'total_win': 0,
                'total_loss': 0
            }
        }