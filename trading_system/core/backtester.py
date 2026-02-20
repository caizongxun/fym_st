import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, 
                 initial_capital: float = 10000.0,
                 taker_fee: float = 0.0006,
                 maker_fee: float = 0.0002,
                 slippage: float = 0.0005,
                 risk_per_trade: float = 0.01,
                 leverage: int = 10):
        """
        Args:
            initial_capital: 初始資金
            taker_fee: 幣安合約 Taker (市價) 費率 0.06%
            maker_fee: 幣安合約 Maker (限價) 費率 0.02%
            slippage: 滑點 0.05%
            risk_per_trade: 每筆交易風險比例 (0.01 = 1%)
            leverage: 槓桿倍數
        """
        self.initial_capital = initial_capital
        self.taker_fee = taker_fee
        self.maker_fee = maker_fee
        self.slippage = slippage
        self.risk_per_trade = risk_per_trade
        self.leverage = leverage
        logger.info(f"Backtester: Capital=${initial_capital}, Risk={risk_per_trade*100}%, Leverage={leverage}x")
    
    def run_backtest(self, 
                     signals: pd.DataFrame,
                     price_column: str = 'close',
                     tp_multiplier: float = 2.5,
                     sl_multiplier: float = 1.5,
                     direction: int = 1,
                     backtest_days: Optional[int] = None) -> Dict:
        """
        執行回測
        
        Args:
            direction: 1 為做多, -1 為做空
        """
        
        if backtest_days is not None:
            cutoff_time = signals['open_time'].max() - pd.Timedelta(days=backtest_days)
            signals = signals[signals['open_time'] >= cutoff_time].copy()
            logger.info(f"Backtesting last {backtest_days} days: {len(signals)} signals")
        else:
            logger.info(f"Backtesting all data: {len(signals)} signals")
        
        if len(signals) == 0:
            logger.warning("No signals in selected backtest period")
            return self._empty_stats()
        
        trades = []
        capital = self.initial_capital
        peak_capital = self.initial_capital
        
        for idx, row in signals.iterrows():
            atr = row['atr']
            if pd.isna(atr) or atr <= 0:
                continue
            
            # 1. 進場計算 (市價單)
            raw_entry = row[price_column]
            if direction == 1:
                entry_price = raw_entry * (1 + self.slippage)
            else:
                entry_price = raw_entry * (1 - self.slippage)
            
            # 2. 倉位計算 (使用槓桿限制)
            risk_amount = capital * self.risk_per_trade
            max_loss_per_unit = sl_multiplier * atr
            
            if max_loss_per_unit <= 0:
                continue
            
            position_units = risk_amount / max_loss_per_unit
            position_value = position_units * entry_price
            
            # 確保不會爆倉 (保證金不能超過當前資金)
            required_margin = position_value / self.leverage
            if required_margin > capital * 0.95:
                position_value = (capital * 0.95) * self.leverage
                position_units = position_value / entry_price
                risk_amount = position_units * max_loss_per_unit
            
            entry_commission = position_value * self.taker_fee
            
            # 3. TP / SL 設定
            if direction == 1:
                tp_price = entry_price + (tp_multiplier * atr)
                sl_price = entry_price - (sl_multiplier * atr)
            else:
                tp_price = entry_price - (tp_multiplier * atr)
                sl_price = entry_price + (sl_multiplier * atr)
            
            # 4. 尋找未來數據
            future_data = signals.loc[idx:].iloc[1:25]
            if len(future_data) == 0:
                continue
            
            exit_price = None
            exit_reason = None
            exit_bars = 0
            exit_fee_rate = self.taker_fee
            
            # 5. 模擬執行
            for i, future_row in enumerate(future_data.iterrows()):
                _, frow = future_row
                current_high = frow['high']
                current_low = frow['low']
                
                if direction == 1:
                    # 做多邏輯: 先檢查 SL (保守回測原則)
                    if current_low <= sl_price:
                        exit_price = sl_price * (1 - self.slippage)
                        exit_reason = 'SL'
                        exit_bars = i + 1
                        exit_fee_rate = self.taker_fee
                        break
                    elif current_high >= tp_price:
                        exit_price = tp_price  # 觸及止盈沒有滑點
                        exit_reason = 'TP'
                        exit_bars = i + 1
                        exit_fee_rate = self.maker_fee  # Maker 費率
                        break
                else:
                    # 做空邏輯
                    if current_high >= sl_price:
                        exit_price = sl_price * (1 + self.slippage)
                        exit_reason = 'SL'
                        exit_bars = i + 1
                        exit_fee_rate = self.taker_fee
                        break
                    elif current_low <= tp_price:
                        exit_price = tp_price
                        exit_reason = 'TP'
                        exit_bars = i + 1
                        exit_fee_rate = self.maker_fee
                        break
            
            # 6. 超時出場 (市價單)
            if exit_price is None:
                raw_exit = future_data.iloc[-1][price_column]
                if direction == 1:
                    exit_price = raw_exit * (1 - self.slippage)
                else:
                    exit_price = raw_exit * (1 + self.slippage)
                exit_reason = 'Timeout'
                exit_bars = len(future_data)
                exit_fee_rate = self.taker_fee
            
            # 7. 結算
            exit_value = position_units * exit_price
            exit_commission = exit_value * exit_fee_rate
            
            if direction == 1:
                pnl_before_fees = (exit_price - entry_price) * position_units
            else:
                pnl_before_fees = (entry_price - exit_price) * position_units
            
            pnl_dollar = pnl_before_fees - entry_commission - exit_commission
            pnl_pct = pnl_dollar / (position_value / self.leverage) if position_value > 0 else 0
            
            capital += pnl_dollar
            
            if capital > peak_capital:
                peak_capital = capital
            
            drawdown = (capital - peak_capital) / peak_capital
            
            trades.append({
                'entry_time': row['open_time'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'exit_bars': exit_bars,
                'direction': direction,
                'position_units': position_units,
                'position_value': position_value,
                'required_margin': position_value / self.leverage,
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
        logger.info(f"Backtest complete: {len(trades_df)} trades, Final: ${stats['final_capital']:.2f}, Return: {stats['total_return']*100:.2f}%")
        
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
        
        returns = trades_df['pnl_dollar'] / trades_df['required_margin']
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