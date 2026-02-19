"""策略C: 斐波那契回調策略"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from scipy.signal import argrelextrema

from backtesting.tick_level_engine import TickLevelBacktestEngine
from data.binance_loader import BinanceDataLoader


class FibonacciStrategy:
    """
    斐波那契回調策略
    
    原理:
    1. 識別波段高低點
    2. 計算斐波那契回調位
    3. 在關鍵位等待反轉確認
    4. 進場交易
    """
    
    def __init__(self, lookback=50):
        self.lookback = lookback
        self.fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        self.fib_ext = [1.272, 1.618, 2.0, 2.618]
    
    def find_swing_points(self, df, order=5):
        """識別波段高低點"""
        df = df.copy()
        
        # 使用scipy找極值
        highs = argrelextrema(df['high'].values, np.greater, order=order)[0]
        lows = argrelextrema(df['low'].values, np.less, order=order)[0]
        
        df['swing_high'] = 0
        df['swing_low'] = 0
        
        df.loc[highs, 'swing_high'] = 1
        df.loc[lows, 'swing_low'] = 1
        
        return df
    
    def identify_trend(self, df):
        """識別趨勢方向"""
        df = df.copy()
        
        # EMA趨勢
        df['ema20'] = df['close'].ewm(span=20).mean()
        df['ema50'] = df['close'].ewm(span=50).mean()
        
        df['trend'] = 0
        df.loc[df['ema20'] > df['ema50'], 'trend'] = 1  # 上升
        df.loc[df['ema20'] < df['ema50'], 'trend'] = -1  # 下降
        
        return df
    
    def calculate_fib_levels(self, high, low, is_uptrend=True):
        """計算斐波那契位"""
        diff = high - low
        
        if is_uptrend:
            # 上升趨勢: 從低到高
            levels = {
                'fib_0': low,
                'fib_236': low + diff * 0.236,
                'fib_382': low + diff * 0.382,
                'fib_500': low + diff * 0.5,
                'fib_618': low + diff * 0.618,
                'fib_786': low + diff * 0.786,
                'fib_100': high,
                # 擴展位
                'fib_1272': low + diff * 1.272,
                'fib_1618': low + diff * 1.618
            }
        else:
            # 下降趨勢: 從高到低
            levels = {
                'fib_0': high,
                'fib_236': high - diff * 0.236,
                'fib_382': high - diff * 0.382,
                'fib_500': high - diff * 0.5,
                'fib_618': high - diff * 0.618,
                'fib_786': high - diff * 0.786,
                'fib_100': low,
                # 擴展位
                'fib_1272': high - diff * 1.272,
                'fib_1618': high - diff * 1.618
            }
        
        return levels
    
    def find_recent_swing(self, df, current_idx, lookback=50):
        """找最近的波段"""
        start_idx = max(0, current_idx - lookback)
        window = df.iloc[start_idx:current_idx]
        
        # 找最近的高低點
        swing_highs = window[window['swing_high'] == 1]
        swing_lows = window[window['swing_low'] == 1]
        
        if len(swing_highs) == 0 or len(swing_lows) == 0:
            return None, None, None
        
        recent_high = swing_highs.iloc[-1]['high']
        recent_low = swing_lows.iloc[-1]['low']
        
        # 判斷趨勢: 看哪個更近
        high_idx = swing_highs.index[-1]
        low_idx = swing_lows.index[-1]
        
        is_uptrend = low_idx > high_idx  # 低點在後 = 上升趨勢
        
        return recent_high, recent_low, is_uptrend
    
    def check_fib_bounce(self, current_price, fib_levels, tolerance=0.005):
        """檢查是否在斐波位附近"""
        key_levels = ['fib_382', 'fib_500', 'fib_618']
        
        for level_name in key_levels:
            level_price = fib_levels[level_name]
            if abs(current_price - level_price) / level_price < tolerance:
                return True, level_name
        
        return False, None
    
    def check_reversal_confirmation(self, df, idx):
        """檢查反轉確認信號"""
        if idx < 2:
            return False
        
        current = df.iloc[idx]
        prev1 = df.iloc[idx-1]
        prev2 = df.iloc[idx-2]
        
        # 看漲反轉: 錘子線/看漲吞沒
        bullish_hammer = (current['close'] > current['open']) and \
                        (current['close'] - current['low']) > 2 * abs(current['close'] - current['open'])
        
        bullish_engulf = (current['close'] > current['open']) and \
                        (prev1['close'] < prev1['open']) and \
                        (current['close'] > prev1['open']) and \
                        (current['open'] < prev1['close'])
        
        # 看跌反轉: 流星線/看跌吞沒
        bearish_star = (current['close'] < current['open']) and \
                      (current['high'] - current['close']) > 2 * abs(current['close'] - current['open'])
        
        bearish_engulf = (current['close'] < current['open']) and \
                        (prev1['close'] > prev1['open']) and \
                        (current['close'] < prev1['open']) and \
                        (current['open'] > prev1['close'])
        
        return bullish_hammer or bullish_engulf, bearish_star or bearish_engulf
    
    def generate_signals(self, df):
        """生成交易信號"""
        df = self.find_swing_points(df)
        df = self.identify_trend(df)
        
        signals = []
        
        for i in range(self.lookback, len(df)):
            r = df.iloc[i]
            
            sig = 0
            reason = ""
            fib_level = None
            
            # 找最近波段
            high, low, is_uptrend = self.find_recent_swing(df, i, self.lookback)
            
            if high is None or low is None:
                signals.append({
                    'signal': 0,
                    'reason': '',
                    'fib_level': None,
                    'stop_loss': np.nan,
                    'take_profit': np.nan
                })
                continue
            
            # 計算斐波位
            fib_levels = self.calculate_fib_levels(high, low, is_uptrend)
            
            # 檢查是否在關鍵斐波位
            at_fib, level_name = self.check_fib_bounce(r['close'], fib_levels)
            
            if not at_fib:
                signals.append({
                    'signal': 0,
                    'reason': '',
                    'fib_level': None,
                    'stop_loss': np.nan,
                    'take_profit': np.nan
                })
                continue
            
            # 檢查反轉確認
            bullish_reversal, bearish_reversal = self.check_reversal_confirmation(df, i)
            
            # 做多信號: 上升趨勢 + 回調到斐波位 + 看漲反轉
            if is_uptrend and bullish_reversal and r['trend'] == 1:
                sig = 1
                reason = f"FIB_LONG_{level_name}"
                
                # 止損: 下一個斐波位或最低點
                if level_name == 'fib_618':
                    stop = fib_levels['fib_786']
                elif level_name == 'fib_500':
                    stop = fib_levels['fib_618']
                else:
                    stop = fib_levels['fib_500']
                
                # 止盈: 前高或擴展位
                take_profit = fib_levels['fib_1272']
                
            # 做空信號: 下降趨勢 + 反彈到斐波位 + 看跌反轉
            elif not is_uptrend and bearish_reversal and r['trend'] == -1:
                sig = -1
                reason = f"FIB_SHORT_{level_name}"
                
                # 止損: 上一個斐波位或最高點
                if level_name == 'fib_618':
                    stop = fib_levels['fib_786']
                elif level_name == 'fib_500':
                    stop = fib_levels['fib_618']
                else:
                    stop = fib_levels['fib_500']
                
                # 止盈: 前低或擴展位
                take_profit = fib_levels['fib_1272']
            
            else:
                stop = np.nan
                take_profit = np.nan
            
            signals.append({
                'signal': sig,
                'reason': reason,
                'fib_level': level_name if at_fib else None,
                'stop_loss': stop,
                'take_profit': take_profit
            })
        
        empty = [{'signal': 0, 'reason': '', 'fib_level': None, 'stop_loss': np.nan, 'take_profit': np.nan}] * self.lookback
        return pd.DataFrame(empty + signals)


def render_strategy_c_tab(loader, symbol_selector):
    st.header("策略 C: 斐波那契回調")
    
    st.info("""
    **Fibonacci Retracement Strategy**:
    
    核心邏輯:
    - 識別波段高低點
    - 計算斐波那契回調位 (38.2%, 50%, 61.8%)
    - 等待價格回調到關鍵位
    - 出現反轉確認信號時進場
    
    優勢:
    - 經典技術分析工具
    - 進場點精準
    - 止損止盈明確
    - 適合趨勢市場
    """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**數據**")
        symbol_list = symbol_selector("strategy_c", multi=False)
        symbol = symbol_list[0]
        test_days = st.slider("回測天數", 7, 60, 30, key="test_c")
    
    with col2:
        st.markdown("**交易**")
        capital = st.number_input("資金", 1000.0, 100000.0, 10000.0, 1000.0, key="cap_c")
        leverage = st.slider("槓桿", 3, 10, 5, key="lev_c")
        position_pct = st.slider("倉位%", 40, 100, 70, 10, key="pos_c")
    
    with col3:
        st.markdown("**斐波設定**")
        lookback = st.slider("波段回顧期", 20, 100, 50, 10, key="lookback_c")
        st.caption("識別最近N根K線的高低點")
        
        tolerance = st.slider("價格容差%", 0.3, 1.5, 0.5, 0.1, key="tol_c")
        st.caption("允許價格偏離斐波位的幅度")
    
    st.markdown("---")
    
    if st.button("執行斐波那契策略", type="primary", use_container_width=True):
        prog = st.progress(0)
        stat = st.empty()
        
        try:
            stat.text("載入...")
            prog.progress(10)
            
            if isinstance(loader, BinanceDataLoader):
                end = datetime.now()
                start = end - timedelta(days=test_days + 30)
                df_test = loader.load_historical_data(symbol, '15m', start, end)
            else:
                df_test = loader.load_klines(symbol, '15m')
                df_test = df_test.tail((test_days + 30) * 96)
            
            df_test = df_test.reset_index(drop=True)
            st.success(f"{len(df_test)}根")
            prog.progress(30)
            
            stat.text("計算斐波那契...")
            strategy = FibonacciStrategy(lookback=lookback)
            df_signals = strategy.generate_signals(df_test)
            
            signal_count = (df_signals['signal'] != 0).sum()
            
            if signal_count == 0:
                st.warning("無斐波信號 - 嘗試調整波段回顧期")
                return
            
            long_cnt = (df_signals['signal'] == 1).sum()
            short_cnt = (df_signals['signal'] == -1).sum()
            st.success(f"{signal_count}信號 (L:{long_cnt} S:{short_cnt})")
            prog.progress(60)
            
            # 設定倉位
            for i in range(len(df_signals)):
                if df_signals.iloc[i]['signal'] != 0:
                    df_signals.at[i, 'position_size'] = position_pct / 100.0
                else:
                    df_signals.at[i, 'position_size'] = 1.0
            
            stat.text("回測...")
            engine = TickLevelBacktestEngine(capital, leverage, 0.0006, 0.01, 100)
            metrics = engine.run_backtest(df_test, df_signals)
            
            prog.progress(100)
            stat.text("完成")
            
            st.markdown("---")
            st.subheader("結果")
            
            c1, c2, c3, c4 = st.columns(4)
            
            pnl = metrics['final_equity'] - capital
            c1.metric("權益", f"${metrics['final_equity']:,.0f}", f"{pnl:+,.0f}")
            c1.metric("交易", metrics['total_trades'])
            
            ret = metrics['total_return_pct']
            monthly = ret * 30 / test_days
            c2.metric("總報酬", f"{ret:.1f}%")
            c2.metric("月化", f"{monthly:.1f}%")
            
            wr = metrics['win_rate']
            c3.metric("勝率", f"{wr:.1f}%")
            pf = metrics['profit_factor']
            c3.metric("盈虧比", f"{pf:.2f}")
            
            dd = metrics['max_drawdown_pct']
            c4.metric("回撤", f"{dd:.1f}%")
            c4.metric("夏普", f"{metrics['sharpe_ratio']:.2f}")
            
            st.markdown("---")
            
            if wr >= 50 and pf >= 1.5:
                st.success("✅ 斐波那契策略有效!")
                st.balloons()
            elif ret > 0:
                st.info("⚠️ 有獲利")
            else:
                st.warning("❌ 調整參數")
            
            st.markdown("---")
            st.subheader("權益")
            fig = engine.plot_equity_curve()
            st.plotly_chart(fig, use_container_width=True)
            
            trades = engine.get_trades_dataframe()
            if not trades.empty:
                st.markdown("---")
                st.subheader("交易")
                
                wins = trades[trades['pnl_usdt'] > 0]
                losses = trades[trades['pnl_usdt'] < 0]
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("贏", len(wins))
                c2.metric("輸", len(losses))
                if len(wins) > 0:
                    c3.metric("平均贏", f"${wins['pnl_usdt'].mean():.2f}")
                if len(losses) > 0:
                    c4.metric("平均輸", f"${losses['pnl_usdt'].mean():.2f}")
                
                st.dataframe(trades[['entry_time', 'direction', 'entry_price', 'exit_price', 'pnl_usdt', 'exit_reason']].tail(30), use_container_width=True)
                
                csv = trades.to_csv(index=False).encode('utf-8')
                st.download_button("CSV", csv, f"{symbol}_fib_{datetime.now():%Y%m%d_%H%M}.csv", "text/csv")
            
        except Exception as e:
            st.error(f"錯: {e}")
            import traceback
            with st.expander("詳情"):
                st.code(traceback.format_exc())
