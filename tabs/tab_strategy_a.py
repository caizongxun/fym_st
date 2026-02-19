"""策略A: EMA交叉 - 經典策略"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

from models.ml_range_bound_strategy import MLRangeBoundStrategy
from backtesting.tick_level_engine import TickLevelBacktestEngine
from data.binance_loader import BinanceDataLoader


def calculate_ema(df, periods=[9, 21, 50]):
    df = df.copy()
    for p in periods:
        df[f'ema_{p}'] = df['close'].ewm(span=p, adjust=False).mean()
    return df


def render_strategy_a_tab(loader, symbol_selector):
    st.header("策略 A: EMA交叉 (經典)")
    
    st.info("""
    **EMA交叉策略** - 最簡單最有效:
    
    做多: EMA9 向上突破 EMA21
    做空: EMA9 向下突破 EMA21
    趨勢過濾: EMA21 vs EMA50
    
    止損: 2 ATR
    止盈: 4 ATR (2:1)
    
    無ML,純技術指標
    """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**數據**")
        symbol_list = symbol_selector("strategy_a", multi=False)
        symbol = symbol_list[0]
        test_days = st.slider("回測天數", 7, 60, 30, key="test")
    
    with col2:
        st.markdown("**交易**")
        capital = st.number_input("資金", 1000.0, 100000.0, 10000.0, 1000.0, key="cap")
        leverage = st.slider("槓桿", 3, 10, 5, key="lev")
        position_pct = st.slider("仓位%", 50, 100, 80, 10, key="pos")
    
    with col3:
        st.markdown("**風控**")
        sl_atr = st.slider("止損 ATR", 1.0, 3.0, 2.0, 0.5, key="sl")
        tp_atr = st.slider("止盈 ATR", 2.0, 6.0, 4.0, 0.5, key="tp")
        use_trend_filter = st.checkbox("趨勢過濾", value=True, key="trend")
    
    st.markdown("---")
    
    if st.button("執行EMA策略", type="primary", use_container_width=True):
        prog = st.progress(0)
        stat = st.empty()
        
        try:
            stat.text("載入...")
            prog.progress(20)
            
            if isinstance(loader, BinanceDataLoader):
                end = datetime.now()
                start = end - timedelta(days=test_days + 10)
                df_test = loader.load_historical_data(symbol, '15m', start, end)
            else:
                df_test = loader.load_klines(symbol, '15m')
                df_test = df_test.tail((test_days + 10) * 96)
            
            st.success(f"{len(df_test)}根")
            prog.progress(40)
            
            stat.text("計算EMA...")
            df_test = calculate_ema(df_test, periods=[9, 21, 50])
            
            # ATR
            df_test['tr'] = np.maximum(
                df_test['high'] - df_test['low'],
                np.maximum(
                    abs(df_test['high'] - df_test['close'].shift(1)),
                    abs(df_test['low'] - df_test['close'].shift(1))
                )
            )
            df_test['atr'] = df_test['tr'].rolling(window=14).mean()
            
            st.success("EMA完成")
            prog.progress(60)
            
            stat.text("生成信號...")
            
            signals = []
            long_count = 0
            short_count = 0
            
            for i in range(50, len(df_test)):
                r = df_test.iloc[i]
                prev = df_test.iloc[i-1]
                
                sig = 0
                sl = np.nan
                tp = np.nan
                
                # EMA交叉
                cross_up = prev['ema_9'] <= prev['ema_21'] and r['ema_9'] > r['ema_21']
                cross_down = prev['ema_9'] >= prev['ema_21'] and r['ema_9'] < r['ema_21']
                
                # 趨勢過濾
                if use_trend_filter:
                    trend_up = r['ema_21'] > r['ema_50']
                    trend_down = r['ema_21'] < r['ema_50']
                else:
                    trend_up = True
                    trend_down = True
                
                # 做多
                if cross_up and trend_up:
                    sig = 1
                    entry = r['close']
                    atr = r['atr']
                    sl = entry - sl_atr * atr
                    tp = entry + tp_atr * atr
                    long_count += 1
                
                # 做空
                elif cross_down and trend_down:
                    sig = -1
                    entry = r['close']
                    atr = r['atr']
                    sl = entry + sl_atr * atr
                    tp = entry - tp_atr * atr
                    short_count += 1
                
                signals.append({
                    'signal': sig,
                    'stop_loss': sl,
                    'take_profit': tp,
                    'position_size': position_pct / 100.0
                })
            
            signals = [{'signal': 0, 'stop_loss': np.nan, 'take_profit': np.nan, 'position_size': 1.0}] * 50 + signals
            df_sig = pd.DataFrame(signals)
            
            cnt = (df_sig['signal'] != 0).sum()
            
            if cnt == 0:
                st.warning("無信號 - 嘗試關閉趨勢過濾")
                return
            
            st.success(f"{cnt}信號 (L:{long_count} S:{short_count})")
            prog.progress(80)
            
            stat.text("回測...")
            engine = TickLevelBacktestEngine(capital, leverage, 0.0006, 0.02, 100)
            metrics = engine.run_backtest(df_test, df_sig)
            
            prog.progress(100)
            stat.text("完成")
            
            # Results
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
            
            if wr >= 45 and pf >= 1.3 and dd > -35:
                st.success("✅ 策略可用")
                st.balloons()
            elif wr >= 40:
                st.info("⚠️ 還可以")
            else:
                st.error("❌ 需調整")
            
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
                c3.metric("平均贏", f"${wins['pnl_usdt'].mean():.2f}" if len(wins)>0 else "$0")
                c4.metric("平均輸", f"${losses['pnl_usdt'].mean():.2f}" if len(losses)>0 else "$0")
                
                st.dataframe(trades[['entry_time', 'direction', 'entry_price', 'exit_price', 'pnl_usdt', 'exit_reason']].tail(30), use_container_width=True)
                
                csv = trades.to_csv(index=False).encode('utf-8')
                st.download_button("CSV", csv, f"{symbol}_ema_{datetime.now():%Y%m%d_%H%M}.csv", "text/csv")
            
            # Suggestion
            st.markdown("---")
            st.subheader("建議")
            
            if wr < 40:
                st.warning("""
                勝率太低,建議:
                1. 關閉趨勢過濾 (增加信號)
                2. 減小止損到 1.5 ATR
                3. 增大止盈到 5 ATR
                """)
            
            if pf < 1.2:
                st.warning("""
                盈虧比低,建議:
                - 止盈距離拉遠 (5-6 ATR)
                - 或考慮分批止盈
                """)
            
            if metrics['total_trades'] < 20:
                st.info("交易次數少,可關閉趨勢過濾")
            
        except Exception as e:
            st.error(f"錯: {e}")
            import traceback
            with st.expander("詳情"):
                st.code(traceback.format_exc())
