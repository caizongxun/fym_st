"""策略A: ML + BB均值回歸 - 最終版"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

from models.ml_range_bound_strategy import MLRangeBoundStrategy
from backtesting.tick_level_engine import TickLevelBacktestEngine
from data.binance_loader import BinanceDataLoader


def render_strategy_a_tab(loader, symbol_selector):
    st.header("策略 A: BB均值回歸策略")
    
    st.info("""
    **策略原理** (經典震盪策略):
    
    進場: 價格觸碰BB上/下軌 + ML確認
    止盈: BB中軌 (均值回歸)
    止損: 1.5 ATR
    優勢: 目標明確,勝率高
    """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**數據**")
        symbol_list = symbol_selector("strategy_a", multi=False)
        symbol = symbol_list[0]
        train_days = st.slider("訓練", 60, 180, 90, key="train")
        test_days = st.slider("回測", 7, 60, 30, key="test")
    
    with col2:
        st.markdown("**交易**")
        capital = st.number_input("資金", 1000.0, 100000.0, 10000.0, 1000.0, key="cap")
        leverage = st.slider("槓桿", 3, 10, 5, key="lev")
        threshold = st.slider("信心度", 0.3, 0.7, 0.45, 0.05, key="th")
    
    with col3:
        st.markdown("**參數**")
        bb_period = st.number_input("BB", 10, 50, 20, key="bb")
        adx_threshold = st.slider("ADX", 20, 40, 25, key="adx")
        sl_atr = st.slider("止損 ATR", 1.0, 3.0, 1.5, 0.1, key="sl")
    
    st.markdown("---")
    
    if st.button("執行", type="primary", use_container_width=True):
        prog = st.progress(0)
        stat = st.empty()
        
        try:
            stat.text("1/4: 載入...")
            prog.progress(10)
            
            if isinstance(loader, BinanceDataLoader):
                end = datetime.now()
                start = end - timedelta(days=train_days + test_days)
                df_all = loader.load_historical_data(symbol, '15m', start, end)
            else:
                df_all = loader.load_klines(symbol, '15m')
                df_all = df_all.tail((train_days + test_days) * 96)
            
            split = len(df_all) - test_days * 96
            df_train = df_all.iloc[:split].copy()
            df_test = df_all.iloc[split:].copy()
            
            st.success(f"{len(df_train)}+{len(df_test)}根")
            prog.progress(20)
            
            stat.text("2/4: 訓練...")
            strategy = MLRangeBoundStrategy(bb_period=bb_period, bb_std=2.0, adx_period=14, adx_threshold=adx_threshold)
            stats = strategy.train(df_train, forward_bars=10)
            st.success(f"L:{stats['long_samples']} S:{stats['short_samples']}")
            prog.progress(50)
            
            stat.text("3/4: 生成信號...")
            df_test = strategy.add_indicators(df_test)
            
            signals = []
            for i in range(50, len(df_test)):
                lp, sp = strategy.predict(df_test, i)
                r = df_test.iloc[i]
                
                sig = 0
                sl = np.nan
                tp = np.nan
                
                # 更寬鬆的進場條件
                near_lower = r['close'] <= r['bb_lower'] * 1.01
                near_upper = r['close'] >= r['bb_upper'] * 0.99
                ranging = r['adx'] < adx_threshold
                
                # 確保BB中軌距離足夠遠(有獲利空間)
                dist_to_mid = abs(r['close'] - r['bb_mid']) / r['close']
                
                if lp > threshold and near_lower and ranging and dist_to_mid > 0.008:
                    sig = 1
                    entry = r['close']
                    atr = r['atr']
                    sl = entry - sl_atr * atr
                    tp = r['bb_mid']  # 目標:均值
                    
                    # 確保盈虧比至少 1.5
                    risk = sl_atr * atr
                    reward = tp - entry
                    if reward / risk < 1.5:
                        sig = 0  # 放棄此交易
                
                elif sp > threshold and near_upper and ranging and dist_to_mid > 0.008:
                    sig = -1
                    entry = r['close']
                    atr = r['atr']
                    sl = entry + sl_atr * atr
                    tp = r['bb_mid']
                    
                    risk = sl_atr * atr
                    reward = entry - tp
                    if reward / risk < 1.5:
                        sig = 0
                
                signals.append({'signal': sig, 'stop_loss': sl, 'take_profit': tp, 'long_proba': lp, 'short_proba': sp})
            
            signals = [{'signal': 0, 'stop_loss': np.nan, 'take_profit': np.nan, 'long_proba': 0, 'short_proba': 0}] * 50 + signals
            df_sig = pd.DataFrame(signals)
            
            cnt = (df_sig['signal'] != 0).sum()
            if cnt == 0:
                st.warning("無信號")
                st.info("建議: 降低信心度到 0.4")
                return
            
            st.success(f"{cnt}個信號")
            prog.progress(70)
            
            stat.text("4/4: 回測...")
            engine = TickLevelBacktestEngine(capital, leverage, 0.0006, 0.02, 100)
            metrics = engine.run_backtest(df_test, df_sig)
            
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
            
            c3.metric("勝率", f"{metrics['win_rate']:.1f}%")
            c3.metric("盈虧比", f"{metrics['profit_factor']:.2f}")
            
            c4.metric("回撤", f"{metrics['max_drawdown_pct']:.1f}%")
            c4.metric("夏普", f"{metrics['sharpe_ratio']:.2f}")
            
            st.markdown("---")
            
            if monthly >= 80 and metrics['profit_factor'] >= 1.5:
                st.success("[優秀] 月化>80% 盈虧比>1.5")
                st.balloons()
            elif monthly >= 50:
                st.success("[良好] 月化>50%")
            elif monthly >= 20:
                st.warning("[普通] 月化>20%")
            else:
                st.error("[待優化]")
            
            if metrics['profit_factor'] < 1.2:
                st.warning("盈虧比偏低,建議提高止損ATR或降低信心度")
            
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
                c1.metric("獲利", len(wins))
                c2.metric("虧損", len(losses))
                c3.metric("平均獲利", f"${wins['pnl_usdt'].mean():.2f}" if len(wins) > 0 else "$0")
                c4.metric("平均虧損", f"${losses['pnl_usdt'].mean():.2f}" if len(losses) > 0 else "$0")
                
                st.dataframe(trades[['entry_time', 'direction', 'entry_price', 'exit_price', 'pnl_usdt', 'exit_reason']].tail(20), use_container_width=True)
                
                csv = trades.to_csv(index=False).encode('utf-8')
                st.download_button("CSV", csv, f"{symbol}_{datetime.now():%Y%m%d_%H%M}.csv", "text/csv")
            
        except Exception as e:
            st.error(f"錯誤: {str(e)}")
            import traceback
            with st.expander("詳情"):
                st.code(traceback.format_exc())
