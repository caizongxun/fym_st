"""策略A: ML + BB混合止盈 - 穩健版"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

from models.ml_range_bound_strategy import MLRangeBoundStrategy
from backtesting.tick_level_engine import TickLevelBacktestEngine
from data.binance_loader import BinanceDataLoader


def render_strategy_a_tab(loader, symbol_selector):
    st.header("策略 A: 混合止盈策略")
    
    st.info("""
    **混合止盈策略**:
    
    進場: BB邊緣 + ML高機率 + 盤整
    止盈: BB中軌 OR 2.5x風險 (取較遠者)
    止損: 1.2 ATR
    優勢: 確保高盈虧比 + 動態仓位
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
        leverage = st.slider("槓桿", 3, 10, 6, key="lev")
        threshold = st.slider("信心度", 0.3, 0.7, 0.42, 0.02, key="th")
    
    with col3:
        st.markdown("**參數**")
        bb_period = st.number_input("BB", 10, 50, 20, key="bb")
        adx_max = st.slider("ADX", 20, 40, 28, key="adx")
        sl_atr = st.slider("止損", 0.8, 2.0, 1.2, 0.1, key="sl")
    
    with st.expander("進階設定"):
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.markdown("**止盈設定**")
            tp_multiplier = st.slider("止盈倍數", 1.5, 4.0, 2.5, 0.1, key="tp_mult",
                                     help="止盈 = 止損 * 此倍數")
            use_bb_mid = st.checkbox("使用BB中軌", value=True, key="use_mid",
                                    help="如果BB中軌更遠則使用")
        
        with col_a2:
            st.markdown("**仓位管理**")
            use_position_sizing = st.checkbox("動態仓位", value=True, key="pos_size",
                                             help="根據機率調整仓位")
            base_size = st.slider("基礎仓位%", 30, 100, 60, 10, key="base") if use_position_sizing else 100
            max_size = st.slider("最大仓位%", 50, 100, 100, 10, key="max") if use_position_sizing else 100
    
    st.markdown("---")
    
    if st.button("執行回測", type="primary", use_container_width=True):
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
            
            st.success(f"{len(df_train)}+{len(df_test)}")
            prog.progress(20)
            
            stat.text("2/4: 訓練...")
            strategy = MLRangeBoundStrategy(bb_period=bb_period, bb_std=2.0, adx_period=14, adx_threshold=adx_max)
            stats = strategy.train(df_train, forward_bars=10)
            st.success(f"L:{stats['long_samples']} S:{stats['short_samples']}")
            prog.progress(50)
            
            stat.text("3/4: 信號...")
            df_test = strategy.add_indicators(df_test)
            
            signals = []
            
            for i in range(50, len(df_test)):
                lp, sp = strategy.predict(df_test, i)
                r = df_test.iloc[i]
                
                sig = 0
                sl = np.nan
                tp = np.nan
                pos_size = 1.0
                
                near_lower = r['close'] <= r['bb_lower'] * 1.015
                near_upper = r['close'] >= r['bb_upper'] * 0.985
                ranging = r['adx'] < adx_max
                
                if lp > threshold and near_lower and ranging:
                    sig = 1
                    entry = r['close']
                    atr = r['atr']
                    
                    # 止損
                    sl = entry - sl_atr * atr
                    risk = sl_atr * atr
                    
                    # 止盈: 取較遠者
                    tp_fixed = entry + risk * tp_multiplier
                    tp_bb = r['bb_mid']
                    
                    if use_bb_mid:
                        tp = max(tp_fixed, tp_bb)  # 取較遠者
                    else:
                        tp = tp_fixed
                    
                    # 確認盈虧比
                    actual_reward = tp - entry
                    actual_pf = actual_reward / risk
                    
                    if actual_pf < 1.8:  # 強制最低 1.8
                        sig = 0
                    
                    # 動態仓位
                    if sig != 0 and use_position_sizing:
                        proba_norm = (lp - threshold) / (1.0 - threshold)
                        pos_pct = base_size + (max_size - base_size) * proba_norm
                        pos_size = pos_pct / 100.0
                
                elif sp > threshold and near_upper and ranging:
                    sig = -1
                    entry = r['close']
                    atr = r['atr']
                    
                    sl = entry + sl_atr * atr
                    risk = sl_atr * atr
                    
                    tp_fixed = entry - risk * tp_multiplier
                    tp_bb = r['bb_mid']
                    
                    if use_bb_mid:
                        tp = min(tp_fixed, tp_bb)
                    else:
                        tp = tp_fixed
                    
                    actual_reward = entry - tp
                    actual_pf = actual_reward / risk
                    
                    if actual_pf < 1.8:
                        sig = 0
                    
                    if sig != 0 and use_position_sizing:
                        proba_norm = (sp - threshold) / (1.0 - threshold)
                        pos_pct = base_size + (max_size - base_size) * proba_norm
                        pos_size = pos_pct / 100.0
                
                signals.append({
                    'signal': sig,
                    'stop_loss': sl,
                    'take_profit': tp,
                    'position_size': pos_size,
                    'long_proba': lp,
                    'short_proba': sp
                })
            
            signals = [{'signal': 0, 'stop_loss': np.nan, 'take_profit': np.nan, 'position_size': 1.0, 'long_proba': 0, 'short_proba': 0}] * 50 + signals
            df_sig = pd.DataFrame(signals)
            
            cnt = (df_sig['signal'] != 0).sum()
            if cnt == 0:
                st.warning("無信號 - 降低信心度到 0.35")
                return
            
            avg_pos = df_sig[df_sig['signal'] != 0]['position_size'].mean() * 100
            st.success(f"{cnt}信號 | 平均仓位:{avg_pos:.0f}%")
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
            pf = metrics['profit_factor']
            c3.metric("盈虧比", f"{pf:.2f}", delta="OK" if pf > 1.5 else "LOW")
            
            c4.metric("回撤", f"{metrics['max_drawdown_pct']:.1f}%")
            c4.metric("夏普", f"{metrics['sharpe_ratio']:.2f}")
            
            st.markdown("---")
            
            if monthly >= 80 and pf >= 1.5 and metrics['max_drawdown_pct'] > -35:
                st.success("[優] 月化>80% 盈虧比>1.5 回撤<35%")
                st.balloons()
            elif monthly >= 50:
                st.success("[良] 月化>50%")
            elif monthly >= 25:
                st.warning("[中] 月化>25%")
            else:
                st.error("[弱]")
            
            st.markdown("---")
            st.subheader("權益曲線")
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
                c3.metric("平均贏", f"${wins['pnl_usdt'].mean():.2f}" if len(wins) > 0 else "$0")
                c4.metric("平均輸", f"${losses['pnl_usdt'].mean():.2f}" if len(losses) > 0 else "$0")
                
                st.dataframe(trades[['entry_time', 'direction', 'entry_price', 'exit_price', 'pnl_usdt', 'exit_reason']].tail(20), use_container_width=True)
                
                csv = trades.to_csv(index=False).encode('utf-8')
                st.download_button("CSV", csv, f"{symbol}_{datetime.now():%Y%m%d_%H%M}.csv", "text/csv")
            
        except Exception as e:
            st.error(f"錯: {e}")
            import traceback
            with st.expander("詳情"):
                st.code(traceback.format_exc())
