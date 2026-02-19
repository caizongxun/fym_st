"""策略A: 純均值回歸 - 高勝率低回撤"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

from models.ml_range_bound_strategy import MLRangeBoundStrategy
from backtesting.tick_level_engine import TickLevelBacktestEngine
from data.binance_loader import BinanceDataLoader


def render_strategy_a_tab(loader, symbol_selector):
    st.header("策略 A: 純均值回歸 (高勝率)")
    
    st.info("""
    **純均值回歸策略**:
    
    目標: 勝率 70%+ | 最大回撤 <25%
    
    原理:
    - 只在BB極端位置進場 (下軌-3%或上軌+3%)
    - 超短持倉: 目標 1.2 ATR 就離場
    - 超緊止損: 0.6 ATR
    - 小仓位高頻率: 每筆 50% 仓位
    - 盈虧比 0.5-1.0 但勝率高
    """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**數據**")
        symbol_list = symbol_selector("strategy_a", multi=False)
        symbol = symbol_list[0]
        train_days = st.slider("訓練天數", 60, 180, 90, key="train")
        test_days = st.slider("回測天數", 7, 60, 30, key="test")
    
    with col2:
        st.markdown("**交易設定**")
        capital = st.number_input("資金 (USDT)", 1000.0, 100000.0, 10000.0, 1000.0, key="cap")
        leverage = st.slider("槓桿倍數", 3, 8, 5, key="lev", help="中等槓桿")
        position_pct = st.slider("每筆仓位%", 20, 80, 50, 10, key="pos", help="小仓位")
    
    with col3:
        st.markdown("**技術參數**")
        bb_period = st.number_input("BB週期", 10, 50, 20, key="bb")
        adx_max = st.slider("ADX最大值", 15, 35, 25, key="adx", help="盤整")
        extreme_pct = st.slider("極端%", 1.0, 5.0, 3.0, 0.5, key="ext", help="超越 BB軌")
    
    with st.expander("進階: 風險參數"):
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.markdown("**止損**")
            sl_atr = st.slider("止損 ATR", 0.3, 1.0, 0.6, 0.1, key="sl", help="超緊止損")
            confidence = st.slider("信心度", 0.3, 0.7, 0.50, 0.05, key="conf")
        
        with col_a2:
            st.markdown("**止盈**")
            tp_atr = st.slider("止盈 ATR", 0.8, 2.0, 1.2, 0.2, key="tp", help="超短持倉")
            use_dynamic_tp = st.checkbox("動態止盈", value=True, key="dyn_tp",
                                        help="到BB中軌50%先平半倉")
    
    st.markdown("---")
    
    if st.button("執行高勝率回測", type="primary", use_container_width=True):
        prog = st.progress(0)
        stat = st.empty()
        
        try:
            # Load
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
            
            # Train
            stat.text("2/4: 訓練...")
            strategy = MLRangeBoundStrategy(
                bb_period=bb_period,
                bb_std=2.0,
                adx_period=14,
                adx_threshold=adx_max
            )
            stats = strategy.train(df_train, forward_bars=10)
            st.success(f"L:{stats['long_samples']} S:{stats['short_samples']}")
            prog.progress(50)
            
            # Signals
            stat.text("3/4: 生成信號...")
            df_test = strategy.add_indicators(df_test)
            
            signals = []
            partial_exits = 0
            
            for i in range(50, len(df_test)):
                lp, sp = strategy.predict(df_test, i)
                r = df_test.iloc[i]
                
                sig = 0
                sl = np.nan
                tp = np.nan
                tp_partial = np.nan
                
                # 只在極端位置進場
                dist_to_lower = (r['bb_lower'] - r['close']) / r['close'] * 100
                dist_to_upper = (r['close'] - r['bb_upper']) / r['close'] * 100
                
                # 盤整確認
                is_ranging = r['adx'] < adx_max
                
                # 做多: 價格低bb下軌之下
                if lp > confidence and dist_to_lower > extreme_pct and is_ranging:
                    sig = 1
                    entry = r['close']
                    atr = r['atr']
                    
                    sl = entry - sl_atr * atr
                    tp = entry + tp_atr * atr
                    
                    # 動態止盈: 50%到BB中軌
                    if use_dynamic_tp:
                        mid_dist = r['bb_mid'] - entry
                        if mid_dist > 0 and mid_dist < tp_atr * atr:
                            tp_partial = r['bb_mid']
                            partial_exits += 1
                
                # 做空: 價格在BB上軌之上
                elif sp > confidence and dist_to_upper > extreme_pct and is_ranging:
                    sig = -1
                    entry = r['close']
                    atr = r['atr']
                    
                    sl = entry + sl_atr * atr
                    tp = entry - tp_atr * atr
                    
                    if use_dynamic_tp:
                        mid_dist = entry - r['bb_mid']
                        if mid_dist > 0 and mid_dist < tp_atr * atr:
                            tp_partial = r['bb_mid']
                            partial_exits += 1
                
                signals.append({
                    'signal': sig,
                    'stop_loss': sl,
                    'take_profit': tp,
                    'take_profit_partial': tp_partial,
                    'position_size': position_pct / 100.0,
                    'long_proba': lp,
                    'short_proba': sp
                })
            
            signals = [{'signal': 0, 'stop_loss': np.nan, 'take_profit': np.nan, 'take_profit_partial': np.nan, 'position_size': 1.0, 'long_proba': 0, 'short_proba': 0}] * 50 + signals
            df_sig = pd.DataFrame(signals)
            
            cnt = (df_sig['signal'] != 0).sum()
            if cnt == 0:
                st.warning("無信號")
                st.info("建議: 降低極端%到 2.5% 或降低信心度到 0.45")
                return
            
            st.success(f"{cnt}信號 | 分批出場:{partial_exits}")
            prog.progress(70)
            
            # Backtest
            stat.text("4/4: 回測...")
            engine = TickLevelBacktestEngine(
                initial_capital=capital,
                leverage=leverage,
                fee_rate=0.0006,
                slippage_pct=0.02,
                ticks_per_candle=100
            )
            metrics = engine.run_backtest(df_test, df_sig)
            
            prog.progress(100)
            stat.text("完成!")
            
            # Results
            st.markdown("---")
            st.subheader("回測結果")
            
            c1, c2, c3, c4 = st.columns(4)
            
            pnl = metrics['final_equity'] - capital
            c1.metric("權益", f"${metrics['final_equity']:,.0f}", f"{pnl:+,.0f}")
            c1.metric("交易", metrics['total_trades'])
            
            ret = metrics['total_return_pct']
            monthly = ret * 30 / test_days
            c2.metric("總報酬", f"{ret:.1f}%")
            c2.metric("月化", f"{monthly:.1f}%")
            
            wr = metrics['win_rate']
            c3.metric("勝率", f"{wr:.1f}%", delta="目標>70%")
            pf = metrics['profit_factor']
            c3.metric("盈虧比", f"{pf:.2f}")
            
            dd = metrics['max_drawdown_pct']
            c4.metric("回撤", f"{dd:.1f}%", delta="目標<-25%")
            c4.metric("夏普", f"{metrics['sharpe_ratio']:.2f}")
            
            # Kelly analysis
            st.markdown("---")
            if pf > 0:
                avg_win_loss_ratio = (pf - 1) if pf > 1 else 0.5
                kelly_pct = wr/100 - (1 - wr/100) / avg_win_loss_ratio
                
                st.subheader("策略評估")
                
                col_e1, col_e2, col_e3 = st.columns(3)
                
                with col_e1:
                    if wr >= 70 and dd > -25:
                        st.success("✅ 達成目標: 勝率>70% & 回撤<25%")
                        st.balloons()
                    elif wr >= 65:
                        st.success("👍 良好: 勝率>65%")
                    elif wr >= 55:
                        st.warning("⚠️ 一般: 勝率>55%")
                    else:
                        st.error("❌ 不佳: 勝率<55%")
                
                with col_e2:
                    st.metric("Kelly%", f"{kelly_pct*100:.1f}%")
                    if kelly_pct > 0.1:
                        st.success("可交易 (Kelly>10%)")
                    elif kelly_pct > 0:
                        st.warning("謹慎 (Kelly<10%)")
                    else:
                        st.error("不建議 (Kelly<0)")
                
                with col_e3:
                    if monthly >= 50:
                        st.success(f"月化>{monthly:.0f}%")
                    elif monthly >= 30:
                        st.info(f"月化>{monthly:.0f}%")
                    else:
                        st.warning(f"月化={monthly:.0f}%")
            
            # Charts
            st.markdown("---")
            st.subheader("權益曲線")
            fig = engine.plot_equity_curve()
            st.plotly_chart(fig, use_container_width=True)
            
            # Trades
            trades = engine.get_trades_dataframe()
            if not trades.empty:
                st.markdown("---")
                st.subheader("交易統計")
                
                wins = trades[trades['pnl_usdt'] > 0]
                losses = trades[trades['pnl_usdt'] < 0]
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("獲利筆數", len(wins))
                c2.metric("虧損筆數", len(losses))
                c3.metric("平均獲利", f"${wins['pnl_usdt'].mean():.2f}" if len(wins)>0 else "$0")
                c4.metric("平均虧損", f"${losses['pnl_usdt'].mean():.2f}" if len(losses)>0 else "$0")
                
                st.markdown("**最近20筆交易**")
                disp = trades[['entry_time', 'direction', 'entry_price', 'exit_price', 'pnl_usdt', 'exit_reason']].tail(20)
                st.dataframe(disp, use_container_width=True)
                
                csv = trades.to_csv(index=False).encode('utf-8')
                st.download_button("CSV下載", csv, f"{symbol}_mean_reversion_{datetime.now():%Y%m%d_%H%M}.csv", "text/csv")
            
        except Exception as e:
            st.error(f"錯誤: {str(e)}")
            import traceback
            with st.expander("詳情"):
                st.code(traceback.format_exc())
