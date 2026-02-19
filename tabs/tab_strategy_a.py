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
    
    目標: 勝率 65%+ | 最大回撤 <25%
    
    原理:
    - BB極端位置進場 (下軌-1.5%或上軌+1.5%)
    - 超短持倉: 1.2 ATR
    - 緊止損: 0.6 ATR
    - 小仓位: 50%
    - 盈虧比 2.0 靠高勝率獲利
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
        leverage = st.slider("槓桿", 3, 8, 5, key="lev")
        position_pct = st.slider("仓位%", 30, 80, 50, 10, key="pos")
    
    with col3:
        st.markdown("**參數**")
        bb_period = st.number_input("BB", 10, 50, 20, key="bb")
        adx_max = st.slider("ADX", 15, 35, 25, key="adx")
        extreme_pct = st.slider("極端%", 0.5, 3.0, 1.5, 0.5, key="ext")
    
    with st.expander("進階設定"):
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.markdown("**風險**")
            sl_atr = st.slider("止損 ATR", 0.3, 1.0, 0.6, 0.1, key="sl")
            confidence = st.slider("信心度", 0.25, 0.65, 0.45, 0.05, key="conf")
        
        with col_a2:
            st.markdown("**止盈**")
            tp_atr = st.slider("止盈 ATR", 0.8, 2.0, 1.2, 0.2, key="tp")
            use_bb_mid = st.checkbox("使用BB中軌", value=True, key="use_mid")
    
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
            
            st.success(f"{len(df_train)}+{len(df_test)}")
            prog.progress(20)
            
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
            
            stat.text("3/4: 信號...")
            df_test = strategy.add_indicators(df_test)
            
            signals = []
            rejected_not_extreme = 0
            rejected_not_ranging = 0
            rejected_low_prob = 0
            
            for i in range(50, len(df_test)):
                lp, sp = strategy.predict(df_test, i)
                r = df_test.iloc[i]
                
                sig = 0
                sl = np.nan
                tp = np.nan
                
                # 檢查是否在極端位置
                near_lower = r['close'] < r['bb_lower']
                near_upper = r['close'] > r['bb_upper']
                
                # 計算超出BB軌的程度
                if near_lower:
                    dist_pct = (r['bb_lower'] - r['close']) / r['close'] * 100
                    is_extreme_long = dist_pct >= extreme_pct
                else:
                    is_extreme_long = False
                    dist_pct = 0
                
                if near_upper:
                    dist_pct = (r['close'] - r['bb_upper']) / r['close'] * 100
                    is_extreme_short = dist_pct >= extreme_pct
                else:
                    is_extreme_short = False
                    dist_pct = 0
                
                # 盤整確認
                is_ranging = r['adx'] < adx_max
                
                # 做多
                if lp > confidence and is_extreme_long and is_ranging:
                    sig = 1
                    entry = r['close']
                    atr = r['atr']
                    
                    sl = entry - sl_atr * atr
                    
                    # 止盈: 取BB中軌或固定ATR較遠者
                    tp_fixed = entry + tp_atr * atr
                    tp_bb = r['bb_mid']
                    
                    if use_bb_mid and tp_bb > entry:
                        tp = max(tp_fixed, tp_bb)
                    else:
                        tp = tp_fixed
                
                # 做空
                elif sp > confidence and is_extreme_short and is_ranging:
                    sig = -1
                    entry = r['close']
                    atr = r['atr']
                    
                    sl = entry + sl_atr * atr
                    
                    tp_fixed = entry - tp_atr * atr
                    tp_bb = r['bb_mid']
                    
                    if use_bb_mid and tp_bb < entry:
                        tp = min(tp_fixed, tp_bb)
                    else:
                        tp = tp_fixed
                
                else:
                    # 紀錄拒絕原因
                    if (near_lower or near_upper) and not (is_extreme_long or is_extreme_short):
                        rejected_not_extreme += 1
                    elif (is_extreme_long or is_extreme_short) and not is_ranging:
                        rejected_not_ranging += 1
                    elif (is_extreme_long and lp <= confidence) or (is_extreme_short and sp <= confidence):
                        rejected_low_prob += 1
                
                signals.append({
                    'signal': sig,
                    'stop_loss': sl,
                    'take_profit': tp,
                    'position_size': position_pct / 100.0,
                    'long_proba': lp,
                    'short_proba': sp
                })
            
            signals = [{'signal': 0, 'stop_loss': np.nan, 'take_profit': np.nan, 'position_size': 1.0, 'long_proba': 0, 'short_proba': 0}] * 50 + signals
            df_sig = pd.DataFrame(signals)
            
            cnt = (df_sig['signal'] != 0).sum()
            
            if cnt == 0:
                st.warning("無信號")
                st.info(f"""
                **拒絕原因**:
                - 不夠極端: {rejected_not_extreme}
                - 不是盤整: {rejected_not_ranging}
                - 機率不足: {rejected_low_prob}
                
                建議: 降低極端%到 1.0 或降低信心度到 0.40
                """)
                return
            
            st.success(f"{cnt}信號 | 拒: 極端{rejected_not_extreme} 盤整{rejected_not_ranging} 機率{rejected_low_prob}")
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
            
            wr = metrics['win_rate']
            c3.metric("勝率", f"{wr:.1f}%", delta="目標>65%")
            pf = metrics['profit_factor']
            c3.metric("盈虧比", f"{pf:.2f}")
            
            dd = metrics['max_drawdown_pct']
            c4.metric("回撤", f"{dd:.1f}%", delta="目標<-25%")
            c4.metric("夏普", f"{metrics['sharpe_ratio']:.2f}")
            
            st.markdown("---")
            
            # Kelly
            if pf > 0 and wr > 0:
                avg_win_loss = pf if pf >= 1 else 0.5
                kelly = wr/100 - (1-wr/100) / avg_win_loss
                
                st.subheader("策略評估")
                col_e1, col_e2, col_e3 = st.columns(3)
                
                with col_e1:
                    if wr >= 65 and dd > -25:
                        st.success("✅ 達標: 勝率>65% 回撤<25%")
                        st.balloons()
                    elif wr >= 60:
                        st.success("👍 良好: 勝率>60%")
                    elif wr >= 50:
                        st.warning("⚠️ 一般: 勝率>50%")
                    else:
                        st.error("❌ 不佳")
                
                with col_e2:
                    st.metric("Kelly%", f"{kelly*100:.1f}%")
                    if kelly > 0.15:
                        st.success("優秀 (>15%)")
                    elif kelly > 0.05:
                        st.info("合格 (>5%)")
                    elif kelly > 0:
                        st.warning("勉強")
                    else:
                        st.error("不建議")
                
                with col_e3:
                    if monthly >= 40:
                        st.success(f"月化>{monthly:.0f}%")
                    elif monthly >= 25:
                        st.info(f"月化>{monthly:.0f}%")
                    else:
                        st.warning(f"月化={monthly:.0f}%")
            
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
                c3.metric("平均贏", f"${wins['pnl_usdt'].mean():.2f}" if len(wins)>0 else "$0")
                c4.metric("平均輸", f"${losses['pnl_usdt'].mean():.2f}" if len(losses)>0 else "$0")
                
                st.dataframe(trades[['entry_time', 'direction', 'entry_price', 'exit_price', 'pnl_usdt', 'exit_reason']].tail(20), use_container_width=True)
                
                csv = trades.to_csv(index=False).encode('utf-8')
                st.download_button("CSV", csv, f"{symbol}_mr_{datetime.now():%Y%m%d_%H%M}.csv", "text/csv")
            
        except Exception as e:
            st.error(f"錯: {e}")
            import traceback
            with st.expander("詳情"):
                st.code(traceback.format_exc())
