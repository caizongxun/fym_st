"""策略A: ML + BB均值回歸 - 高頻版"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

from models.ml_range_bound_strategy import MLRangeBoundStrategy
from backtesting.tick_level_engine import TickLevelBacktestEngine
from data.binance_loader import BinanceDataLoader


def render_strategy_a_tab(loader, symbol_selector):
    st.header("策略 A: BB均值回歸 (高頻版)")
    
    st.info("""
    **策略原理**:
    
    進場: BB上/下軌附近 + ML確認 + 盤整市場
    止盈: BB中軌 (均值回歸)
    止損: 1.3 ATR (緊止損)
    優勢: 高頻率 + 強制盈虧比 > 1.4
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
        leverage = st.slider("槓桿倍數", 3, 12, 8, key="lev", help="高頻+高槓桿")
        threshold = st.slider("信心度閾值", 0.25, 0.65, 0.38, 0.02, key="th", help="降低=更多交易")
    
    with col3:
        st.markdown("**技術參數**")
        bb_period = st.number_input("BB週期", 10, 50, 20, key="bb")
        adx_threshold = st.slider("ADX閾值", 20, 40, 30, key="adx", help="提高=更寬鬆盤整")
        sl_atr = st.slider("止損 ATR", 0.8, 2.0, 1.3, 0.1, key="sl", help="緊止損")
    
    with st.expander("進階設定"):
        bb_entry_pct = st.slider("進場BB範圍%", 0.5, 2.0, 1.5, 0.1, key="bb_pct", 
                                 help="允許在BB帶這個範圍內進場")
        min_profit_factor = st.slider("最低盈虧比", 1.2, 2.0, 1.4, 0.1, key="min_pf",
                                      help="預期盈虧比低於此值放棄交易")
        min_distance_pct = st.slider("最小到中軌距離%", 0.3, 1.5, 0.5, 0.1, key="min_dist",
                                     help="確保有足夠獲利空間")
    
    st.markdown("---")
    
    if st.button("開始執行", type="primary", use_container_width=True):
        prog = st.progress(0)
        stat = st.empty()
        
        try:
            # Load data
            stat.text("1/4: 載入數據...")
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
            
            st.success(f"載入: {len(df_train)}+{len(df_test)}根")
            prog.progress(20)
            
            # Train
            stat.text("2/4: 訓練ML模型...")
            strategy = MLRangeBoundStrategy(
                bb_period=bb_period,
                bb_std=2.0,
                adx_period=14,
                adx_threshold=adx_threshold
            )
            stats = strategy.train(df_train, forward_bars=10)
            
            st.success(f"訓練完成: 做多{stats['long_samples']} 做空{stats['short_samples']}")
            prog.progress(50)
            
            # Generate signals
            stat.text("3/4: 生成交易信號...")
            df_test = strategy.add_indicators(df_test)
            
            signals = []
            rejected_low_pf = 0
            rejected_low_dist = 0
            
            for i in range(50, len(df_test)):
                lp, sp = strategy.predict(df_test, i)
                r = df_test.iloc[i]
                
                sig = 0
                sl = np.nan
                tp = np.nan
                
                # 更寬鬆的進場範圍
                near_lower = r['close'] <= r['bb_lower'] * (1 + bb_entry_pct / 100)
                near_upper = r['close'] >= r['bb_upper'] * (1 - bb_entry_pct / 100)
                ranging = r['adx'] < adx_threshold
                
                # 到中軌距離
                dist_to_mid_pct = abs(r['close'] - r['bb_mid']) / r['close'] * 100
                
                if lp > threshold and near_lower and ranging:
                    if dist_to_mid_pct > min_distance_pct:
                        entry = r['close']
                        atr = r['atr']
                        sl = entry - sl_atr * atr
                        tp = r['bb_mid']
                        
                        risk = sl_atr * atr
                        reward = tp - entry
                        pf = reward / risk
                        
                        if pf >= min_profit_factor:
                            sig = 1
                        else:
                            rejected_low_pf += 1
                    else:
                        rejected_low_dist += 1
                
                elif sp > threshold and near_upper and ranging:
                    if dist_to_mid_pct > min_distance_pct:
                        entry = r['close']
                        atr = r['atr']
                        sl = entry + sl_atr * atr
                        tp = r['bb_mid']
                        
                        risk = sl_atr * atr
                        reward = entry - tp
                        pf = reward / risk
                        
                        if pf >= min_profit_factor:
                            sig = -1
                        else:
                            rejected_low_pf += 1
                    else:
                        rejected_low_dist += 1
                
                signals.append({
                    'signal': sig,
                    'stop_loss': sl,
                    'take_profit': tp,
                    'long_proba': lp,
                    'short_proba': sp
                })
            
            signals = [{'signal': 0, 'stop_loss': np.nan, 'take_profit': np.nan, 'long_proba': 0, 'short_proba': 0}] * 50 + signals
            df_sig = pd.DataFrame(signals)
            
            cnt = (df_sig['signal'] != 0).sum()
            
            if cnt == 0:
                st.warning("未生成任何交易信號")
                st.info(f"""
                **被拒絕的交易**:
                - 盈虧比不足: {rejected_low_pf}
                - 距離不足: {rejected_low_dist}
                
                建議: 降低信心度到 0.35 或 降低最低盈虧比到 1.3
                """)
                return
            
            st.success(f"生成 {cnt} 個信號 | 拒絕: PF={rejected_low_pf} Dist={rejected_low_dist}")
            prog.progress(70)
            
            # Backtest
            stat.text("4/4: 執行Tick級別回測...")
            engine = TickLevelBacktestEngine(capital, leverage, 0.0006, 0.02, 100)
            metrics = engine.run_backtest(df_test, df_sig)
            
            prog.progress(100)
            stat.text("完成!")
            
            # Results
            st.markdown("---")
            st.subheader("回測結果")
            
            c1, c2, c3, c4 = st.columns(4)
            
            pnl = metrics['final_equity'] - capital
            c1.metric("最終權益", f"${metrics['final_equity']:,.0f}", f"{pnl:+,.0f}")
            c1.metric("交易次數", metrics['total_trades'])
            
            ret = metrics['total_return_pct']
            monthly = ret * 30 / test_days
            c2.metric("總報酬率", f"{ret:.1f}%")
            c2.metric("月化報酬", f"{monthly:.1f}%", delta=f"vs 100%")
            
            c3.metric("勝率", f"{metrics['win_rate']:.1f}%")
            c3.metric("盈虧比", f"{metrics['profit_factor']:.2f}")
            
            c4.metric("最大回撤", f"{metrics['max_drawdown_pct']:.1f}%")
            c4.metric("夏普比率", f"{metrics['sharpe_ratio']:.2f}")
            
            # Evaluation
            st.markdown("---")
            
            if monthly >= 90 and metrics['profit_factor'] >= 1.4:
                st.success("[目標達成] 月化報酬>90% & 盈虧比>1.4")
                st.balloons()
            elif monthly >= 60:
                st.success("[良好] 月化報酬>60%")
            elif monthly >= 30:
                st.warning("[一般] 月化報酬>30%")
            else:
                st.info("[需優化] 月化<30%")
            
            if metrics['total_trades'] < 50:
                st.warning(f"交易次數偏少({metrics['total_trades']}), 建議降低信心度或放寬進場範圍")
            
            # Charts
            st.markdown("---")
            st.subheader("權益曲線")
            fig = engine.plot_equity_curve()
            st.plotly_chart(fig, use_container_width=True)
            
            # Trades
            trades = engine.get_trades_dataframe()
            if not trades.empty:
                st.markdown("---")
                st.subheader("交易分析")
                
                wins = trades[trades['pnl_usdt'] > 0]
                losses = trades[trades['pnl_usdt'] < 0]
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("獲利交易", len(wins))
                c2.metric("虧損交易", len(losses))
                c3.metric("平均獲利", f"${wins['pnl_usdt'].mean():.2f}" if len(wins) > 0 else "$0")
                c4.metric("平均虧損", f"${losses['pnl_usdt'].mean():.2f}" if len(losses) > 0 else "$0")
                
                st.markdown("**最近20筆交易**")
                disp = trades[['entry_time', 'direction', 'entry_price', 'exit_price', 'pnl_usdt', 'exit_reason']].tail(20)
                st.dataframe(disp, use_container_width=True)
                
                csv = trades.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "CSV下載",
                    csv,
                    f"{symbol}_{datetime.now():%Y%m%d_%H%M}.csv",
                    "text/csv"
                )
            
        except Exception as e:
            st.error(f"執行錯誤: {str(e)}")
            import traceback
            with st.expander("查看詳細錯誤"):
                st.code(traceback.format_exc())
