"""策略A: ML驅動的區間震盪交易 - 穩健翻倉版"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

from models.ml_range_bound_strategy import MLRangeBoundStrategy
from backtesting.tick_level_engine import TickLevelBacktestEngine
from data.binance_loader import BinanceDataLoader


def render_strategy_a_tab(loader, symbol_selector):
    """渲柔策略A Tab"""
    
    st.header("策略 A: 穩健翻倉策略")
    
    st.info("""
    **穩健翻倉方案**:
    
    [+] 中等槓桿 + 高頻率 - 平衡風險報酬
    [+] 放寬進場 + 嚴格止損 - 多交易控回撤
    [+] 分批止盈 - 50%先出保證獲利
    [+] 移動止損 - 保護獲利不回吐
    [+] 目標: 月報酬 100% | 最大回撤 <30%
    """)
    
    st.markdown("---")
    st.subheader("策略參數")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**數據設定**")
        symbol_list = symbol_selector("strategy_a", multi=False)
        symbol = symbol_list[0]
        train_days = st.slider("訓練天數", 60, 180, 90, key="train_days")
        test_days = st.slider("回測天數", 7, 60, 30, key="test_days")
    
    with col2:
        st.markdown("**交易設定**")
        initial_capital = st.number_input("初始資金", 1000.0, 100000.0, 10000.0, 1000.0, key="cap")
        leverage = st.slider("槓桿倍數", 3, 10, 6, key="lev", help="中等槓桿")
        confidence_threshold = st.slider("信心度", 0.3, 0.7, 0.48, 0.02, key="conf", help="降低閾值增加交易")
    
    with col3:
        st.markdown("**技術參數**")
        bb_period = st.number_input("BB週期", 10, 50, 20, key="bb")
        adx_threshold = st.slider("ADX閾值", 20, 40, 28, key="adx", help="降低=更容易識別盤整")
        ticks = st.select_slider("Tick密度", [50, 100, 200], 100, key="tk")
    
    with st.expander("進階設定"):
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.markdown("**風險管理**")
            stop_loss_atr = st.slider("止損 ATR", 0.8, 2.0, 1.2, 0.2, key="sl")
            use_trailing_stop = st.checkbox("移動止損", value=True, key="trail")
            trailing_pct = st.slider("移動止損%", 0.5, 2.0, 1.0, 0.1, key="trail_pct") if use_trailing_stop else 0
        
        with col_a2:
            st.markdown("**止盈管理**")
            use_partial_tp = st.checkbox("分批止盈", value=True, key="partial")
            first_tp_atr = st.slider("第一止盈 ATR", 1.0, 3.0, 1.8, 0.2, key="tp1") if use_partial_tp else 0
            second_tp_atr = st.slider("第二止盈 ATR", 2.0, 5.0, 3.5, 0.5, key="tp2") if use_partial_tp else 0
    
    st.markdown("---")
    
    if st.button("開始執行", type="primary", use_container_width=True):
        progress = st.progress(0)
        status = st.empty()
        
        try:
            # Load data
            status.text("1/4: 載入數據...")
            progress.progress(10)
            
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
            progress.progress(20)
            
            # Train
            status.text("2/4: 訓練模型...")
            strategy = MLRangeBoundStrategy(bb_period=bb_period, bb_std=2.0, adx_period=14, adx_threshold=adx_threshold)
            stats = strategy.train(df_train, forward_bars=10)
            
            st.success(f"訓練: {stats['long_samples']}L + {stats['short_samples']}S")
            progress.progress(50)
            
            # Generate signals
            status.text("3/4: 生成信號...")
            df_test = strategy.add_indicators(df_test)
            signals = []
            
            for i in range(50, len(df_test)):
                long_p, short_p = strategy.predict(df_test, i)
                row = df_test.iloc[i]
                
                signal = 0
                sl = np.nan
                tp1 = np.nan
                tp2 = np.nan
                
                near_lower = row['close'] <= row['bb_lower'] * 1.008
                near_upper = row['close'] >= row['bb_upper'] * 0.992
                ranging = row['adx'] < adx_threshold
                
                if long_p > confidence_threshold and near_lower and ranging:
                    signal = 1
                    entry = row['close']
                    atr = row['atr']
                    sl = entry - stop_loss_atr * atr
                    
                    if use_partial_tp:
                        tp1 = entry + first_tp_atr * atr
                        tp2 = entry + second_tp_atr * atr
                    else:
                        tp1 = entry + 2.0 * atr
                        tp2 = entry + 3.0 * atr
                
                elif short_p > confidence_threshold and near_upper and ranging:
                    signal = -1
                    entry = row['close']
                    atr = row['atr']
                    sl = entry + stop_loss_atr * atr
                    
                    if use_partial_tp:
                        tp1 = entry - first_tp_atr * atr
                        tp2 = entry - second_tp_atr * atr
                    else:
                        tp1 = entry - 2.0 * atr
                        tp2 = entry - 3.0 * atr
                
                signals.append({
                    'signal': signal,
                    'stop_loss': sl,
                    'take_profit': tp1,
                    'take_profit_2': tp2,
                    'use_trailing': use_trailing_stop,
                    'trailing_pct': trailing_pct
                })
            
            signals = [{'signal': 0, 'stop_loss': np.nan, 'take_profit': np.nan, 'take_profit_2': np.nan, 'use_trailing': False, 'trailing_pct': 0}] * 50 + signals
            df_sig = pd.DataFrame(signals)
            
            sig_count = (df_sig['signal'] != 0).sum()
            if sig_count == 0:
                st.warning("未生成信號 - 降低信心度閾值")
                return
            
            st.success(f"信號: {sig_count}個")
            progress.progress(70)
            
            # Backtest
            status.text("4/4: 回測...")
            engine = TickLevelBacktestEngine(initial_capital, leverage, 0.0006, 0.02, ticks)
            metrics = engine.run_backtest(df_test, df_sig)
            
            progress.progress(100)
            status.text("完成!")
            
            # Results
            st.markdown("---")
            st.subheader("結果")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                pnl = metrics['final_equity'] - initial_capital
                st.metric("最終權益", f"${metrics['final_equity']:,.0f}", f"{pnl:+,.0f}")
                st.metric("交易", metrics['total_trades'])
            
            with col2:
                ret = metrics['total_return_pct']
                monthly = ret * 30 / test_days
                st.metric("總報酬", f"{ret:.1f}%")
                st.metric("月化", f"{monthly:.1f}%", "vs 100%")
            
            with col3:
                st.metric("勝率", f"{metrics['win_rate']:.1f}%")
                st.metric("盈虧比", f"{metrics['profit_factor']:.2f}")
            
            with col4:
                st.metric("最大回撤", f"{metrics['max_drawdown_pct']:.1f}%")
                st.metric("夏普", f"{metrics['sharpe_ratio']:.2f}")
            
            # Evaluation
            st.markdown("---")
            if monthly >= 90 and metrics['max_drawdown_pct'] > -30:
                st.success("[達標] 月化>90% 回撤<30%")
                st.balloons()
            elif monthly >= 60:
                st.success("[良好] 月化>60%")
            elif monthly >= 30:
                st.warning("[一般] 月化>30%")
            else:
                st.error("[不佳] 需調整")
            
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
                c1.metric("獲利交易", len(wins))
                c2.metric("虧損交易", len(losses))
                c3.metric("平均獲利", f"${wins['pnl_usdt'].mean():.2f}" if len(wins) > 0 else "$0")
                c4.metric("平均虧損", f"${losses['pnl_usdt'].mean():.2f}" if len(losses) > 0 else "$0")
                
                st.markdown("**最近20筆**")
                disp = trades[['entry_time', 'direction', 'entry_price', 'exit_price', 'pnl_usdt', 'exit_reason']].tail(20)
                st.dataframe(disp, use_container_width=True)
                
                csv = trades.to_csv(index=False).encode('utf-8')
                st.download_button("CSV", csv, f"{symbol}_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv")
            
        except Exception as e:
            st.error(f"錯誤: {str(e)}")
            import traceback
            with st.expander("詳情"):
                st.code(traceback.format_exc())
