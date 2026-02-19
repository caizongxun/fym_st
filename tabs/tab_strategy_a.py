"""策略A: ML驅動的區間震盪交易"""

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
    
    st.header("策略 A: ML驅動的區間震盪交易")
    
    st.info("""
    **策略核心優勢**:
    
    [+] 無固定RSI限制 - AI模型動態學習最佳進場時機
    [+] 20+智能特徵 - 價格、波動、成交量、趨勢多維分析
    [+] 雙模型架構 - 做多/做空獨立預測
    [+] Tick級別回測 - 模擬K線內100個tick
    [+] 優化止盈止損 - 提高盈虧比
    """)
    
    st.markdown("---")
    st.subheader("策略參數設定")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**數據設定**")
        symbol_list = symbol_selector("strategy_a", multi=False)
        symbol = symbol_list[0]
        
        train_days = st.slider("訓練數據天數", 30, 180, 90, key="train_days")
        test_days = st.slider("回測天數", 7, 60, 30, key="test_days")
    
    with col2:
        st.markdown("**交易設定**")
        initial_capital = st.number_input("初始資金 (USDT)", 1000.0, 100000.0, 10000.0, 1000.0, key="capital")
        leverage = st.slider("槓桿倍數", 1, 10, 3, key="leverage")
        confidence_threshold = st.slider("模型信心度閾值", 0.3, 0.8, 0.5, 0.05, key="confidence")
    
    with col3:
        st.markdown("**技術參數**")
        bb_period = st.number_input("BB週期", 10, 50, 20, key="bb_period")
        adx_threshold = st.slider("ADX閾值", 15, 35, 30, key="adx")
        ticks_per_candle = st.select_slider("Tick模擬密度", [50, 100, 200], 100, key="ticks")
    
    # 止盈止損設定
    with st.expander("進階設定: 止盈止損"):
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            stop_loss_atr = st.slider("止損 ATR 倍數", 1.0, 3.0, 1.5, 0.5, key="sl_atr", 
                                      help="止損 = 進場價 +/- (ATR * 此倍數)")
        with col_a2:
            take_profit_mode = st.radio(
                "止盈模式",
                ["中軌回歸", "固定盈虧比"],
                key="tp_mode",
                help="中軌回歸: 目標BB中軌 | 固定盈虧比: 止盈 = 止損 * 盈虧比"
            )
            
            if take_profit_mode == "固定盈虧比":
                profit_factor_target = st.slider("目標盈虧比", 1.5, 3.0, 2.0, 0.5, key="pf_target")
    
    st.markdown("---")
    
    if st.button("開始執行: 訓練 + 回測", key="execute_all", type="primary", use_container_width=True):
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1
            status_text.text("步驟 1/4: 載入訓練數據...")
            progress_bar.progress(10)
            
            if isinstance(loader, BinanceDataLoader):
                end_date = datetime.now()
                start_date = end_date - timedelta(days=train_days + test_days)
                df_all = loader.load_historical_data(symbol, '15m', start_date, end_date)
            else:
                df_all = loader.load_klines(symbol, '15m')
                df_all = df_all.tail((train_days + test_days) * 96)
            
            split_idx = len(df_all) - test_days * 96
            df_train = df_all.iloc[:split_idx].copy()
            df_test = df_all.iloc[split_idx:].copy()
            
            st.success(f"載入完成: 訓練 {len(df_train)} 根K線, 測試 {len(df_test)} 根K線")
            progress_bar.progress(20)
            
            # Step 2
            status_text.text("步驟 2/4: 訓練機器學習模型...")
            
            strategy = MLRangeBoundStrategy(
                bb_period=bb_period,
                bb_std=2.0,
                adx_period=14,
                adx_threshold=adx_threshold
            )
            
            train_stats = strategy.train(df_train, forward_bars=10)
            
            st.success("訓練完成!")
            col_t1, col_t2, col_t3, col_t4 = st.columns(4)
            with col_t1:
                st.metric("總樣本數", train_stats['total_samples'])
            with col_t2:
                st.metric("做多樣本", f"{train_stats['long_samples']} ({train_stats['long_ratio']})")
            with col_t3:
                st.metric("做空樣本", f"{train_stats['short_samples']} ({train_stats['short_ratio']})")
            with col_t4:
                st.metric("測試預測", f"L:{train_stats['test_long_proba']:.3f} S:{train_stats['test_short_proba']:.3f}")
            
            progress_bar.progress(50)
            
            # Step 3
            status_text.text("步驟 3/4: 生成交易信號...")
            
            df_test = strategy.add_indicators(df_test)
            
            signals = []
            
            for i in range(50, len(df_test)):
                long_proba, short_proba = strategy.predict(df_test, i)
                
                row = df_test.iloc[i]
                signal = 0
                stop_loss = np.nan
                take_profit = np.nan
                
                near_lower = row['close'] <= row['bb_lower'] * 1.005
                near_upper = row['close'] >= row['bb_upper'] * 0.995
                is_ranging = row['adx'] < adx_threshold
                
                if long_proba > confidence_threshold and near_lower and is_ranging:
                    signal = 1
                    entry = row['close']
                    atr = row['atr']
                    
                    # 止損
                    stop_loss = entry - stop_loss_atr * atr
                    
                    # 止盈
                    if take_profit_mode == "中軌回歸":
                        take_profit = row['bb_mid']
                    else:
                        # 固定盈虧比
                        risk = stop_loss_atr * atr
                        reward = risk * profit_factor_target
                        take_profit = entry + reward
                        
                elif short_proba > confidence_threshold and near_upper and is_ranging:
                    signal = -1
                    entry = row['close']
                    atr = row['atr']
                    
                    # 止損
                    stop_loss = entry + stop_loss_atr * atr
                    
                    # 止盈
                    if take_profit_mode == "中軌回歸":
                        take_profit = row['bb_mid']
                    else:
                        # 固定盈虧比
                        risk = stop_loss_atr * atr
                        reward = risk * profit_factor_target
                        take_profit = entry - reward
                
                signals.append({
                    'signal': signal,
                    'long_proba': long_proba,
                    'short_proba': short_proba,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                })
            
            signals = [{'signal': 0, 'long_proba': 0, 'short_proba': 0, 'stop_loss': np.nan, 'take_profit': np.nan}] * 50 + signals
            df_signals = pd.DataFrame(signals)
            
            signal_count = (df_signals['signal'] != 0).sum()
            long_count = (df_signals['signal'] == 1).sum()
            short_count = (df_signals['signal'] == -1).sum()
            
            if signal_count == 0:
                st.warning("未生成任何交易信號")
                st.info("建議: 降低信心度閾值")
                return
            
            st.success(f"信號生成完成: 總共 {signal_count} 個 (做多: {long_count}, 做空: {short_count})")
            progress_bar.progress(70)
            
            # Step 4
            status_text.text("步驟 4/4: 執行Tick級別回測...")
            
            engine = TickLevelBacktestEngine(
                initial_capital=initial_capital,
                leverage=leverage,
                fee_rate=0.0006,
                slippage_pct=0.02,
                ticks_per_candle=ticks_per_candle
            )
            
            metrics = engine.run_backtest(df_test, df_signals)
            
            progress_bar.progress(100)
            status_text.text("全部完成!")
            
            # Results
            st.markdown("---")
            st.subheader("回測結果")
            
            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            
            with col_r1:
                profit = metrics['final_equity'] - initial_capital
                st.metric("最終權益", f"${metrics['final_equity']:,.2f}", delta=f"{profit:+,.2f} USDT")
                st.metric("交易次數", metrics['total_trades'])
            
            with col_r2:
                st.metric("報酬率", f"{metrics['total_return_pct']:.2f}%", delta="Tick級別")
                st.metric("勝率", f"{metrics['win_rate']:.1f}%")
            
            with col_r3:
                pf = metrics['profit_factor']
                pf_delta = "OK" if pf > 1.5 else "LOW"
                st.metric("盈虧比", f"{pf:.2f}", delta=pf_delta)
                st.metric("夏普比率", f"{metrics['sharpe_ratio']:.2f}")
            
            with col_r4:
                st.metric("最大回撤", f"{metrics['max_drawdown_pct']:.2f}%")
                st.metric("平均每筆", f"${metrics['avg_pnl_per_trade']:.2f}")
            
            # Performance
            st.markdown("---")
            return_pct = metrics['total_return_pct']
            
            if pf < 1.0:
                st.error("[警告] 盈虧比 < 1.0 - 策略需要優化")
                st.info("建議: 使用'固定盈虧比'模式,設定目標盈虧比 2.0")
            elif return_pct > 15 and metrics['win_rate'] > 50 and pf > 1.5:
                st.success("[優秀] 策略表現非常出色!")
                st.balloons()
            elif return_pct > 10 and pf > 1.3:
                st.success("[良好] 策略有穩定的獲利能力")
            elif return_pct > 0:
                st.warning("[一般] 報酬率偏低")
            else:
                st.error("[不佳] 表現不佳")
            
            # Equity curve
            st.markdown("---")
            st.subheader("權益曲線")
            fig = engine.plot_equity_curve()
            st.plotly_chart(fig, use_container_width=True)
            
            # Trades
            trades_df = engine.get_trades_dataframe()
            if not trades_df.empty:
                st.markdown("---")
                st.subheader("交易明細")
                
                # 顯示盈虧分析
                wins = trades_df[trades_df['pnl_usdt'] > 0]
                losses = trades_df[trades_df['pnl_usdt'] < 0]
                
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                with col_s1:
                    st.metric("獲利交易", len(wins))
                with col_s2:
                    st.metric("虧損交易", len(losses))
                with col_s3:
                    avg_win = wins['pnl_usdt'].mean() if len(wins) > 0 else 0
                    st.metric("平均獲利", f"${avg_win:.2f}")
                with col_s4:
                    avg_loss = losses['pnl_usdt'].mean() if len(losses) > 0 else 0
                    st.metric("平均虧損", f"${avg_loss:.2f}")
                
                # 交易記錄
                st.markdown("**最近20筆交易**")
                display_df = trades_df[[
                    'entry_time', 'exit_time', 'direction',
                    'entry_price', 'exit_price', 'pnl_usdt', 'pnl_pct', 'exit_reason'
                ]].tail(20).copy()
                
                display_df['pnl_usdt'] = display_df['pnl_usdt'].apply(lambda x: f"${x:.2f}")
                display_df['entry_price'] = display_df['entry_price'].apply(lambda x: f"${x:.2f}")
                display_df['exit_price'] = display_df['exit_price'].apply(lambda x: f"${x:.2f}")
                
                st.dataframe(display_df, use_container_width=True)
                
                csv = trades_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="下載完整交易記錄 CSV",
                    data=csv,
                    file_name=f"{symbol}_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download"
                )
            
            # Feature importance
            st.markdown("---")
            st.subheader("特徵重要性")
            
            col_fi1, col_fi2 = st.columns(2)
            
            with col_fi1:
                st.markdown("**做多模型**")
                if hasattr(strategy.long_model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        '特徵': train_stats['feature_names'],
                        '重要性': strategy.long_model.feature_importances_
                    }).sort_values('重要性', ascending=False).head(10)
                    
                    fig_long = go.Figure(go.Bar(
                        x=importance_df['重要性'],
                        y=importance_df['特徵'],
                        orientation='h',
                        marker_color='green'
                    ))
                    fig_long.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_long, use_container_width=True)
            
            with col_fi2:
                st.markdown("**做空模型**")
                if hasattr(strategy.short_model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        '特徵': train_stats['feature_names'],
                        '重要性': strategy.short_model.feature_importances_
                    }).sort_values('重要性', ascending=False).head(10)
                    
                    fig_short = go.Figure(go.Bar(
                        x=importance_df['重要性'],
                        y=importance_df['特徵'],
                        orientation='h',
                        marker_color='red'
                    ))
                    fig_short.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_short, use_container_width=True)
            
            # Save model
            st.markdown("---")
            if st.checkbox("保存此模型供未來使用"):
                model_path = f'models/saved/{symbol}_strategy_a_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
                strategy.save_models(model_path)
                st.success(f"模型已保存: {model_path}")
                
        except Exception as e:
            st.error(f"執行錯誤: {str(e)}")
            import traceback
            with st.expander("查看詳細錯誤信息"):
                st.code(traceback.format_exc())
