"""策略A: ML驅動的區間震盪交易 - 激進版"""

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
    
    st.header("策略 A: ML驅動的區間震盪交易 - 激進版")
    
    st.info("""
    **激進翻倉策略**:
    
    [+] 高槓桿複利 - 獲利後提高槓桿
    [+] 嚴格進場 - 只做高機率交易
    [+] 寬止盈 - 3-4x 盈虧比
    [+] 緊止損 - 0.8-1.0 ATR
    [+] 目標: 月報酬 100%
    """)
    
    st.markdown("---")
    st.subheader("策略參數設定")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**數據設定**")
        symbol_list = symbol_selector("strategy_a", multi=False)
        symbol = symbol_list[0]
        
        train_days = st.slider("訓練數據天數", 30, 180, 120, key="train_days", help="更多數據提高模型準確度")
        test_days = st.slider("回測天數", 7, 60, 30, key="test_days")
    
    with col2:
        st.markdown("**激進設定**")
        initial_capital = st.number_input("初始資金 (USDT)", 1000.0, 100000.0, 10000.0, 1000.0, key="capital")
        base_leverage = st.slider("基礎槓桿", 3, 10, 8, key="leverage", help="高槓桿高報酬")
        confidence_threshold = st.slider("信心度閾值", 0.5, 0.8, 0.65, 0.05, key="confidence", 
                                        help="提高閾值=更嚴格進場")
    
    with col3:
        st.markdown("**技術參數**")
        bb_period = st.number_input("BB週期", 10, 50, 20, key="bb_period")
        adx_threshold = st.slider("ADX閾值", 20, 40, 35, key="adx", help="提高=更寬鬆盤整")
        ticks_per_candle = st.select_slider("Tick密度", [50, 100, 200], 100, key="ticks")
    
    with st.expander("進階: 風險/報酬參數"):
        col_a1, col_a2, col_a3 = st.columns(3)
        
        with col_a1:
            st.markdown("**止損/止盈**")
            stop_loss_atr = st.slider("止損 ATR", 0.5, 2.0, 0.8, 0.1, key="sl_atr", help="緊止損")
            profit_factor_target = st.slider("目標盈虧比", 2.0, 5.0, 3.5, 0.5, key="pf_target", help="寬止盈")
        
        with col_a2:
            st.markdown("**動態仓位**")
            base_position_pct = st.slider("基礎仓位%", 30, 100, 80, 10, key="base_pos")
            max_position_pct = st.slider("最大仓位%", 50, 100, 100, 10, key="max_pos")
        
        with col_a3:
            st.markdown("**複利設定**")
            enable_compound = st.checkbox("啟用複利槓桿", value=True, key="compound",
                                         help="獲利後提高槓桿")
            max_leverage = st.slider("最大槓桿", 5, 20, 15, 1, key="max_lev") if enable_compound else base_leverage
            enable_drawdown_control = st.checkbox("啟用回撤控制", value=True, key="dd_ctrl")
    
    st.markdown("---")
    
    if st.button("開始執行: 訓練 + 回測", key="execute_all", type="primary", use_container_width=True):
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1
            status_text.text("步驟 1/4: 載入數據...")
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
            
            st.success(f"載入: 訓練{len(df_train)}根 測試{len(df_test)}根")
            progress_bar.progress(20)
            
            # Step 2
            status_text.text("步驟 2/4: 訓練ML模型...")
            
            strategy = MLRangeBoundStrategy(
                bb_period=bb_period,
                bb_std=2.0,
                adx_period=14,
                adx_threshold=adx_threshold
            )
            
            train_stats = strategy.train(df_train, forward_bars=10)
            
            st.success("訓練完成!")
            col_t1, col_t2, col_t3 = st.columns(3)
            with col_t1:
                st.metric("總樣本", train_stats['total_samples'])
            with col_t2:
                st.metric("做多", f"{train_stats['long_samples']} ({train_stats['long_ratio']})")
            with col_t3:
                st.metric("做空", f"{train_stats['short_samples']} ({train_stats['short_ratio']})")
            
            progress_bar.progress(50)
            
            # Step 3
            status_text.text("步驟 3/4: 生成交易信號(複利模式)...")
            
            df_test = strategy.add_indicators(df_test)
            
            signals = []
            peak_equity = initial_capital
            current_equity = initial_capital
            winning_streak = 0
            
            for i in range(50, len(df_test)):
                long_proba, short_proba = strategy.predict(df_test, i)
                
                row = df_test.iloc[i]
                signal = 0
                stop_loss = np.nan
                take_profit = np.nan
                position_size = 1.0
                current_leverage = base_leverage
                
                near_lower = row['close'] <= row['bb_lower'] * 1.003
                near_upper = row['close'] >= row['bb_upper'] * 0.997
                is_ranging = row['adx'] < adx_threshold
                
                # 複利槓桿: 獲利後提高槓桿
                if enable_compound and current_equity > initial_capital:
                    profit_ratio = (current_equity - initial_capital) / initial_capital
                    leverage_boost = min(profit_ratio * 10, max_leverage - base_leverage)
                    current_leverage = min(base_leverage + leverage_boost, max_leverage)
                
                # 計算動態仓位
                if long_proba > confidence_threshold or short_proba > confidence_threshold:
                    proba = max(long_proba, short_proba)
                    proba_normalized = (proba - confidence_threshold) / (1.0 - confidence_threshold)
                    position_pct = base_position_pct + (max_position_pct - base_position_pct) * proba_normalized
                    position_size = position_pct / 100.0
                    
                    # 連續獲利加仓
                    if winning_streak >= 2:
                        position_size = min(position_size * 1.2, 1.0)
                    
                    # 回撤控制
                    if enable_drawdown_control:
                        drawdown = (peak_equity - current_equity) / peak_equity
                        if drawdown > 0.15:
                            position_size *= 0.4
                            current_leverage = max(base_leverage * 0.5, 3)
                        elif drawdown > 0.08:
                            position_size *= 0.6
                
                if long_proba > confidence_threshold and near_lower and is_ranging:
                    signal = 1
                    entry = row['close']
                    atr = row['atr']
                    
                    stop_loss = entry - stop_loss_atr * atr
                    risk = stop_loss_atr * atr
                    reward = risk * profit_factor_target
                    take_profit = entry + reward
                        
                elif short_proba > confidence_threshold and near_upper and is_ranging:
                    signal = -1
                    entry = row['close']
                    atr = row['atr']
                    
                    stop_loss = entry + stop_loss_atr * atr
                    risk = stop_loss_atr * atr
                    reward = risk * profit_factor_target
                    take_profit = entry - reward
                
                signals.append({
                    'signal': signal,
                    'long_proba': long_proba,
                    'short_proba': short_proba,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'position_size': position_size if signal != 0 else 1.0,
                    'leverage': current_leverage if signal != 0 else base_leverage
                })
                
                # 簡單更新狀態
                if i > 50 and len(signals) > 1:
                    prev_signal = signals[-2]
                    if prev_signal['signal'] != 0:
                        if prev_signal['signal'] == 1:
                            pnl_pct = (row['close'] - df_test.iloc[i-1]['close']) / df_test.iloc[i-1]['close']
                        else:
                            pnl_pct = (df_test.iloc[i-1]['close'] - row['close']) / df_test.iloc[i-1]['close']
                        
                        pnl = current_equity * pnl_pct * prev_signal.get('leverage', base_leverage) * prev_signal.get('position_size', 1.0)
                        current_equity += pnl
                        peak_equity = max(peak_equity, current_equity)
                        
                        if pnl > 0:
                            winning_streak += 1
                        else:
                            winning_streak = 0
            
            signals = [{'signal': 0, 'long_proba': 0, 'short_proba': 0, 'stop_loss': np.nan, 'take_profit': np.nan, 'position_size': 1.0, 'leverage': base_leverage}] * 50 + signals
            df_signals = pd.DataFrame(signals)
            
            signal_count = (df_signals['signal'] != 0).sum()
            
            if signal_count == 0:
                st.warning("未生成交易信號")
                st.info("建議: 降低信心度閾值到 0.55")
                return
            
            avg_lev = df_signals[df_signals['signal'] != 0]['leverage'].mean()
            st.success(f"信號: {signal_count}個 | 平均槓桿: {avg_lev:.1f}x")
            progress_bar.progress(70)
            
            # Step 4 - 使用動態槓桿回測
            status_text.text("步驟 4/4: Tick級別回測(複利槓桿)...")
            
            # 使用基礎槓桿初始化,但signals中包含動態槓桿
            engine = TickLevelBacktestEngine(
                initial_capital=initial_capital,
                leverage=1,  # 設為1,由signals中的leverage控制
                fee_rate=0.0006,
                slippage_pct=0.02,
                ticks_per_candle=ticks_per_candle
            )
            
            metrics = engine.run_backtest_with_dynamic_leverage(df_test, df_signals)
            
            progress_bar.progress(100)
            status_text.text("完成!")
            
            # Results
            st.markdown("---")
            st.subheader("回測結果")
            
            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            
            with col_r1:
                profit = metrics['final_equity'] - initial_capital
                st.metric("最終權益", f"${metrics['final_equity']:,.2f}", delta=f"{profit:+,.2f}")
                st.metric("交易次數", metrics['total_trades'])
            
            with col_r2:
                return_pct = metrics['total_return_pct']
                monthly_return = return_pct * 30 / test_days
                st.metric("總報酬", f"{return_pct:.1f}%")
                st.metric("月化報酬", f"{monthly_return:.1f}%", delta="目標100%")
            
            with col_r3:
                st.metric("勝率", f"{metrics['win_rate']:.1f}%")
                st.metric("盈虧比", f"{metrics['profit_factor']:.2f}")
            
            with col_r4:
                st.metric("最大回撤", f"{metrics['max_drawdown_pct']:.1f}%")
                st.metric("夏普", f"{metrics['sharpe_ratio']:.2f}")
            
            # Performance
            st.markdown("---")
            
            if monthly_return >= 100:
                st.success("[目標達成] 月化報酬 100%+")
                st.balloons()
            elif monthly_return >= 70:
                st.success("[接近目標] 月化報酬 70%+")
            elif monthly_return >= 40:
                st.warning("[需優化] 月化 40%+, 建議提高槓桿或降低閾值")
            else:
                st.error("[未達標] 月化<40%, 需重新調整")
            
            # Equity curve
            st.markdown("---")
            st.subheader("權益曲線")
            fig = engine.plot_equity_curve()
            st.plotly_chart(fig, use_container_width=True)
            
            # Trades analysis
            trades_df = engine.get_trades_dataframe()
            if not trades_df.empty:
                st.markdown("---")
                st.subheader("交易統計")
                
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
                
                st.markdown("**最近20筆**")
                display_df = trades_df[['entry_time', 'exit_time', 'direction', 'entry_price', 'exit_price', 'pnl_usdt', 'exit_reason']].tail(20).copy()
                display_df['pnl_usdt'] = display_df['pnl_usdt'].apply(lambda x: f"${x:.2f}")
                st.dataframe(display_df, use_container_width=True)
                
                csv = trades_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "CSV下載",
                    csv,
                    f"{symbol}_aggressive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    key="dl"
                )
            
            # Save model
            st.markdown("---")
            if st.checkbox("保存模型"):
                model_path = f'models/saved/{symbol}_aggressive_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
                strategy.save_models(model_path)
                st.success(f"已保存: {model_path}")
                
        except Exception as e:
            st.error(f"錯誤: {str(e)}")
            import traceback
            with st.expander("詳細錯誤"):
                st.code(traceback.format_exc())
