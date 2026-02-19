"""策略A: ML驅動的區間震盪交易 (一鍵執行)"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

from models.ml_range_bound_strategy import MLRangeBoundStrategy
from backtesting.tick_level_engine import TickLevelBacktestEngine
from data.binance_loader import BinanceDataLoader


def render_strategy_a_tab(loader, symbol_selector):
    """渲染策略A Tab - 一鍵執行ML訓練和Tick級別回測"""
    
    st.header("策略 A: ML驅動的區間震盪交易")
    
    st.info("""
    **策略核心優勢**:
    
    ✅ **無固定RSI限制** - AI模型動態學習最佳進場時機
    
    ✅ **20+智能特徵** - 價格、波動、成交量、趨勢多維分析
    
    ✅ **雙模型架構** - 做多/做空獨立預測,更精準
    
    ✅ **Tick級別回測** - 模擬K線內100個tick,真實反映止損觸發
    
    ✅ **自適應止損** - 基於ATR動態調整,適應市場波動
    
    ---
    
    **一鍵執行流程**: 選擇參數 → 點擊按鈕 → 自動訓練 → 自動回測 → 查看結果
    """)
    
    st.markdown("---")
    
    # ========== 參數設定區 ==========
    st.subheader("⚙️ 策略參數設定")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**數據設定**")
        symbol_list = symbol_selector("strategy_a", multi=False)
        symbol = symbol_list[0]
        
        train_days = st.slider(
            "訓練數據天數",
            min_value=30,
            max_value=180,
            value=90,
            key="train_days",
            help="更多數據 = 更好的模型"
        )
        
        test_days = st.slider(
            "回測天數",
            min_value=7,
            max_value=60,
            value=30,
            key="test_days"
        )
    
    with col2:
        st.markdown("**交易設定**")
        
        initial_capital = st.number_input(
            "初始資金 (USDT)",
            min_value=1000.0,
            max_value=100000.0,
            value=10000.0,
            step=1000.0,
            key="capital"
        )
        
        leverage = st.slider(
            "槓桿倍數",
            min_value=1,
            max_value=10,
            value=3,
            key="leverage",
            help="建議3-5倍"
        )
        
        confidence_threshold = st.slider(
            "模型信心度閾值",
            min_value=0.3,
            max_value=0.8,
            value=0.6,
            step=0.05,
            key="confidence",
            help="模型預測機率 > 此值才進場"
        )
    
    with col3:
        st.markdown("**技術參數**")
        
        bb_period = st.number_input(
            "BB週期",
            min_value=10,
            max_value=50,
            value=20,
            key="bb_period"
        )
        
        adx_threshold = st.slider(
            "ADX閾值",
            min_value=15,
            max_value=35,
            value=25,
            key="adx",
            help="ADX < 此值 = 盤整市場"
        )
        
        ticks_per_candle = st.select_slider(
            "Tick模擬密度",
            options=[50, 100, 200],
            value=100,
            key="ticks",
            help="越多越真實但越慢"
        )
    
    st.markdown("---")
    
    # ========== 一鍵執行按鈕 ==========
    if st.button("🚀 一鍵執行: 訓練 + 回測", key="execute_all", type="primary", use_container_width=True):
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # ========== Step 1: 載入訓練數據 ==========
            status_text.text("📥 步驟 1/4: 載入訓練數據...")
            progress_bar.progress(10)
            
            if isinstance(loader, BinanceDataLoader):
                end_date = datetime.now()
                start_date = end_date - timedelta(days=train_days + test_days)
                df_all = loader.load_historical_data(symbol, '15m', start_date, end_date)
            else:
                df_all = loader.load_klines(symbol, '15m')
                df_all = df_all.tail((train_days + test_days) * 96)
            
            # Split train/test
            split_idx = len(df_all) - test_days * 96
            df_train = df_all.iloc[:split_idx].copy()
            df_test = df_all.iloc[split_idx:].copy()
            
            st.success(f"✅ 載入完成: 訓練 {len(df_train)} 根K線, 測試 {len(df_test)} 根K線")
            progress_bar.progress(20)
            
            # ========== Step 2: 訓練ML模型 ==========
            status_text.text("🤖 步驟 2/4: 訓練機器學習模型...")
            
            strategy = MLRangeBoundStrategy(
                bb_period=bb_period,
                bb_std=2.0,
                adx_period=14,
                adx_threshold=adx_threshold
            )
            
            train_stats = strategy.train(df_train, forward_bars=10)
            
            st.success(f"✅ 訓練完成: 做多樣本 {train_stats['long_samples']}, 做空樣本 {train_stats['short_samples']}")
            progress_bar.progress(50)
            
            # ========== Step 3: 生成交易信號 ==========
            status_text.text("📊 步驟 3/4: 生成交易信號...")
            
            df_test = strategy.add_indicators(df_test)
            
            signals = []
            for i in range(50, len(df_test)):
                long_proba, short_proba = strategy.predict(df_test, i)
                
                signal = 0
                stop_loss = np.nan
                take_profit = np.nan
                
                if long_proba > confidence_threshold:
                    signal = 1
                    entry = df_test.iloc[i]['close']
                    atr = df_test.iloc[i]['atr']
                    stop_loss = entry - 2 * atr
                    take_profit = df_test.iloc[i]['bb_mid']
                elif short_proba > confidence_threshold:
                    signal = -1
                    entry = df_test.iloc[i]['close']
                    atr = df_test.iloc[i]['atr']
                    stop_loss = entry + 2 * atr
                    take_profit = df_test.iloc[i]['bb_mid']
                
                signals.append({
                    'signal': signal,
                    'long_proba': long_proba,
                    'short_proba': short_proba,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                })
            
            # Pad signals
            signals = [{'signal': 0, 'long_proba': 0, 'short_proba': 0, 'stop_loss': np.nan, 'take_profit': np.nan}] * 50 + signals
            df_signals = pd.DataFrame(signals)
            
            signal_count = (df_signals['signal'] != 0).sum()
            long_count = (df_signals['signal'] == 1).sum()
            short_count = (df_signals['signal'] == -1).sum()
            
            st.success(f"✅ 信號生成完成: 總共 {signal_count} 個 (做多: {long_count}, 做空: {short_count})")
            progress_bar.progress(70)
            
            # ========== Step 4: Tick級別回測 ==========
            status_text.text("⚡ 步驟 4/4: 執行Tick級別回測...")
            
            engine = TickLevelBacktestEngine(
                initial_capital=initial_capital,
                leverage=leverage,
                fee_rate=0.0006,
                slippage_pct=0.02,
                ticks_per_candle=ticks_per_candle
            )
            
            metrics = engine.run_backtest(df_test, df_signals)
            
            progress_bar.progress(100)
            status_text.text("✅ 全部完成!")
            
            st.balloons()
            
            # ========== 顯示結果 ==========
            st.markdown("---")
            st.subheader("📈 回測結果")
            
            # 關鍵指標
            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            
            with col_r1:
                profit = metrics['final_equity'] - initial_capital
                st.metric(
                    "最終權益",
                    f"${metrics['final_equity']:,.2f}",
                    delta=f"{profit:+,.2f} USDT"
                )
                st.metric("交易次數", metrics['total_trades'])
            
            with col_r2:
                return_pct = metrics['total_return_pct']
                st.metric(
                    "報酬率",
                    f"{return_pct:.2f}%",
                    delta="Tick級別精度"
                )
                st.metric("勝率", f"{metrics['win_rate']:.1f}%")
            
            with col_r3:
                pf = metrics['profit_factor']
                pf_color = "normal" if pf < 1.5 else "inverse"
                st.metric(
                    "盈虧比",
                    f"{pf:.2f}",
                    delta="目標 > 1.5"
                )
                st.metric("夏普比率", f"{metrics['sharpe_ratio']:.2f}")
            
            with col_r4:
                st.metric(
                    "最大回撤",
                    f"{metrics['max_drawdown_pct']:.2f}%"
                )
                st.metric(
                    "平均每筆獲利",
                    f"${metrics['avg_pnl_per_trade']:.2f}"
                )
            
            # 績效評估
            st.markdown("---")
            st.subheader("💡 績效評估")
            
            if return_pct > 15 and metrics['win_rate'] > 50:
                st.success("🎉 優秀! 策略表現非常出色,報酬率和勝率都很高!")
            elif return_pct > 10:
                st.success("✅ 良好! 策略有穩定的獲利能力。")
            elif return_pct > 5:
                st.warning("⚠️ 一般。報酬率偏低,建議調整槓桿或信心度閾值。")
            else:
                st.error("❌ 表現不佳。建議重新訓練或調整參數。")
            
            # 權益曲線
            st.markdown("---")
            st.subheader("📊 權益曲線 (Tick級別模擬)")
            fig = engine.plot_equity_curve()
            st.plotly_chart(fig, use_container_width=True)
            
            # 交易明細
            trades_df = engine.get_trades_dataframe()
            if not trades_df.empty:
                st.markdown("---")
                st.subheader("📝 交易明細 (最近20筆)")
                
                # 顯示格式化的交易記錄
                display_df = trades_df[[
                    'entry_time', 'exit_time', 'direction',
                    'entry_price', 'exit_price', 'pnl_usdt', 'pnl_pct', 'exit_reason'
                ]].tail(20).copy()
                
                # 格式化
                display_df['pnl_usdt'] = display_df['pnl_usdt'].apply(lambda x: f"${x:.2f}")
                display_df['entry_price'] = display_df['entry_price'].apply(lambda x: f"${x:.2f}")
                display_df['exit_price'] = display_df['exit_price'].apply(lambda x: f"${x:.2f}")
                
                st.dataframe(display_df, use_container_width=True)
                
                # 下載按鈕
                csv = trades_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 下載完整交易記錄 CSV",
                    data=csv,
                    file_name=f"{symbol}_strategy_a_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_trades"
                )
            
            # 特徵重要性
            st.markdown("---")
            st.subheader("🎯 模型特徵重要性")
            
            col_fi1, col_fi2 = st.columns(2)
            
            with col_fi1:
                st.markdown("**做多模型 - Top 10 特徵**")
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
                st.markdown("**做空模型 - Top 10 特徵**")
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
            
            # 保存模型選項
            st.markdown("---")
            if st.checkbox("💾 保存此模型供未來使用"):
                model_path = f'models/saved/{symbol}_strategy_a_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
                strategy.save_models(model_path)
                st.success(f"✅ 模型已保存: {model_path}")
                
        except Exception as e:
            st.error(f"❌ 執行錯誤: {str(e)}")
            import traceback
            with st.expander("查看詳細錯誤信息"):
                st.code(traceback.format_exc())
