import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.binance_loader import BinanceDataLoader
from data.feature_engineer import FeatureEngineer
from training.train_trend import TrendModelTrainer
from training.train_reversal import ReversalModelTrainer
from utils.signal_generator import SignalGenerator
from backtesting.engine import BacktestEngine

st.set_page_config(page_title="AI 加密貨幣交易儀表板", layout="wide")

st.title("AI 加密貨幣交易儀表板 - 15分鐘反轉 + ATR 策略")

tabs = st.tabs(["模型訓練", "回測", "即時分析"])

with tabs[0]:
    st.header("模型訓練 (15分鐘趨勢 + 反轉)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        symbol = st.text_input("交易對", value="BTCUSDT")
    
    with col2:
        days = st.number_input("訓練天數", min_value=30, max_value=365, value=180)
    
    with col3:
        oos_size = st.number_input("樣本外數量", min_value=500, max_value=3000, value=1500)
    
    if st.button("開始訓練"):
        with st.spinner("載入數據中..."):
            loader = BinanceDataLoader()
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            df_15m = loader.load_historical_data(symbol, '15m', start_date, end_date)
            
            feature_engineer = FeatureEngineer()
            df_15m = feature_engineer.create_features(df_15m, timeframe='15m')
            
            st.write(f"15分鐘數據: {len(df_15m)} 筆")
        
        st.subheader("訓練 15分鐘趨勢偵測模型")
        with st.spinner("訓練趨勢模型中..."):
            trend_trainer = TrendModelTrainer()
            train_df_trend, oos_df_trend = trend_trainer.prepare_data(df_15m, oos_size=oos_size)
            
            st.write(f"訓練樣本: {len(train_df_trend)}, 樣本外: {len(oos_df_trend)}")
            
            metrics = trend_trainer.train(train_df_trend)
            
            if not oos_df_trend.empty:
                oos_metrics = trend_trainer.evaluate_oos(oos_df_trend)
            
            trend_trainer.save_models(symbol)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("分類準確率", f"{metrics['classification_accuracy']*100:.2f}%")
                if not oos_df_trend.empty:
                    st.metric("樣本外分類準確率", f"{oos_metrics['oos_classification_accuracy']*100:.2f}%")
            
            with col2:
                st.metric("回歸RMSE", f"{metrics['regression_rmse']:.2f}")
                if not oos_df_trend.empty:
                    st.metric("樣本外回歸RMSE", f"{oos_metrics['oos_regression_rmse']:.2f}")
        
        st.subheader("訓練 15分鐘反轉偵測模型")
        with st.spinner("訓練反轉模型中..."):
            reversal_trainer = ReversalModelTrainer()
            train_df_rev, oos_df_rev = reversal_trainer.prepare_data(df_15m, oos_size=oos_size)
            
            st.write(f"訓練樣本: {len(train_df_rev)}, 樣本外: {len(oos_df_rev)}")
            
            metrics = reversal_trainer.train(train_df_rev)
            
            if not oos_df_rev.empty:
                oos_metrics = reversal_trainer.evaluate_oos(oos_df_rev)
            
            reversal_trainer.save_models(symbol)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("方向準確率", f"{metrics['direction_accuracy']*100:.2f}%")
                if not oos_df_rev.empty:
                    st.metric("樣本外方向準確率", f"{oos_metrics['oos_direction_accuracy']*100:.2f}%")
            
            with col2:
                st.metric("概率RMSE", f"{metrics['probability_rmse']:.2f}")
                if not oos_df_rev.empty:
                    st.metric("樣本外概率RMSE", f"{oos_metrics['oos_probability_rmse']:.2f}")
            
            with col3:
                st.metric("支撐MAE", f"{metrics['support_mae']:.2f}")
                if not oos_df_rev.empty:
                    st.metric("支撐MAE %", f"{oos_metrics['oos_support_mae_pct']:.2f}%")
        
        st.success("訓練完成")

with tabs[1]:
    st.header("回測 - 15分鐘反轉 + ATR 策略")
    
    st.info("""
    策略邏輯:
    - 15分鐘趨勢偵測 (短線交易更快反應)
    - 空頭 + 多頭反轉 → 做多
    - 多頭 + 空頭反轉 → 做空
    - ATR止盈止損 (優先級: 止盈/止損 > 反轉信號)
    - 獲利時: 觸發止盈 或 反轉信號 → 反手
    - 虧損時: 觸發止損 → 反手
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        bt_symbol = st.text_input("回測交易對", value="BTCUSDT", key="bt_symbol")
    
    with col2:
        bt_days = st.number_input("回測天數", min_value=7, max_value=180, value=60, key="bt_days")
    
    col3, col4 = st.columns(2)
    with col3:
        initial_capital = st.number_input("初始資金 (USDT)", min_value=10.0, value=100.0)
    
    with col4:
        leverage = st.number_input("槓桿倍數", min_value=1, max_value=20, value=10)
    
    col5, col6 = st.columns(2)
    with col5:
        tp_atr_mult = st.number_input("止盈 ATR 倍數", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
    
    with col6:
        sl_atr_mult = st.number_input("止損 ATR 倍數", min_value=0.5, max_value=3.0, value=1.5, step=0.5)
    
    # NEW: Position sizing controls
    st.subheader("倉位管理設定")
    col7, col8 = st.columns(2)
    
    with col7:
        position_mode = st.selectbox(
            "倉位模式",
            options=['fixed', 'compound'],
            format_func=lambda x: '固定倉位 (使用初始資金比例)' if x == 'fixed' else '複利模式 (使用當前權益比例)',
            index=0
        )
    
    with col8:
        position_size_pct = st.slider(
            "倉位大小 (%)",
            min_value=5,
            max_value=50,
            value=10,
            step=5,
            help="每筆交易使用的資金比例"
        ) / 100
    
    # Display example
    if position_mode == 'fixed':
        example_text = f"範例: 初始資金 {initial_capital}U,每筆開單 {initial_capital * position_size_pct:.1f}U (固定)"
    else:
        example_text = f"範例: 初始 {initial_capital}U 開 {initial_capital * position_size_pct:.1f}U,賺到 120U 則開 {120 * position_size_pct:.1f}U"
    
    st.caption(example_text)
    
    if st.button("執行回測"):
        with st.spinner("載入數據和模型..."):
            loader = BinanceDataLoader()
            feature_engineer = FeatureEngineer()
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=bt_days)
            
            df_15m = loader.load_historical_data(bt_symbol, '15m', start_date, end_date)
            df_15m = feature_engineer.create_features(df_15m, timeframe='15m')
            
            trend_trainer = TrendModelTrainer()
            reversal_trainer = ReversalModelTrainer()
            
            try:
                trend_trainer.load_models(bt_symbol)
                reversal_trainer.load_models(bt_symbol)
            except:
                st.error(f"找不到 {bt_symbol} 的模型,請先訓練")
                st.stop()
            
            df_15m = trend_trainer.predict(df_15m)
            df_15m = reversal_trainer.predict(df_15m)
            
            signal_gen = SignalGenerator()
            df_signals = signal_gen.generate_signals(df_15m)
            df_signals = signal_gen.add_signal_metadata(df_signals)
            
            signal_count = (df_signals['signal'] != 0).sum()
            st.write(f"產生 {signal_count} 個反轉信號")
            
            if signal_count == 0:
                st.warning("未產生信號,檢查條件...")
                st.write("趨勢方向分布:")
                st.write(df_signals['trend_direction'].value_counts())
                st.write("\n反轉方向分布:")
                st.write(df_signals['reversal_direction_pred'].value_counts())
                st.write("\n樣本數據 (最後20筆):")
                st.dataframe(df_signals[['open_time', 'close', 'trend_direction', 'reversal_direction_pred', 'signal']].tail(20))
        
        with st.spinner("執行回測中..."):
            engine = BacktestEngine(
                initial_capital=initial_capital,
                leverage=leverage,
                tp_atr_mult=tp_atr_mult,
                sl_atr_mult=sl_atr_mult,
                position_size_pct=position_size_pct,
                position_mode=position_mode
            )
            
            signals_dict = {bt_symbol: df_signals}
            metrics = engine.run_backtest(signals_dict)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("總交易次數", metrics.get('total_trades', 0))
                st.metric("勝率", f"{metrics.get('win_rate', 0):.2f}%")
            
            with col2:
                st.metric("最終權益", f"${metrics.get('final_equity', 0):.2f}")
                st.metric("總回報", f"{metrics.get('total_return_pct', 0):.2f}%")
            
            with col3:
                st.metric("獲利因子", f"{metrics.get('profit_factor', 0):.2f}")
                st.metric("夏普比率", f"{metrics.get('sharpe_ratio', 0):.2f}")
            
            with col4:
                st.metric("最大回撤", f"{metrics.get('max_drawdown_pct', 0):.2f}%")
                avg_duration = metrics.get('avg_duration_min', 0)
                st.metric("平均持倉時長", f"{avg_duration:.0f}分鐘")
            
            # Exit reasons breakdown
            if 'exit_reasons' in metrics:
                st.subheader("離場原因分布")
                st.write(metrics['exit_reasons'])
            
            st.plotly_chart(engine.plot_equity_curve(), use_container_width=True)
            
            if metrics.get('total_trades', 0) > 0:
                st.subheader("交易明細")
                trades_df = engine.get_trades_dataframe()
                
                # Display detailed Chinese columns
                display_cols = [
                    'symbol', '方向', '進場時間', '離場時間',
                    '進場價格', '離場價格', 'position_value', '手續費', '損益率',
                    '離場原因', '持倉時長(分)', '進場趨勢', '離場趨勢'
                ]
                
                st.dataframe(trades_df[display_cols].round(4))
                
                # Download button for full data
                csv = trades_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="下載完整交易記錄 (CSV)",
                    data=csv,
                    file_name=f'{bt_symbol}_backtest_{datetime.now().strftime("%Y%m%d")}.csv',
                    mime='text/csv'
                )

with tabs[2]:
    st.header("即時分析")
    
    live_symbol = st.text_input("交易對", value="BTCUSDT", key="live_symbol")
    
    if st.button("分析當前市場"):
        with st.spinner("載入數據..."):
            loader = BinanceDataLoader()
            feature_engineer = FeatureEngineer()
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            df_15m = loader.load_historical_data(live_symbol, '15m', start_date, end_date)
            df_15m = feature_engineer.create_features(df_15m, timeframe='15m')
            
            trend_trainer = TrendModelTrainer()
            reversal_trainer = ReversalModelTrainer()
            
            try:
                trend_trainer.load_models(live_symbol)
                reversal_trainer.load_models(live_symbol)
            except:
                st.error(f"找不到 {live_symbol} 的模型")
                st.stop()
            
            df_15m = trend_trainer.predict(df_15m)
            df_15m = reversal_trainer.predict(df_15m)
            
            signal_gen = SignalGenerator()
            df_signals = signal_gen.generate_signals(df_15m)
            df_signals = signal_gen.add_signal_metadata(df_signals)
            
            latest = df_signals.iloc[-1]
            
            st.subheader(f"當前分析 - {live_symbol}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("當前價格", f"${latest['close']:.2f}")
                trend_map = {1: '多頭', -1: '空頭', 0: '盤整'}
                trend_name = trend_map.get(int(latest['trend_direction']), '未知')
                st.metric("趨勢方向 (15分鐘)", trend_name)
                st.metric("ATR", f"{latest.get('15m_atr', 0):.2f}")
            
            with col2:
                reversal_name = latest.get('reversal_name', '無')
                st.metric("反轉信號", reversal_name)
                st.metric("反轉概率", f"{latest['reversal_prob_pred']:.1f}%")
            
            with col3:
                signal_name = latest.get('signal_name', 'HOLD')
                signal_name_cn = {'LONG': '做多', 'SHORT': '做空', 'HOLD': '觀望'}.get(signal_name, signal_name)
                color = 'green' if signal_name == 'LONG' else ('red' if signal_name == 'SHORT' else 'gray')
                st.markdown(f"### 操作建議: :{color}[{signal_name_cn}]")
                
                atr = latest.get('15m_atr', 0)
                if signal_name == 'LONG':
                    st.metric("止盈目標", f"${latest['close'] + atr*tp_atr_mult:.2f}")
                    st.metric("止損位置", f"${latest['close'] - atr*sl_atr_mult:.2f}")
                elif signal_name == 'SHORT':
                    st.metric("止盈目標", f"${latest['close'] - atr*tp_atr_mult:.2f}")
                    st.metric("止損位置", f"${latest['close'] + atr*sl_atr_mult:.2f}")
            
            st.subheader("最近信號")
            recent_signals = df_signals[df_signals['signal'] != 0].tail(10)
            if not recent_signals.empty:
                st.dataframe(recent_signals[[
                    'open_time', 'close', 'signal_name', 'reversal_name', 
                    'reversal_prob_pred', 'trend_direction'
                ]].round(2))
            else:
                st.info("無最近信號")