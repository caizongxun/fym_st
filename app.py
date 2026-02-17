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

st.set_page_config(page_title="AI Crypto Trading Dashboard", layout="wide")

st.title("AI Crypto Trading Dashboard - Pure Reversal Strategy")

tabs = st.tabs(["Model Training", "Backtest", "Live Analysis"])

with tabs[0]:
    st.header("Model Training (Reversal Only)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        symbol = st.text_input("Symbol", value="BTCUSDT")
    
    with col2:
        days = st.number_input("Training Days", min_value=30, max_value=365, value=180)
    
    with col3:
        oos_size = st.number_input("OOS Size", min_value=500, max_value=3000, value=1500)
    
    if st.button("Train Models"):
        with st.spinner("Loading data..."):
            loader = BinanceDataLoader()
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            df_1h = loader.load_historical_data(symbol, '1h', start_date, end_date)
            df_15m = loader.load_historical_data(symbol, '15m', start_date, end_date)
            
            feature_engineer = FeatureEngineer()
            df_1h = feature_engineer.create_features(df_1h, timeframe='1h')
            df_15m = feature_engineer.create_features(df_15m, timeframe='15m')
            
            st.write(f"1h data: {len(df_1h)} samples")
            st.write(f"15m data: {len(df_15m)} samples")
        
        st.subheader("Training Trend Detection Model")
        with st.spinner("Training trend model..."):
            trend_trainer = TrendModelTrainer()
            train_df_trend, oos_df_trend = trend_trainer.prepare_data(df_1h, oos_size=oos_size)
            
            st.write(f"Training samples: {len(train_df_trend)}, OOS samples: {len(oos_df_trend)}")
            
            metrics = trend_trainer.train(train_df_trend)
            
            if not oos_df_trend.empty:
                oos_metrics = trend_trainer.evaluate_oos(oos_df_trend)
            
            trend_trainer.save_models(symbol)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Classification Accuracy", f"{metrics['classification_accuracy']*100:.2f}%")
                if not oos_df_trend.empty:
                    st.metric("OOS Classification Accuracy", f"{oos_metrics['oos_classification_accuracy']*100:.2f}%")
            
            with col2:
                st.metric("Regression RMSE", f"{metrics['regression_rmse']:.2f}")
                if not oos_df_trend.empty:
                    st.metric("OOS Regression RMSE", f"{oos_metrics['oos_regression_rmse']:.2f}")
        
        st.subheader("Training Reversal Detection Model")
        with st.spinner("Training reversal model..."):
            reversal_trainer = ReversalModelTrainer()
            train_df_rev, oos_df_rev = reversal_trainer.prepare_data(df_15m, oos_size=oos_size)
            
            st.write(f"Training samples: {len(train_df_rev)}, OOS samples: {len(oos_df_rev)}")
            
            metrics = reversal_trainer.train(train_df_rev)
            
            if not oos_df_rev.empty:
                oos_metrics = reversal_trainer.evaluate_oos(oos_df_rev)
            
            reversal_trainer.save_models(symbol)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Direction Accuracy", f"{metrics['direction_accuracy']*100:.2f}%")
                if not oos_df_rev.empty:
                    st.metric("OOS Direction Accuracy", f"{oos_metrics['oos_direction_accuracy']*100:.2f}%")
            
            with col2:
                st.metric("Probability RMSE", f"{metrics['probability_rmse']:.2f}")
                if not oos_df_rev.empty:
                    st.metric("OOS Probability RMSE", f"{oos_metrics['oos_probability_rmse']:.2f}")
            
            with col3:
                st.metric("Support MAE", f"{metrics['support_mae']:.2f}")
                if not oos_df_rev.empty:
                    st.metric("Support MAE %", f"{oos_metrics['oos_support_mae_pct']:.2f}%")
        
        st.success("Training complete")

with tabs[1]:
    st.header("Backtest - Pure Reversal Strategy")
    
    st.info("""
    Strategy Logic:
    - Uses indicator-based trend detection (EMA, MACD, ADX) to know current market direction
    - When in DOWNTREND and BULLISH reversal appears -> GO LONG
    - When in UPTREND and BEARISH reversal appears -> GO SHORT
    - Exit and flip position on next reversal signal
    - No TP/SL, no filters, pure flip-flop system
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        bt_symbol = st.text_input("Backtest Symbol", value="BTCUSDT", key="bt_symbol")
    
    with col2:
        bt_days = st.number_input("Backtest Days", min_value=7, max_value=180, value=60, key="bt_days")
    
    col3, col4 = st.columns(2)
    with col3:
        initial_capital = st.number_input("Initial Capital (USDT)", min_value=10.0, value=100.0)
    
    with col4:
        leverage = st.number_input("Leverage", min_value=1, max_value=20, value=10)
    
    if st.button("Run Backtest"):
        with st.spinner("Loading data and models..."):
            loader = BinanceDataLoader()
            feature_engineer = FeatureEngineer()
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=bt_days)
            
            df_1h = loader.load_historical_data(bt_symbol, '1h', start_date, end_date)
            df_15m = loader.load_historical_data(bt_symbol, '15m', start_date, end_date)
            
            df_1h = feature_engineer.create_features(df_1h, timeframe='1h')
            df_15m = feature_engineer.create_features(df_15m, timeframe='15m')
            
            trend_trainer = TrendModelTrainer()
            reversal_trainer = ReversalModelTrainer()
            
            try:
                trend_trainer.load_models(bt_symbol)
                reversal_trainer.load_models(bt_symbol)
            except:
                st.error(f"Models not found for {bt_symbol}. Please train first.")
                st.stop()
            
            df_1h = trend_trainer.predict(df_1h)
            df_15m = reversal_trainer.predict(df_15m)
            
            df_1h['timeframe'] = '1h'
            df_15m['timeframe'] = '15m'
            
            # Use 'min' instead of deprecated 'T'
            df_1h_resampled = df_1h.set_index('open_time').resample('15min').last().reset_index()
            df_1h_resampled = df_1h_resampled.add_suffix('_1h')
            df_1h_resampled.rename(columns={'open_time_1h': 'open_time'}, inplace=True)
            
            df_combined = pd.merge(df_15m, df_1h_resampled, on='open_time', how='left')
            df_combined['trend_direction'] = df_combined['trend_direction_1h'].fillna(0)
            
            # NO PARAMETERS - pure reversal
            signal_gen = SignalGenerator()
            df_signals = signal_gen.generate_signals(df_combined)
            df_signals = signal_gen.add_signal_metadata(df_signals)
            
            signal_count = (df_signals['signal'] != 0).sum()
            st.write(f"Generated {signal_count} reversal signals")
            
            # Debug info
            if signal_count == 0:
                st.warning("No signals generated. Checking conditions...")
                st.write("Trend direction distribution:")
                st.write(df_signals['trend_direction'].value_counts())
                st.write("\nReversal direction distribution:")
                st.write(df_signals['reversal_direction_pred'].value_counts())
                st.write("\nSample data (last 20 rows):")
                st.dataframe(df_signals[['open_time', 'close', 'trend_direction', 'reversal_direction_pred', 'signal']].tail(20))
        
        with st.spinner("Running backtest..."):
            engine = BacktestEngine(
                initial_capital=initial_capital,
                leverage=leverage,
                tp_atr_mult=None,
                sl_atr_mult=None
            )
            
            signals_dict = {bt_symbol: df_signals}
            metrics = engine.run_backtest(signals_dict)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Trades", metrics.get('total_trades', 0))
                st.metric("Win Rate", f"{metrics.get('win_rate', 0):.2f}%")
            
            with col2:
                st.metric("Final Equity", f"${metrics.get('final_equity', 0):.2f}")
                st.metric("Total Return", f"{metrics.get('total_return_pct', 0):.2f}%")
            
            with col3:
                st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
                st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
            
            with col4:
                st.metric("Max Drawdown", f"{metrics.get('max_drawdown_pct', 0):.2f}%")
                avg_duration = metrics.get('avg_duration_min', metrics.get('avg_duration', 0))
                st.metric("Avg Trade Duration", f"{avg_duration:.0f}m")
            
            st.plotly_chart(engine.plot_equity_curve(), use_container_width=True)
            
            if metrics.get('total_trades', 0) > 0:
                st.subheader("Trade Details")
                trades_df = engine.get_trades_dataframe()
                st.dataframe(trades_df[[
                    'symbol', 'direction', 'entry_time', 'exit_time', 
                    'entry_price', 'exit_price', 'pnl', 'pnl_pct', 'exit_reason'
                ]].round(4))

with tabs[2]:
    st.header("Live Analysis")
    
    live_symbol = st.text_input("Symbol", value="BTCUSDT", key="live_symbol")
    
    if st.button("Analyze Current Market"):
        with st.spinner("Loading data..."):
            loader = BinanceDataLoader()
            feature_engineer = FeatureEngineer()
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # Increased to 30 days for better indicators
            
            df_1h = loader.load_historical_data(live_symbol, '1h', start_date, end_date)
            df_15m = loader.load_historical_data(live_symbol, '15m', start_date, end_date)
            
            df_1h = feature_engineer.create_features(df_1h, timeframe='1h')
            df_15m = feature_engineer.create_features(df_15m, timeframe='15m')
            
            trend_trainer = TrendModelTrainer()
            reversal_trainer = ReversalModelTrainer()
            
            try:
                trend_trainer.load_models(live_symbol)
                reversal_trainer.load_models(live_symbol)
            except:
                st.error(f"Models not found for {live_symbol}")
                st.stop()
            
            df_1h = trend_trainer.predict(df_1h)
            df_15m = reversal_trainer.predict(df_15m)
            
            df_1h_resampled = df_1h.set_index('open_time').resample('15min').last().reset_index()
            df_1h_resampled = df_1h_resampled.add_suffix('_1h')
            df_1h_resampled.rename(columns={'open_time_1h': 'open_time'}, inplace=True)
            
            df_combined = pd.merge(df_15m, df_1h_resampled, on='open_time', how='left')
            df_combined['trend_direction'] = df_combined['trend_direction_1h'].fillna(0)
            
            signal_gen = SignalGenerator()
            df_signals = signal_gen.generate_signals(df_combined)
            df_signals = signal_gen.add_signal_metadata(df_signals)
            
            latest = df_signals.iloc[-1]
            
            st.subheader(f"Current Analysis - {live_symbol}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Price", f"${latest['close']:.2f}")
                trend_map = {1: 'BULL', -1: 'BEAR', 0: 'NEUTRAL'}
                trend_name = trend_map.get(int(latest['trend_direction']), 'UNKNOWN')
                st.metric("Trend Direction", trend_name)
            
            with col2:
                reversal_name = latest.get('reversal_name', 'None')
                st.metric("Reversal Signal", reversal_name)
                st.metric("Reversal Probability", f"{latest['reversal_prob_pred']:.1f}%")
            
            with col3:
                signal_name = latest.get('signal_name', 'HOLD')
                color = 'green' if signal_name == 'LONG' else ('red' if signal_name == 'SHORT' else 'gray')
                st.markdown(f"### Action: :{color}[{signal_name}]")
                st.metric("Support Level", f"${latest['support_pred']:.2f}")
                st.metric("Resistance Level", f"{latest['resistance_pred']:.2f}")
            
            st.subheader("Recent Signals")
            recent_signals = df_signals[df_signals['signal'] != 0].tail(10)
            if not recent_signals.empty:
                st.dataframe(recent_signals[[
                    'open_time', 'close', 'signal_name', 'reversal_name', 
                    'reversal_prob_pred', 'trend_direction'
                ]].round(2))
            else:
                st.info("No recent signals")