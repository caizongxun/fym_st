import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta

from config import Config
from data.data_loader import DataLoader
from data.feature_engineer import FeatureEngineer
from training.train_trend import TrendModelTrainer
from training.train_volatility import VolatilityModelTrainer
from training.train_reversal import ReversalModelTrainer
from backtesting.engine import BacktestEngine
from utils.signal_generator import SignalGenerator

st.set_page_config(
    page_title="FYM_ST - AI Trading System",
    page_icon="chart_with_upwards_trend",
    layout="wide"
)

st.title("FYM_ST - Multi-Timeframe AI Trading System")

@st.cache_resource
def init_data_loader():
    return DataLoader(
        hf_repo_id=Config.HF_DATASET_ID,
        binance_api_key=Config.BINANCE_API_KEY,
        binance_api_secret=Config.BINANCE_API_SECRET
    )

@st.cache_resource
def init_feature_engineer():
    return FeatureEngineer()

tabs = st.tabs(["Model Training", "Backtesting", "Live Signals", "Documentation"])

# TAB 1: Model Training
with tabs[0]:
    st.header("Model Training")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Training Configuration")
        
        selected_symbol = st.selectbox(
            "Select Symbol for Training",
            Config.SUPPORTED_SYMBOLS,
            index=Config.SUPPORTED_SYMBOLS.index('BTCUSDT')
        )
        
        train_size = st.number_input(
            "Training Data Size (candles)",
            min_value=1000,
            max_value=10000,
            value=Config.TRAIN_SIZE,
            step=500
        )
        
        oos_size = st.number_input(
            "OOS Validation Size (candles)",
            min_value=500,
            max_value=3000,
            value=Config.OOS_SIZE,
            step=100
        )
        
        st.info(f"Training will use {train_size} candles, with {oos_size} reserved for OOS validation")
    
    with col2:
        st.subheader("Model Selection")
        
        train_trend_model = st.checkbox("Train Trend Detection Model (1h)", value=True)
        train_volatility_model = st.checkbox("Train Volatility Prediction Model (15m)", value=True)
        train_reversal_model = st.checkbox("Train Reversal Detection Model (15m)", value=True)
    
    if st.button("Start Training", type="primary"):
        with st.spinner("Loading data and computing features..."):
            loader = init_data_loader()
            engineer = init_feature_engineer()
            
            # Load data from HuggingFace
            df_1h = loader.load_from_huggingface(selected_symbol, '1h')
            df_15m = loader.load_from_huggingface(selected_symbol, '15m')
            
            if df_1h.empty or df_15m.empty:
                st.error("Failed to load data. Check your configuration.")
                st.stop()
            
            # Use only last N candles for training
            df_1h = df_1h.tail(train_size + oos_size + 200).copy()
            df_15m = df_15m.tail((train_size + oos_size) * 4 + 200).copy()
            
            st.success(f"Loaded {len(df_1h)} x 1h candles and {len(df_15m)} x 15m candles")
            
            # Compute features
            df_1h = engineer.compute_features(df_1h, '1h_')
            df_15m = engineer.compute_features(df_15m, '15m_')
            
            # Merge timeframes
            df_merged = engineer.merge_timeframes(df_15m, df_1h)
            
            st.success("Feature engineering complete")
        
        # Train Trend Model
        if train_trend_model:
            st.subheader("Training Trend Detection Model")
            with st.spinner("Training trend model..."):
                trend_trainer = TrendModelTrainer(model_dir=Config.MODEL_DIR)
                train_df, oos_df = trend_trainer.prepare_data(df_1h, oos_size=oos_size)
                
                st.write(f"Training samples: {len(train_df)}, OOS samples: {len(oos_df)}")
                
                metrics = trend_trainer.train(train_df)
                
                if not oos_df.empty:
                    oos_metrics = trend_trainer.evaluate_oos(oos_df)
                    metrics.update(oos_metrics)
                
                trend_trainer.save_models(selected_symbol)
                
                col_a, col_b = st.columns(2)
                col_a.metric("Classification Accuracy", f"{metrics['classification_accuracy']:.2%}")
                col_b.metric("Regression RMSE", f"{metrics['regression_rmse']:.2f}")
                
                if 'oos_classification_accuracy' in metrics:
                    col_a.metric("OOS Classification Accuracy", f"{metrics['oos_classification_accuracy']:.2%}")
                    col_b.metric("OOS Regression RMSE", f"{metrics['oos_regression_rmse']:.2f}")
        
        # Train Volatility Model
        if train_volatility_model:
            st.subheader("Training Volatility Prediction Model")
            with st.spinner("Training volatility model..."):
                vol_trainer = VolatilityModelTrainer(model_dir=Config.MODEL_DIR)
                train_df, oos_df = vol_trainer.prepare_data(df_15m, oos_size=oos_size)
                
                st.write(f"Training samples: {len(train_df)}, OOS samples: {len(oos_df)}")
                
                metrics = vol_trainer.train(train_df)
                
                if not oos_df.empty:
                    oos_metrics = vol_trainer.evaluate_oos(oos_df)
                    metrics.update(oos_metrics)
                
                vol_trainer.save_models(selected_symbol)
                
                col_a, col_b = st.columns(2)
                col_a.metric("Regime Accuracy", f"{metrics['regime_accuracy']:.2%}")
                col_b.metric("Trend Init RMSE", f"{metrics['trend_init_rmse']:.2f}")
                
                if 'oos_regime_accuracy' in metrics:
                    col_a.metric("OOS Regime Accuracy", f"{metrics['oos_regime_accuracy']:.2%}")
                    col_b.metric("OOS Trend Init RMSE", f"{metrics['oos_trend_init_rmse']:.2f}")
        
        # Train Reversal Model
        if train_reversal_model:
            st.subheader("Training Reversal Detection Model")
            with st.spinner("Training reversal model..."):
                rev_trainer = ReversalModelTrainer(model_dir=Config.MODEL_DIR)
                train_df, oos_df = rev_trainer.prepare_data(df_15m, oos_size=oos_size)
                
                st.write(f"Training samples: {len(train_df)}, OOS samples: {len(oos_df)}")
                
                metrics = rev_trainer.train(train_df)
                
                if not oos_df.empty:
                    oos_metrics = rev_trainer.evaluate_oos(oos_df)
                    metrics.update(oos_metrics)
                
                rev_trainer.save_models(selected_symbol)
                
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Direction Accuracy", f"{metrics['direction_accuracy']:.2%}")
                col_b.metric("Probability RMSE", f"{metrics['probability_rmse']:.2f}")
                col_c.metric("Support MAE", f"{metrics['support_mae']:.2f}")
                
                if 'oos_direction_accuracy' in metrics:
                    col_a.metric("OOS Direction Accuracy", f"{metrics['oos_direction_accuracy']:.2%}")
                    col_b.metric("OOS Probability RMSE", f"{metrics['oos_probability_rmse']:.2f}")
        
        st.success("All models trained successfully!")

# TAB 2: Backtesting
with tabs[1]:
    st.header("Backtesting Engine")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Capital & Risk")
        
        initial_capital = st.number_input(
            "Initial Capital (USDT)",
            min_value=1.0,
            max_value=10000.0,
            value=Config.DEFAULT_CAPITAL,
            step=1.0
        )
        
        leverage = st.slider(
            "Leverage",
            min_value=1,
            max_value=50,
            value=Config.DEFAULT_LEVERAGE
        )
        
        position_size_pct = st.slider(
            "Position Size (% of capital)",
            min_value=10,
            max_value=100,
            value=int(Config.DEFAULT_POSITION_SIZE * 100)
        ) / 100
        
        max_positions = st.number_input(
            "Max Concurrent Positions",
            min_value=1,
            max_value=10,
            value=Config.DEFAULT_MAX_POSITIONS
        )
    
    with col2:
        st.subheader("Stop Loss & Take Profit")
        
        tp_atr_mult = st.number_input(
            "Take Profit (ATR multiplier)",
            min_value=1.0,
            max_value=10.0,
            value=Config.DEFAULT_TP_ATR,
            step=0.5
        )
        
        sl_atr_mult = st.number_input(
            "Stop Loss (ATR multiplier)",
            min_value=0.5,
            max_value=5.0,
            value=Config.DEFAULT_SL_ATR,
            step=0.5
        )
    
    with col3:
        st.subheader("Backtest Period")
        
        backtest_days = st.number_input(
            "Backtest Days",
            min_value=7,
            max_value=365,
            value=30,
            step=7
        )
        
        selected_symbols_bt = st.multiselect(
            "Select Symbols",
            Config.SUPPORTED_SYMBOLS,
            default=['BTCUSDT', 'ETHUSDT']
        )
    
    if st.button("Run Backtest", type="primary"):
        if not selected_symbols_bt:
            st.error("Please select at least one symbol")
            st.stop()
        
        with st.spinner("Loading data and models..."):
            loader = init_data_loader()
            engineer = init_feature_engineer()
            
            # Check if models exist
            missing_models = []
            for sym in selected_symbols_bt:
                for model_type in ['trend_classifier', 'volatility_regime', 'reversal_direction']:
                    model_path = os.path.join(Config.MODEL_DIR, f"{sym}_{model_type}.pkl")
                    if not os.path.exists(model_path):
                        missing_models.append(f"{sym} - {model_type}")
            
            if missing_models:
                st.error(f"Missing models: {', '.join(missing_models)}. Please train them first.")
                st.stop()
            
            signals_dict = {}
            
            for symbol in selected_symbols_bt:
                st.write(f"Processing {symbol}...")
                
                # Load Binance data for backtesting
                df_1h = loader.load_from_binance(symbol, '1h', days=backtest_days)
                df_15m = loader.load_from_binance(symbol, '15m', days=backtest_days)
                
                if df_1h.empty or df_15m.empty:
                    st.warning(f"Could not load data for {symbol}, skipping")
                    continue
                
                # Get completed candles only
                df_1h = loader.get_completed_candles(df_1h)
                df_15m = loader.get_completed_candles(df_15m)
                
                # Compute features
                df_1h = engineer.compute_features(df_1h, '1h_')
                df_15m = engineer.compute_features(df_15m, '15m_')
                df_merged = engineer.merge_timeframes(df_15m, df_1h)
                
                # Load models and predict
                trend_trainer = TrendModelTrainer(model_dir=Config.MODEL_DIR)
                trend_trainer.load_models(symbol)
                
                vol_trainer = VolatilityModelTrainer(model_dir=Config.MODEL_DIR)
                vol_trainer.load_models(symbol)
                
                rev_trainer = ReversalModelTrainer(model_dir=Config.MODEL_DIR)
                rev_trainer.load_models(symbol)
                
                # Make predictions
                df_with_trend = trend_trainer.predict(df_merged)
                df_with_vol = vol_trainer.predict(df_with_trend)
                df_with_rev = rev_trainer.predict(df_with_vol)
                
                # Generate signals
                signal_gen = SignalGenerator(
                    min_reversal_prob=Config.MIN_REVERSAL_PROB,
                    min_trend_strength=Config.MIN_TREND_SCORE,
                    volume_multiplier=Config.VOLUME_MULTIPLIER
                )
                df_signals = signal_gen.generate_signals(df_with_rev)
                df_signals = signal_gen.add_signal_metadata(df_signals)
                
                signals_dict[symbol] = df_signals
            
            st.success(f"Loaded and processed {len(signals_dict)} symbols")
        
        with st.spinner("Running backtest..."):
            engine = BacktestEngine(
                initial_capital=initial_capital,
                leverage=leverage,
                tp_atr_mult=tp_atr_mult,
                sl_atr_mult=sl_atr_mult,
                position_size_pct=position_size_pct,
                max_positions=max_positions,
                maker_fee=Config.MAKER_FEE,
                taker_fee=Config.TAKER_FEE
            )
            
            metrics = engine.run_backtest(signals_dict)
        
        st.subheader("Backtest Results")
        
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Final Equity", f"{metrics['final_equity']:.2f} USDT")
        col_b.metric("Total Return", f"{metrics['total_return_pct']:.2f}%")
        col_c.metric("Win Rate", f"{metrics['win_rate']:.2f}%")
        col_d.metric("Total Trades", metrics['total_trades'])
        
        col_e, col_f, col_g, col_h = st.columns(4)
        col_e.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
        col_f.metric("Max Drawdown", f"{metrics['max_drawdown_pct']:.2f}%")
        col_g.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        col_h.metric("Avg Duration", f"{metrics['avg_duration_min']:.0f} min")
        
        # Equity curve
        st.subheader("Equity Curve")
        fig_equity = engine.plot_equity_curve()
        st.plotly_chart(fig_equity, use_container_width=True)
        
        # Trade history
        st.subheader("Trade History")
        trades_df = engine.get_trades_dataframe()
        if not trades_df.empty:
            st.dataframe(trades_df.sort_values('entry_time', ascending=False), use_container_width=True)
            
            # Download button
            csv = trades_df.to_csv(index=False)
            st.download_button(
                "Download Trade History",
                csv,
                f"backtest_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
        else:
            st.info("No trades executed")
        
        # Trades per symbol
        if 'trades_per_symbol' in metrics and metrics['trades_per_symbol']:
            st.subheader("Trades by Symbol")
            symbol_trades = pd.DataFrame(
                list(metrics['trades_per_symbol'].items()),
                columns=['Symbol', 'Trade Count']
            )
            st.bar_chart(symbol_trades.set_index('Symbol'))

st.sidebar.title("System Info")
st.sidebar.info(
    f"Supported Symbols: {len(Config.SUPPORTED_SYMBOLS)}\n\n"
    f"Trading TF: {Config.TRADING_TIMEFRAME}\n"
    f"Trend TF: {Config.TREND_TIMEFRAME}\n\n"
    f"Model Directory: {Config.MODEL_DIR}"
)

if st.sidebar.button("Clear Cache"):
    st.cache_resource.clear()
    st.sidebar.success("Cache cleared!")
