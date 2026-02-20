import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core import (
    CryptoDataLoader, FeatureEngineer, ModelTrainer,
    KellyCriterion, Backtester, RealtimePredictor,
    SignalFilter, StrategyOptimizer
)

def render():
    st.title("Strategy Optimization")
    
    st.markdown("""
    Optimize your strategy parameters to improve profitability:
    - **Probability Threshold**: Find optimal minimum win probability
    - **TP/SL Ratio**: Optimize take-profit and stop-loss multipliers
    - **Signal Filters**: Test different filter combinations
    """)
    
    st.markdown("---")
    
    with st.expander("Optimization Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            model_files = [f for f in os.listdir("trading_system/models") if f.endswith('.pkl')] if os.path.exists("trading_system/models") else []
            
            if len(model_files) == 0:
                st.warning("No trained models found. Train a model first.")
                return
            
            model_file = st.selectbox("Select Model", model_files)
            
            loader = CryptoDataLoader()
            symbol = st.selectbox("Symbol", loader.get_available_symbols(), index=10)
            timeframe = st.selectbox("Timeframe", loader.get_available_timeframes(), index=1)
        
        with col2:
            initial_capital = st.number_input("Initial Capital", value=10000.0, step=1000.0)
            commission_rate = st.number_input("Commission Rate", value=0.001, step=0.0001, format="%.4f")
            kelly_fraction = st.number_input("Kelly Fraction", value=0.5, step=0.1)
    
    optimization_type = st.radio(
        "Optimization Type",
        ["Probability Threshold", "TP/SL Ratio", "Signal Filters"],
        horizontal=True
    )
    
    if st.button("Run Optimization", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Loading model and data...")
            progress_bar.progress(10)
            
            trainer = ModelTrainer()
            trainer.load_model(model_file)
            
            df = loader.load_klines(symbol, timeframe)
            
            status_text.text("Building features...")
            progress_bar.progress(25)
            
            feature_engineer = FeatureEngineer()
            df_features = feature_engineer.build_features(df)
            
            status_text.text("Generating predictions...")
            progress_bar.progress(40)
            
            kelly = KellyCriterion(2.5, 1.5, kelly_fraction)
            predictor = RealtimePredictor(trainer, feature_engineer, kelly)
            predictions = predictor.predict_from_completed_bars(df_features)
            
            st.info(f"Generated predictions for {len(predictions)} bars")
            
            status_text.text("Running optimization...")
            progress_bar.progress(55)
            
            backtester = Backtester(initial_capital, commission_rate, 0.0005)
            signal_filter = SignalFilter()
            optimizer = StrategyOptimizer(backtester, predictor, signal_filter)
            
            if optimization_type == "Probability Threshold":
                status_text.text("Optimizing probability threshold...")
                
                thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
                results = optimizer.optimize_probability_threshold(
                    predictions,
                    thresholds=thresholds,
                    tp_multiplier=2.5,
                    sl_multiplier=1.5
                )
                
                progress_bar.progress(100)
                status_text.text("Optimization complete")
                
                if results['best_threshold'] is None:
                    st.error("Optimization failed. No valid results.")
                    return
                
                st.success(f"Best probability threshold: {results['best_threshold']}")
                
                st.markdown("### Optimization Results")
                
                results_df = results['results']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Best Threshold", f"{results['best_threshold']:.2f}")
                with col2:
                    st.metric("Best Return", f"{results['best_stats']['total_return']*100:.2f}%")
                with col3:
                    st.metric("Win Rate", f"{results['best_stats']['win_rate']*100:.2f}%")
                
                st.dataframe(results_df, use_container_width=True)
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=results_df['threshold'],
                    y=results_df['total_return'] * 100,
                    mode='lines+markers',
                    name='Total Return',
                    line=dict(color='blue', width=2),
                    marker=dict(size=8)
                ))
                
                fig.update_layout(
                    title="Total Return vs Probability Threshold",
                    xaxis_title="Probability Threshold",
                    yaxis_title="Total Return (%)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif optimization_type == "TP/SL Ratio":
                status_text.text("Optimizing TP/SL ratios...")
                
                results = optimizer.optimize_tp_sl_ratio(
                    predictions,
                    tp_multipliers=[2.0, 2.5, 3.0, 3.5, 4.0],
                    sl_multipliers=[1.0, 1.25, 1.5, 1.75, 2.0],
                    min_probability=0.65
                )
                
                progress_bar.progress(100)
                status_text.text("Optimization complete")
                
                if results['best_tp'] is None:
                    st.error("Optimization failed. No valid results.")
                    return
                
                st.success(f"Best TP/SL: {results['best_tp']}/{results['best_sl']} (RR: {results['best_stats']['risk_reward_ratio']:.2f})")
                
                st.markdown("### Optimization Results")
                
                results_df = results['results']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Best TP", f"{results['best_tp']:.2f}x")
                with col2:
                    st.metric("Best SL", f"{results['best_sl']:.2f}x")
                with col3:
                    st.metric("Best Return", f"{results['best_stats']['total_return']*100:.2f}%")
                with col4:
                    st.metric("Profit Factor", f"{results['best_stats']['profit_factor']:.2f}")
                
                st.dataframe(results_df.sort_values('total_return', ascending=False), use_container_width=True)
                
                fig = go.Figure(data=go.Heatmap(
                    z=results_df.pivot(index='sl_multiplier', columns='tp_multiplier', values='total_return').values * 100,
                    x=results_df['tp_multiplier'].unique(),
                    y=results_df['sl_multiplier'].unique(),
                    colorscale='RdYlGn',
                    text=results_df.pivot(index='sl_multiplier', columns='tp_multiplier', values='total_return').values * 100,
                    texttemplate='%{text:.1f}%',
                    colorbar=dict(title="Return (%)")
                ))
                
                fig.update_layout(
                    title="Total Return Heatmap (TP vs SL)",
                    xaxis_title="TP Multiplier",
                    yaxis_title="SL Multiplier",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif optimization_type == "Signal Filters":
                status_text.text("Testing filter combinations...")
                
                results = optimizer.optimize_filters(
                    predictions,
                    tp_multiplier=2.5,
                    sl_multiplier=1.5
                )
                
                progress_bar.progress(100)
                status_text.text("Optimization complete")
                
                if results['best_config'] is None:
                    st.error("Optimization failed. No valid results.")
                    return
                
                st.success(f"Best filter configuration: {results['best_config']}")
                
                st.markdown("### Filter Comparison")
                
                results_df = results['results']
                
                st.dataframe(results_df.sort_values('total_return', ascending=False), use_container_width=True)
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=results_df['config_name'],
                    y=results_df['total_return'] * 100,
                    text=results_df['total_return'].apply(lambda x: f"{x*100:.1f}%"),
                    textposition='auto',
                    marker_color=['green' if x > 0 else 'red' for x in results_df['total_return']]
                ))
                
                fig.update_layout(
                    title="Total Return by Filter Configuration",
                    xaxis_title="Configuration",
                    yaxis_title="Total Return (%)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### Recommendation")
                best_config = results['best_config']
                best_stats = results['best_stats']
                
                st.info(f"""
                **Recommended Configuration**: {best_config}
                
                - Total Return: {best_stats['total_return']*100:.2f}%
                - Win Rate: {best_stats['win_rate']*100:.2f}%
                - Profit Factor: {best_stats['profit_factor']:.2f}
                - Number of Signals: {int(best_stats['num_signals'])}
                - Max Drawdown: {best_stats['max_drawdown']*100:.2f}%
                """)
        
        except Exception as e:
            st.error(f"Optimization failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())