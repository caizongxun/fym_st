import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core import (
    CryptoDataLoader, FeatureEngineer, ModelTrainer,
    KellyCriterion, RealtimePredictor, ProbabilityCalibrator
)

def render():
    st.title("Probability Calibration Analysis")
    
    st.markdown("""
    Check if your model's predicted probabilities match actual outcomes.
    
    **Why calibration matters:**
    - If model says 70% win probability, it should win ~70% of the time
    - Poor calibration means the model is overconfident or underconfident
    - Affects optimal probability threshold selection
    """)
    
    st.markdown("---")
    
    with st.expander("Configuration", expanded=True):
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
            tp_multiplier = st.number_input("TP Multiplier", value=4.0, step=0.5)
            sl_multiplier = st.number_input("SL Multiplier", value=2.0, step=0.25)
            n_bins = st.number_input("Number of Bins", value=10, min_value=5, max_value=20)
    
    if st.button("Analyze Calibration", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Loading model and data...")
            progress_bar.progress(10)
            
            trainer = ModelTrainer()
            trainer.load_model(model_file)
            
            df = loader.load_klines(symbol, timeframe)
            
            status_text.text("Building features...")
            progress_bar.progress(30)
            
            feature_engineer = FeatureEngineer()
            df_features = feature_engineer.build_features(df)
            
            status_text.text("Generating predictions...")
            progress_bar.progress(50)
            
            kelly = KellyCriterion(tp_multiplier, sl_multiplier, 0.5)
            predictor = RealtimePredictor(trainer, feature_engineer, kelly)
            predictions = predictor.predict_from_completed_bars(df_features)
            
            status_text.text("Simulating trades to get actual results...")
            progress_bar.progress(70)
            
            signals = predictions[predictions['signal'] == 1].copy()
            
            if len(signals) == 0:
                st.warning("No signals generated")
                return
            
            actual_wins = []
            
            for idx, row in signals.iterrows():
                entry_price = row['close']
                atr = row['atr']
                
                tp_price = entry_price + (tp_multiplier * atr)
                sl_price = entry_price - (sl_multiplier * atr)
                
                future_data = df_features.loc[idx:].iloc[1:25]
                
                if len(future_data) == 0:
                    continue
                
                hit_tp = False
                hit_sl = False
                
                for _, frow in future_data.iterrows():
                    if frow['high'] >= tp_price:
                        hit_tp = True
                        break
                    elif frow['low'] <= sl_price:
                        hit_sl = True
                        break
                
                if hit_tp:
                    actual_wins.append(1)
                else:
                    actual_wins.append(0)
            
            signals = signals.iloc[:len(actual_wins)].copy()
            signals['actual_win'] = actual_wins
            
            progress_bar.progress(90)
            status_text.text("Analyzing calibration...")
            
            calibrator = ProbabilityCalibrator()
            calibration_df = calibrator.analyze_calibration(
                signals['actual_win'].values,
                signals['win_probability'].values,
                n_bins=n_bins
            )
            
            progress_bar.progress(100)
            status_text.text("Analysis complete")
            
            st.success(f"Analyzed {len(signals)} signals")
            
            st.markdown("### Calibration Curve")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Perfect Calibration',
                line=dict(color='gray', dash='dash', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=calibration_df['predicted_prob'],
                y=calibration_df['actual_prob'],
                mode='lines+markers',
                name='Model Calibration',
                line=dict(color='blue', width=3),
                marker=dict(size=10, color=calibration_df['count'], colorscale='Viridis', showscale=True,
                           colorbar=dict(title="Samples"))
            ))
            
            fig.update_layout(
                title="Calibration Curve (Predicted vs Actual Win Rate)",
                xaxis_title="Predicted Probability",
                yaxis_title="Actual Win Rate",
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Calibration Metrics")
            
            mse = ((calibration_df['predicted_prob'] - calibration_df['actual_prob']) ** 2).mean()
            mae = (calibration_df['predicted_prob'] - calibration_df['actual_prob']).abs().mean()
            bias = (calibration_df['predicted_prob'] - calibration_df['actual_prob']).mean()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Calibration Error (MSE)", f"{mse:.4f}")
            with col2:
                st.metric("Mean Absolute Error", f"{mae:.4f}")
            with col3:
                bias_label = "Overconfident" if bias > 0 else "Underconfident"
                st.metric("Bias", f"{bias:.4f}", delta=bias_label)
            
            st.markdown("### Detailed Calibration Data")
            st.dataframe(calibration_df, use_container_width=True)
            
            st.markdown("### Interpretation")
            
            if abs(bias) < 0.05:
                st.success("Model is well-calibrated. Predicted probabilities are reliable.")
            elif bias > 0.05:
                st.warning(f"""Model is **overconfident** by {bias:.2%}.
                
**Recommendation:**
- Lower your probability threshold (e.g., use 0.55 instead of 0.65)
- Or retrain with calibration enabled
- Model says it's more confident than it should be
                """)
            else:
                st.warning(f"""Model is **underconfident** by {abs(bias):.2%}.
                
**Recommendation:**
- Increase your probability threshold (e.g., use 0.75 instead of 0.65)
- Model is less confident than reality
- This is safer but may miss opportunities
                """)
            
            if mae > 0.1:
                st.error("""Large calibration error detected.
                
**Actions needed:**
1. Retrain model with `use_calibration=True` (now default)
2. Use larger validation set
3. Check if training and test data distributions match
                """)
            
            st.markdown("### Probability Distribution")
            
            fig2 = make_subplots(rows=1, cols=2, subplot_titles=("Predicted Probabilities", "Actual Win Rates"))
            
            fig2.add_trace(
                go.Histogram(x=signals['win_probability'], nbinsx=20, name="Predicted"),
                row=1, col=1
            )
            
            fig2.add_trace(
                go.Bar(x=calibration_df['bin_center'], y=calibration_df['actual_prob'], name="Actual"),
                row=1, col=2
            )
            
            fig2.update_xaxes(title_text="Probability", row=1, col=1)
            fig2.update_xaxes(title_text="Probability Bin", row=1, col=2)
            fig2.update_yaxes(title_text="Count", row=1, col=1)
            fig2.update_yaxes(title_text="Win Rate", row=1, col=2)
            
            fig2.update_layout(height=400, showlegend=False)
            
            st.plotly_chart(fig2, use_container_width=True)
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())