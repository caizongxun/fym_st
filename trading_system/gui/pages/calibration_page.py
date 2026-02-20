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
    TripleBarrierLabeling, EventFilter
)

def render():
    st.title("機率校準分析")
    
    st.markdown("""
    檢查模型預測機率是否與實際結果匹配:
    
    **重要性**:
    - 如果模型說 70% 勝率,實際應該勝 ~70%
    - 差的校準意味著模型過度自信或不夠自信
    - 影響最佳機率門檻選擇
    """)
    
    st.markdown("---")
    
    with st.expander("配置", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            model_files = [f for f in os.listdir("trading_system/models") if f.endswith('.pkl')] if os.path.exists("trading_system/models") else []
            
            if len(model_files) == 0:
                st.warning("未找到已訓練的模型。請先訓練模型。")
                return
            
            # 按時間排序,最新在前
            model_files = sorted(model_files, reverse=True)
            model_file = st.selectbox("選擇模型", model_files)
            
            loader = CryptoDataLoader()
            symbol = st.selectbox("交易對", loader.get_available_symbols(), index=10)
            timeframe = st.selectbox("時間框架", loader.get_available_timeframes(), index=1)
        
        with col2:
            tp_multiplier = st.number_input("TP 倍數", value=2.5, step=0.5)
            sl_multiplier = st.number_input("SL 倍數", value=1.5, step=0.25)
            n_bins = st.number_input("分組數", value=10, min_value=5, max_value=20)
            use_recent_data = st.checkbox("只使用2024+數據", value=True,
                                         help="Out-of-Sample 測試")
    
    with st.expander("門檻測試配置", expanded=False):
        st.markdown("測試不同機率門檻的精確率")
        thresholds = st.multiselect(
            "選擇門檻",
            [0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
            default=[0.50, 0.60, 0.65, 0.70]
        )
    
    if st.button("開始分析", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("載入模型...")
            progress_bar.progress(10)
            
            trainer = ModelTrainer()
            trainer.load_model(model_file)
            
            status_text.text("載入數據...")
            progress_bar.progress(20)
            df = loader.load_klines(symbol, timeframe)
            
            if use_recent_data:
                df = df[df['open_time'] >= '2024-01-01'].copy()
                st.info(f"使用2024+數據: {len(df)} 筆")
            
            status_text.text("建立特徵...")
            progress_bar.progress(30)
            
            feature_engineer = FeatureEngineer()
            df_features = feature_engineer.build_features(df)
            
            # 應用事件過濾 (與訓練時一致)
            event_filter = EventFilter(
                use_strict_mode=True,
                min_volume_ratio=1.5,
                min_vsr=1.0,
                bb_squeeze_threshold=0.5,
                lookback_period=20
            )
            df_filtered = event_filter.filter_events(df_features)
            st.info(f"事件過濾: {len(df_features)} → {len(df_filtered)} ({100*len(df_filtered)/len(df_features):.1f}%)")
            
            status_text.text("準備預測特徵...")
            progress_bar.progress(40)
            
            exclude_cols = [
                'open_time', 'close_time', 'open', 'high', 'low', 'close', 'volume',
                'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote_volume',
                'taker_buy_base_asset_volume',
                'label', 'label_return', 'hit_time', 'exit_type', 'sample_weight', 'mae_ratio',
                'bb_middle', 'bb_upper', 'bb_lower', 'bb_std',
                'volume_ma_20'
            ]
            
            X_pred = pd.DataFrame(index=df_filtered.index)
            
            for feature_name in trainer.feature_names:
                if feature_name in df_filtered.columns and feature_name not in exclude_cols:
                    X_pred[feature_name] = df_filtered[feature_name]
                else:
                    X_pred[feature_name] = 0
            
            X_pred = X_pred.fillna(0)
            X_pred = X_pred.replace([np.inf, -np.inf], 0)
            
            for col in X_pred.select_dtypes(include=['bool']).columns:
                X_pred[col] = X_pred[col].astype(int)
            
            status_text.text("生成預測...")
            progress_bar.progress(50)
            
            probabilities = trainer.predict_proba(X_pred)
            df_filtered = df_filtered.copy()
            df_filtered['win_probability'] = probabilities
            
            status_text.text("模擬交易獲取實際結果...")
            progress_bar.progress(60)
            
            # 使用三重屏障標記獲取實際結果
            labeler = TripleBarrierLabeling(
                tp_multiplier=tp_multiplier,
                sl_multiplier=sl_multiplier,
                max_holding_bars=24
            )
            df_labeled = labeler.apply_triple_barrier(df_filtered)
            
            # 合併預測和實際結果
            df_labeled = df_labeled[df_labeled['win_probability'].notna()].copy()
            
            if len(df_labeled) == 0:
                st.error("無有有效的預測結果")
                return
            
            st.success(f"分析 {len(df_labeled)} 個預測")
            
            progress_bar.progress(80)
            status_text.text("分析校準...")
            
            # 按機率分組
            bins = np.linspace(0, 1, n_bins + 1)
            df_labeled['prob_bin'] = pd.cut(df_labeled['win_probability'], bins=bins, labels=False, include_lowest=True)
            
            calibration_data = []
            for bin_idx in range(n_bins):
                bin_data = df_labeled[df_labeled['prob_bin'] == bin_idx]
                if len(bin_data) > 0:
                    predicted_prob = bin_data['win_probability'].mean()
                    actual_prob = bin_data['label'].mean()
                    count = len(bin_data)
                    bin_center = (bins[bin_idx] + bins[bin_idx + 1]) / 2
                    
                    calibration_data.append({
                        'bin': bin_idx,
                        'bin_center': bin_center,
                        'predicted_prob': predicted_prob,
                        'actual_prob': actual_prob,
                        'count': count
                    })
            
            calibration_df = pd.DataFrame(calibration_data)
            
            progress_bar.progress(100)
            status_text.text("分析完成")
            
            # 繪製校準曲線
            st.markdown("### 校準曲線")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='完美校準',
                line=dict(color='gray', dash='dash', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=calibration_df['predicted_prob'],
                y=calibration_df['actual_prob'],
                mode='lines+markers',
                name='模型校準',
                line=dict(color='blue', width=3),
                marker=dict(size=10, color=calibration_df['count'], colorscale='Viridis', showscale=True,
                           colorbar=dict(title="樣本數"))
            ))
            
            fig.update_layout(
                title="校準曲線 (預測 vs 實際勝率)",
                xaxis_title="預測機率",
                yaxis_title="實際勝率",
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 校準指標
            st.markdown("### 校準指標")
            
            mse = ((calibration_df['predicted_prob'] - calibration_df['actual_prob']) ** 2).mean()
            mae = (calibration_df['predicted_prob'] - calibration_df['actual_prob']).abs().mean()
            bias = (calibration_df['predicted_prob'] - calibration_df['actual_prob']).mean()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("校準誤差 (MSE)", f"{mse:.4f}")
            with col2:
                st.metric("平均絕對誤差", f"{mae:.4f}")
            with col3:
                bias_label = "過度自信" if bias > 0 else "不夠自信"
                st.metric("偏差", f"{bias:.4f}", delta=bias_label)
            
            # 門檻測試
            if len(thresholds) > 0:
                st.markdown("### 門檻測試結果")
                
                threshold_results = []
                for threshold in sorted(thresholds):
                    signals = df_labeled[df_labeled['win_probability'] >= threshold]
                    if len(signals) > 0:
                        precision = signals['label'].mean()
                        recall = len(signals) / len(df_labeled)
                        ev = (precision * tp_multiplier) - ((1 - precision) * sl_multiplier)
                        
                        threshold_results.append({
                            '門檻': threshold,
                            '信號數': len(signals),
                            '精確率': f"{precision:.1%}",
                            '召回率': f"{recall:.1%}",
                            '期望值': f"{ev:.3f}R"
                        })
                
                threshold_df = pd.DataFrame(threshold_results)
                st.dataframe(threshold_df, use_container_width=True)
                
                # 找到最佳門檻
                best_threshold = None
                best_ev = -999
                for result in threshold_results:
                    ev = float(result['期望值'].replace('R', ''))
                    if ev > best_ev:
                        best_ev = ev
                        best_threshold = result['門檻']
                
                if best_threshold:
                    st.success(f"最佳門檻: {best_threshold} (期望值: {best_ev:.3f}R)")
            
            st.markdown("### 詳細校準數據")
            st.dataframe(calibration_df, use_container_width=True)
            
            # 解釋
            st.markdown("### 解釋")
            
            if abs(bias) < 0.05:
                st.success("模型校準良好。預測機率可靠。")
            elif bias > 0.05:
                st.warning(f"""模型 **過度自信** {bias:.2%}
                
**建議**:
- 降低門檻 (例如使用 0.55 而非 0.65)
- 或使用 use_calibration=True 重新訓練
- 模型自信度高於實際
                """)
            else:
                st.info(f"""模型 **不夠自信** {abs(bias):.2%}
                
**建議**:
- 提高門檻 (例如使用 0.75 而非 0.65)
- 模型比實際保守
- 這更安全但可能錯過機會
                """)
            
        except Exception as e:
            st.error(f"分析失敗: {str(e)}")
            import traceback
            st.code(traceback.format_exc())