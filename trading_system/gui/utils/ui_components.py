"""Reusable UI components for Streamlit interface"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional

class UIComponents:
    """Reusable UI components"""
    
    @staticmethod
    def render_header(title: str, subtitle: str = ""):
        """Render page header"""
        st.title(title)
        if subtitle:
            st.markdown(f"*{subtitle}*")
        st.markdown("---")
    
    @staticmethod
    def render_info_box(title: str, content: str, box_type: str = "info"):
        """Render info/warning/error box"""
        if box_type == "success":
            st.success(f"**{title}**\n\n{content}")
        elif box_type == "warning":
            st.warning(f"**{title}**\n\n{content}")
        elif box_type == "error":
            st.error(f"**{title}**\n\n{content}")
        else:
            st.info(f"**{title}**\n\n{content}")
    
    @staticmethod
    def render_metrics_row(metrics: Dict[str, Any]):
        """Render metrics in columns"""
        cols = st.columns(len(metrics))
        for idx, (label, value) in enumerate(metrics.items()):
            with cols[idx]:
                if isinstance(value, dict):
                    st.metric(
                        label=label,
                        value=value.get('value', ''),
                        delta=value.get('delta', None)
                    )
                else:
                    st.metric(label=label, value=value)
    
    @staticmethod
    def render_parameter_section(title: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Render parameter input section"""
        with st.expander(title, expanded=True):
            results = {}
            for param_name, param_config in params.items():
                widget_type = param_config.get('type', 'text')
                label = param_config.get('label', param_name)
                
                if widget_type == 'number':
                    results[param_name] = st.number_input(
                        label,
                        value=param_config.get('value', 0),
                        min_value=param_config.get('min', None),
                        max_value=param_config.get('max', None),
                        step=param_config.get('step', 1),
                        help=param_config.get('help', '')
                    )
                elif widget_type == 'slider':
                    results[param_name] = st.slider(
                        label,
                        min_value=param_config.get('min', 0),
                        max_value=param_config.get('max', 100),
                        value=param_config.get('value', 50),
                        step=param_config.get('step', 1),
                        help=param_config.get('help', '')
                    )
                elif widget_type == 'select':
                    results[param_name] = st.selectbox(
                        label,
                        options=param_config.get('options', []),
                        index=param_config.get('index', 0),
                        help=param_config.get('help', '')
                    )
                elif widget_type == 'multiselect':
                    results[param_name] = st.multiselect(
                        label,
                        options=param_config.get('options', []),
                        default=param_config.get('default', []),
                        help=param_config.get('help', '')
                    )
                elif widget_type == 'checkbox':
                    results[param_name] = st.checkbox(
                        label,
                        value=param_config.get('value', False),
                        help=param_config.get('help', '')
                    )
                elif widget_type == 'text':
                    results[param_name] = st.text_input(
                        label,
                        value=param_config.get('value', ''),
                        help=param_config.get('help', '')
                    )
            
            return results
    
    @staticmethod
    def render_progress(steps: List[str], current_step: int):
        """Render progress indicator"""
        progress = (current_step + 1) / len(steps)
        st.progress(progress)
        st.text(f"Step {current_step + 1}/{len(steps)}: {steps[current_step]}")
    
    @staticmethod
    def render_dataframe_preview(df: pd.DataFrame, title: str = "Data Preview", max_rows: int = 10):
        """Render dataframe preview"""
        st.markdown(f"### {title}")
        st.markdown(f"**Shape**: {df.shape[0]} rows x {df.shape[1]} columns")
        st.dataframe(df.head(max_rows), use_container_width=True)
    
    @staticmethod
    def render_feature_importance(feature_names: List[str], importance_values: List[float], top_n: int = 20):
        """Render feature importance chart"""
        df_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=False).head(top_n)
        
        fig = go.Figure(go.Bar(
            x=df_importance['importance'],
            y=df_importance['feature'],
            orientation='h'
        ))
        
        fig.update_layout(
            title=f"Top {top_n} Feature Importance",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_equity_curve(trades_df: pd.DataFrame, symbol: str):
        """Render equity curve"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=trades_df['time'],
            y=trades_df['balance'],
            mode='lines',
            name='Equity',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.update_layout(
            title=f"{symbol} Equity Curve",
            xaxis_title="Time",
            yaxis_title="Balance (USDT)",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_performance_summary(metrics: Dict[str, Any]):
        """Render performance summary"""
        st.markdown("### Performance Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            win_rate = metrics.get('win_rate', 0)
            color = "normal" if 0.5 <= win_rate <= 0.7 else "inverse"
            st.metric(
                "Win Rate",
                f"{win_rate*100:.1f}%",
                f"{metrics.get('winning_trades', 0)}/{metrics.get('total_trades', 0)}"
            )
        
        with col2:
            profit_factor = metrics.get('profit_factor', 0)
            st.metric(
                "Profit Factor",
                f"{profit_factor:.2f}",
                "Good" if profit_factor > 1.5 else "Poor"
            )
        
        with col3:
            total_pnl = metrics.get('total_pnl', 0)
            roi = metrics.get('roi', 0)
            st.metric(
                "Total PnL",
                f"${total_pnl:,.0f}",
                f"{roi:.1f}%"
            )
        
        with col4:
            max_dd = metrics.get('max_drawdown', 0)
            st.metric(
                "Max Drawdown",
                f"{max_dd*100:.1f}%",
                "High Risk" if max_dd > 0.3 else "Acceptable"
            )
    
    @staticmethod
    def render_trade_list(trades_df: pd.DataFrame):
        """Render trade list"""
        st.markdown("### Trade History")
        
        display_df = trades_df[['time', 'side', 'entry', 'exit', 'pnl', 'outcome', 'prob']].copy()
        display_df['pnl'] = display_df['pnl'].apply(lambda x: f"${x:,.2f}")
        display_df['prob'] = display_df['prob'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(display_df, use_container_width=True, height=400)
    
    @staticmethod
    def render_confusion_matrix(y_true: List[int], y_pred: List[int]):
        """Render confusion matrix"""
        from sklearn.metrics import confusion_matrix
        import plotly.figure_factory as ff
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig = ff.create_annotated_heatmap(
            z=cm,
            x=['Predicted 0', 'Predicted 1'],
            y=['Actual 0', 'Actual 1'],
            colorscale='Blues'
        )
        
        fig.update_layout(
            title="Confusion Matrix",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_sidebar_status(status_info: Dict[str, Any]):
        """Render system status in sidebar"""
        st.sidebar.markdown("---")
        st.sidebar.markdown("### System Status")
        
        for key, value in status_info.items():
            if isinstance(value, bool):
                status = "Active" if value else "Inactive"
                st.sidebar.text(f"{key}: {status}")
            else:
                st.sidebar.text(f"{key}: {value}")