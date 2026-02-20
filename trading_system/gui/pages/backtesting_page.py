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
    KellyCriterion, RiskManager, Backtester, RealtimePredictor
)

def render():
    st.title("回測分析")
    
    st.markdown("""
    在歷史數據上測試你的模型績效:
    - 模擬真實交易手續費和滑點
    - Kelly Criterion 動態倉位管理
    - 完整的績效指標和資金曲線
    """)
    
    st.markdown("---")
    
    with st.expander("回測配置", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            model_files = [f for f in os.listdir("trading_system/models") if f.endswith('.pkl')] if os.path.exists("trading_system/models") else []
            
            if len(model_files) == 0:
                st.warning("未找到已訓練的模型。請先訓練模型。")
                return
            
            model_file = st.selectbox("選擇模型", model_files)
            
            loader = CryptoDataLoader()
            symbol = st.selectbox("測試交易對", loader.get_available_symbols(), index=10)
            timeframe = st.selectbox("時間框架", loader.get_available_timeframes(), index=1)
            
            backtest_days = st.number_input("回測天數 (0 = 全部數據)", value=90, min_value=0, step=30)
        
        with col2:
            initial_capital = st.number_input("初始資金", value=10000.0, step=1000.0)
            commission_rate = st.number_input("手續費率", value=0.001, step=0.0001, format="%.4f")
            slippage = st.number_input("滑點", value=0.0005, step=0.0001, format="%.4f")
        
        with col3:
            tp_multiplier = st.number_input("止盈倍數", value=4.0, step=0.1)
            sl_multiplier = st.number_input("止損倍數", value=2.0, step=0.1)
            kelly_fraction = st.number_input("Kelly 分數", value=0.5, step=0.1)
    
    if st.button("運行回測", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("載入模型中...")
            progress_bar.progress(10)
            trainer = ModelTrainer()
            trainer.load_model(model_file)
            
            status_text.text("載入數據中...")
            progress_bar.progress(20)
            df = loader.load_klines(symbol, timeframe)
            
            status_text.text("建立特徵中...")
            progress_bar.progress(35)
            feature_engineer = FeatureEngineer()
            df_features = feature_engineer.build_features(df)
            
            status_text.text("生成預測中...")
            progress_bar.progress(50)
            kelly = KellyCriterion(tp_multiplier, sl_multiplier, kelly_fraction)
            predictor = RealtimePredictor(trainer, feature_engineer, kelly)
            predictions = predictor.predict_from_completed_bars(df_features)
            
            signals = predictions[predictions['signal'] == 1].copy()
            st.info(f"生成 {len(signals)} 個交易信號,時間範圍: {predictions['open_time'].min()} 至 {predictions['open_time'].max()}")
            
            if len(signals) == 0:
                st.warning("未生成任何信號。請調整參數。")
                return
            
            status_text.text("執行回測中...")
            progress_bar.progress(70)
            backtester = Backtester(initial_capital, commission_rate, slippage)
            
            backtest_days_param = int(backtest_days) if backtest_days > 0 else None
            
            results = backtester.run_backtest(
                signals, 
                tp_multiplier=tp_multiplier, 
                sl_multiplier=sl_multiplier,
                backtest_days=backtest_days_param
            )
            
            progress_bar.progress(100)
            status_text.text("回測完成")
            
            st.success("回測執行成功")
            
            stats = results['statistics']
            trades_df = results['trades']
            
            st.markdown("### 績效摘要")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("初始資金", f"${initial_capital:,.2f}")
            with col2:
                st.metric("最終資金", f"${stats['final_capital']:,.2f}")
            with col3:
                st.metric("淨損益", f"${stats['net_pnl']:,.2f}")
            with col4:
                st.metric("總回報", f"{stats['total_return']*100:.2f}%")
            with col5:
                st.metric("總手續費", f"${stats['total_commission']:,.2f}")
            
            st.markdown("### 績效指標")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("總交易次數", stats['total_trades'])
                st.metric("勝率", f"{stats['win_rate']*100:.2f}%")
            with col2:
                st.metric("獲利次數", stats['winning_trades'])
                st.metric("虧損次數", stats['losing_trades'])
            with col3:
                st.metric("平均獲利", f"${stats['avg_win']:.2f}")
                st.metric("平均虧損", f"${stats['avg_loss']:.2f}")
            with col4:
                st.metric("盈虧比", f"{stats['profit_factor']:.2f}")
                st.metric("Sharpe 比率", f"{stats['sharpe_ratio']:.2f}")
            
            st.markdown("### 風險指標")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("最大回撤", f"{stats['max_drawdown']*100:.2f}%")
            with col2:
                st.metric("平均持倉時間", f"{stats['avg_trade_duration']:.1f} 根")
            with col3:
                st.metric("總獲利金額", f"${stats['total_win']:,.2f}")
            with col4:
                st.metric("總虧損金額", f"${stats['total_loss']:,.2f}")
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("資金曲線", "回撤曲線"),
                vertical_spacing=0.15,
                row_heights=[0.7, 0.3]
            )
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(trades_df))),
                    y=trades_df['capital'],
                    mode='lines',
                    name='資金',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            fig.add_hline(
                y=initial_capital,
                line_dash="dash",
                line_color="gray",
                annotation_text="初始資金",
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(trades_df))),
                    y=trades_df['drawdown_pct'] * 100,
                    mode='lines',
                    name='回撤',
                    fill='tozeroy',
                    line=dict(color='red', width=1)
                ),
                row=2, col=1
            )
            
            fig.update_xaxes(title_text="交易次數", row=2, col=1)
            fig.update_yaxes(title_text="資金 ($)", row=1, col=1)
            fig.update_yaxes(title_text="回撤 (%)", row=2, col=1)
            
            fig.update_layout(height=700, showlegend=True)
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 退出原因分布")
            exit_counts = trades_df['exit_reason'].value_counts()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("止盈出場", exit_counts.get('TP', 0))
            with col2:
                st.metric("止損出場", exit_counts.get('SL', 0))
            with col3:
                st.metric("超時出場", exit_counts.get('Timeout', 0))
            
            st.markdown("### 近期交易紀錄")
            display_cols = ['entry_time', 'entry_price', 'exit_price', 'exit_reason', 'exit_bars', 
                          'pnl_dollar', 'total_commission', 'capital']
            display_df = trades_df[display_cols].tail(50).copy()
            display_df['entry_time'] = display_df['entry_time'].dt.strftime('%Y-%m-%d %H:%M')
            display_df.columns = ['進場時間', '進場價', '出場價', '退出原因', '持倉時間', '損益', '手續費', '累計資金']
            st.dataframe(display_df, use_container_width=True)
            
            csv = trades_df.to_csv(index=False)
            st.download_button(
                label="下載完整交易紀錄 (CSV)",
                data=csv,
                file_name=f"backtest_{symbol}_{timeframe}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"回測失敗: {str(e)}")
            import traceback
            st.code(traceback.format_exc())