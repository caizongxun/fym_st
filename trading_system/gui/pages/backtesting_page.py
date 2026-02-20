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
    TripleBarrierLabeling, EventFilter, Backtester
)

def render():
    st.title("回測分析")
    
    st.markdown("""
    在歷史數據上測試你的模型績效:
    - 模擬真實交易手續費和滑點
    - ATR 基礎風險管理 (每筆風險固定)
    - 機率門檻控制 (基於校準分析)
    - 嚴格事件過濾 (與訓練一致)
    """)
    
    st.markdown("---")
    
    with st.expander("回測配置", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            model_files = [f for f in os.listdir("trading_system/models") if f.endswith('.pkl')] if os.path.exists("trading_system/models") else []
            
            if len(model_files) == 0:
                st.warning("未找到已訓練的模型。請先訓練模型。")
                return
            
            model_files = sorted(model_files, reverse=True)
            model_file = st.selectbox("選擇模型", model_files)
            
            loader = CryptoDataLoader()
            symbol = st.selectbox("測試交易對", loader.get_available_symbols(), index=10)
            timeframe = st.selectbox("時間框架", loader.get_available_timeframes(), index=1)
            
            data_source = st.radio(
                "數據來源",
                ["Binance API (最新)", "HuggingFace (快速)"],
                help="Binance API 獲取最新數據,HuggingFace 使用緩存"
            )
            
            if data_source == "Binance API (最新)":
                backtest_days = st.number_input("回測天數", value=90, min_value=7, max_value=365, step=7,
                                               help="從 Binance 直接抽取最近 N 天數據")
            else:
                use_recent_data = st.checkbox("只使用2024+數據 (OOS)", value=True)
        
        with col2:
            initial_capital = st.number_input("初始資金", value=10000.0, step=1000.0)
            risk_per_trade = st.number_input("每筆風險%", value=1.0, step=0.5,
                                            help="每筆交易風險的資金比例")
            commission_rate = st.number_input("手續費率", value=0.0006, step=0.0001, format="%.4f",
                                             help="合約吃單 0.06%")
            slippage = st.number_input("滑點", value=0.0005, step=0.0001, format="%.4f")
        
        with col3:
            tp_multiplier = st.number_input("TP 倍數 (ATR)", value=2.5, step=0.5)
            sl_multiplier = st.number_input("SL 倍數 (ATR)", value=1.5, step=0.25)
            probability_threshold = st.number_input("機率門檻", value=0.60, step=0.05,
                                                    help="基於校準分析的最佳門檻")
            max_holding_bars = st.number_input("最大持倉根數", value=24, step=6)
    
    with st.expander("事件過濾設定", expanded=False):
        st.markdown("與訓練時保持一致")
        use_event_filter = st.checkbox("啟用事件過濾", value=True)
        if use_event_filter:
            col1, col2 = st.columns(2)
            with col1:
                min_volume_ratio = st.number_input("最小成交量比率", value=1.5, step=0.1)
                use_strict = st.checkbox("嚴格模式", value=True)
            with col2:
                min_vsr = st.number_input("最小波動率", value=1.0, step=0.1)
                bb_squeeze = st.number_input("BB壓縮門檻", value=0.5, step=0.1)
    
    if st.button("運行回測", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("載入模型...")
            progress_bar.progress(10)
            trainer = ModelTrainer()
            trainer.load_model(model_file)
            
            status_text.text("載入數據...")
            progress_bar.progress(20)
            
            if data_source == "Binance API (最新)":
                st.info(f"從 Binance 抽取最近 {backtest_days} 天數據...")
                df = loader.fetch_latest_klines(symbol, timeframe, days=int(backtest_days))
            else:
                df = loader.load_klines(symbol, timeframe)
                if use_recent_data:
                    df = df[df['open_time'] >= '2024-01-01'].copy()
                    st.info(f"Out-of-Sample 測試: {len(df)} 筆 (2024-01-01 至今)")
            
            st.info(f"載入 {len(df)} 筆數據,時間範圍: {df['open_time'].min()} 至 {df['open_time'].max()}")
            
            status_text.text("建立特徵...")
            progress_bar.progress(30)
            feature_engineer = FeatureEngineer()
            df_features = feature_engineer.build_features(df)
            
            if use_event_filter:
                status_text.text("應用事件過濾...")
                progress_bar.progress(35)
                event_filter = EventFilter(
                    use_strict_mode=use_strict,
                    min_volume_ratio=min_volume_ratio,
                    min_vsr=min_vsr,
                    bb_squeeze_threshold=bb_squeeze,
                    lookback_period=20
                )
                df_filtered = event_filter.filter_events(df_features)
                filter_ratio = len(df_filtered) / len(df_features)
                st.info(f"事件過濾: {len(df_features)} → {len(df_filtered)} ({100*filter_ratio:.1f}%)")
            else:
                df_filtered = df_features
            
            status_text.text("生成預測...")
            progress_bar.progress(45)
            
            exclude_cols = [
                'open_time', 'close_time', 'open', 'high', 'low', 'close', 'volume',
                'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote_volume',
                'taker_buy_base_asset_volume',
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
            
            probabilities = trainer.predict_proba(X_pred)
            df_filtered = df_filtered.copy()
            df_filtered['win_probability'] = probabilities
            
            signals = df_filtered[df_filtered['win_probability'] >= probability_threshold].copy()
            st.info(f"生成 {len(signals)} 個交易信號 (門檻: {probability_threshold})")
            
            if len(signals) == 0:
                st.warning("未生成任何信號。請降低門檻。")
                return
            
            status_text.text("執行回測...")
            progress_bar.progress(60)
            
            backtester = Backtester(
                initial_capital=initial_capital,
                commission_rate=commission_rate,
                slippage=slippage,
                risk_per_trade=risk_per_trade / 100.0
            )
            results = backtester.run_backtest(
                signals,
                tp_multiplier=tp_multiplier,
                sl_multiplier=sl_multiplier
            )
            
            progress_bar.progress(100)
            status_text.text("回測完成")
            
            st.success("回測執行成功")
            
            stats = results['statistics']
            trades_df = results['trades']
            
            st.markdown("### 績效摘要")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("初始資金", f"${initial_capital:,.0f}")
            with col2:
                st.metric("最終資金", f"${stats['final_capital']:,.0f}")
            with col3:
                st.metric("淨損益", f"${stats['net_pnl']:,.0f}", 
                         delta=f"{stats['total_return']*100:.1f}%")
            with col4:
                st.metric("總手續費", f"${stats['total_commission']:,.0f}")
            with col5:
                ev_actual = (stats['win_rate'] * tp_multiplier) - ((1 - stats['win_rate']) * sl_multiplier)
                st.metric("實際期望值", f"{ev_actual:.3f}R")
            
            st.markdown("### 績效指標")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("總交易次數", stats['total_trades'])
                st.metric("勝率", f"{stats['win_rate']*100:.1f}%")
            with col2:
                st.metric("獲利次數", stats['winning_trades'])
                st.metric("虧損次數", stats['losing_trades'])
            with col3:
                st.metric("平均獲利", f"${stats['avg_win']:.0f}")
                st.metric("平均虧損", f"${stats['avg_loss']:.0f}")
            with col4:
                st.metric("盈虧比", f"{stats['profit_factor']:.2f}")
                st.metric("Sharpe 比率", f"{stats['sharpe_ratio']:.2f}")
            
            st.markdown("### 風險指標")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("最大回撤", f"{stats['max_drawdown']*100:.1f}%")
            with col2:
                st.metric("平均持倉時間", f"{stats['avg_trade_duration']:.1f} 根")
            with col3:
                st.metric("總獲利金額", f"${stats['total_win']:,.0f}")
            with col4:
                st.metric("總虧損金額", f"${stats['total_loss']:,.0f}")
            
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
                tp_count = exit_counts.get('TP', 0)
                st.metric("止盈出場", tp_count, 
                         delta=f"{100*tp_count/len(trades_df):.1f}%")
            with col2:
                sl_count = exit_counts.get('SL', 0)
                st.metric("止損出場", sl_count,
                         delta=f"{100*sl_count/len(trades_df):.1f}%")
            with col3:
                timeout_count = exit_counts.get('Timeout', 0)
                st.metric("超時出場", timeout_count,
                         delta=f"{100*timeout_count/len(trades_df):.1f}%")
            
            st.markdown("### 近期交易紀錄")
            display_cols = ['entry_time', 'entry_price', 'exit_price', 'position_value', 'exit_reason', 'exit_bars', 
                          'pnl_dollar', 'total_commission', 'capital']
            display_df = trades_df[display_cols].tail(50).copy()
            display_df['entry_time'] = display_df['entry_time'].dt.strftime('%Y-%m-%d %H:%M')
            display_df.columns = ['進場時間', '進場價', '出場價', '倉位金額', '退出原因', '持倉時間', '損益', '手續費', '累計資金']
            st.dataframe(display_df, use_container_width=True)
            
            csv = trades_df.to_csv(index=False)
            st.download_button(
                label="下載完整交易紀錄 (CSV)",
                data=csv,
                file_name=f"backtest_{symbol}_{timeframe}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            st.markdown("### 建議")
            if stats['total_return'] > 0.5 and stats['sharpe_ratio'] > 1.5:
                st.success("優秀的回測結果! 可以考慮實盤測試")
            elif stats['total_return'] > 0.2 and stats['win_rate'] > 0.5:
                st.info("合格的結果,建議繼續優化參數")
            else:
                st.warning("績效不佳,建議重新訓練模型或調整門檻")
            
        except Exception as e:
            st.error(f"回測失敗: {str(e)}")
            import traceback
            st.code(traceback.format_exc())