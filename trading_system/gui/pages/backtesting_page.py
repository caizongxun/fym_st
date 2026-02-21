import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core import (
    CryptoDataLoader, FeatureEngineer, ModelTrainer,
    TripleBarrierLabeling, EventFilter, Backtester
)

def render():
    st.title("回測分析 (MTF 支援)")
    
    st.markdown("""
    在歷史數據上測試你的模型績效:
    - **MTF 支援**: 自動偵測並載入 15m + 1h 數據
    - 正確的 Maker/Taker 費率模型
    - 槓桿合約交易模擬
    - TP 無滑點,SL 有滑點 (真實情況)
    - ATR 基礎風險管理
    """)
    
    with st.expander("優化建議", expanded=False):
        st.markdown("""
        ### 提升收益率的三大方向
        
        **1. 拉高 TP 倍數 (3.5-4.0)**
        - 目的: 讓獲利遠大於手續費 (0.12%)
        - 代價: 勝率可能從 66% 降至 55%
        - 結果: 平均獲利大幅提升,盈虧比改善
        
        **2. 降低機率門檻 (0.52-0.53)**
        - 目的: 增加交易頻率 (90天 12筆 → 25-30筆)
        - 條件: 新信號依然保持正期望值
        - 結果: 總獲利翻倍
        
        **3. 實盤使用 Maker 費率**
        - 方法: TP 使用限價單 (Limit Order)
        - 節省: 60% 的出場手續費
        - 影響: 平均獲利立即提升
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
            
            is_mtf_model = 'MTF' in model_file or '_15m_1h' in model_file
            if is_mtf_model:
                st.success("✅ 偵測到 MTF 模型，將載入 15m + 1h 數據")
            
            loader = CryptoDataLoader()
            symbol = st.selectbox("測試交易對", loader.get_available_symbols(), index=10)
            
            if is_mtf_model:
                timeframe = '15m'
                st.info("🔒 MTF 模型鎖定為 15m 進場時間框架")
            else:
                timeframe = st.selectbox("時間框架", loader.get_available_timeframes(), index=1)
            
            data_source = st.radio(
                "數據來源",
                ["Binance API (最新)", "HuggingFace (快速)"],
                help="Binance API 獲取最新數據"
            )
            
            if data_source == "Binance API (最新)":
                backtest_days = st.number_input("回測天數", value=90, min_value=7, max_value=365, step=7)
            else:
                use_recent_data = st.checkbox("只使用2024+數據 (OOS)", value=True)
        
        with col2:
            initial_capital = st.number_input("初始資金", value=10000.0, step=1000.0)
            risk_per_trade = st.number_input("每筆風險%", value=2.0, step=0.5)
            leverage = st.number_input("槓桿倍數", value=10, min_value=1, max_value=20, step=1)
            
        with col3:
            tp_multiplier = st.number_input(
                "TP 倍數 (ATR)", 
                value=3.0, 
                step=0.5,
                help="建議 3.0-3.5 以覆蓋手續費"
            )
            sl_multiplier = st.number_input("SL 倍數 (ATR)", value=1.0, step=0.25)
            probability_threshold = st.number_input(
                "機率門檻", 
                value=0.55, 
                step=0.01,
                help="建議 0.55 以維持高勝率"
            )
    
    with st.expander("手續費與滑點", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            taker_fee = st.number_input("Taker 費率", value=0.0006, step=0.0001, format="%.4f")
            maker_fee = st.number_input("Maker 費率", value=0.0002, step=0.0001, format="%.4f")
        with col2:
            slippage = st.number_input("滑點", value=0.0005, step=0.0001, format="%.4f")
            st.info("TP 使用限價單可省 60% 費用")
    
    with st.expander("事件過濾設定", expanded=False):
        use_event_filter = st.checkbox("啟用事件過濾", value=True)
        if use_event_filter:
            col1, col2 = st.columns(2)
            with col1:
                min_volume_ratio = st.number_input("最小成交量比率", value=2.0 if is_mtf_model else 1.5, step=0.1)
                use_strict = st.checkbox("嚴格模式", value=True)
            with col2:
                min_vsr = st.number_input("最小波動率", value=1.0, step=0.1)
                bb_squeeze = st.number_input("BB壓縮門檻", value=0.5, step=0.1)
                lookback_period = st.number_input("突破回看週期", value=40 if is_mtf_model else 20, step=10)
    
    if st.button("運行回測", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("載入模型...")
            progress_bar.progress(10)
            trainer = ModelTrainer()
            trainer.load_model(model_file)
            
            st.info(f"模型特徵: {len(trainer.feature_names)} 個")
            
            status_text.text("載入數據...")
            progress_bar.progress(20)
            
            if is_mtf_model:
                st.info("🔄 MTF 模式: 載入 15m + 1h 數據...")
                
                if data_source == "Binance API (最新)":
                    df_15m = loader.fetch_latest_klines(symbol, '15m', days=int(backtest_days))
                    df_1h = loader.fetch_latest_klines(symbol, '1h', days=int(backtest_days))
                else:
                    df_15m = loader.load_klines(symbol, '15m')
                    df_1h = loader.load_klines(symbol, '1h')
                    if use_recent_data:
                        df_15m = df_15m[df_15m['open_time'] >= '2024-01-01'].copy()
                        df_1h = df_1h[df_1h['open_time'] >= '2024-01-01'].copy()
                
                st.info(f"載入完成: 15m ({len(df_15m)} 筆), 1h ({len(df_1h)} 筆)")
                st.info(f"數據範圍: {df_15m['open_time'].min()} ~ {df_15m['open_time'].max()}")
                
                status_text.text("建立 MTF 特徵...")
                progress_bar.progress(30)
                feature_engineer = FeatureEngineer()
                
                df_15m_features = feature_engineer.build_features(df_15m, include_microstructure=True)
                df_1h_features = feature_engineer.build_features(df_1h, include_microstructure=True)
                
                df_features = feature_engineer.merge_and_build_mtf_features(df_15m_features, df_1h_features)
                st.success(f"MTF 特徵合併完成! 形狀: {df_features.shape}")
                
            else:
                if data_source == "Binance API (最新)":
                    df = loader.fetch_latest_klines(symbol, timeframe, days=int(backtest_days))
                else:
                    df = loader.load_klines(symbol, timeframe)
                    if use_recent_data:
                        df = df[df['open_time'] >= '2024-01-01'].copy()
                
                st.info(f"載入 {len(df)} 筆,範圍: {df['open_time'].min()} ~ {df['open_time'].max()}")
                
                status_text.text("建立特徵...")
                progress_bar.progress(30)
                feature_engineer = FeatureEngineer()
                df_features = feature_engineer.build_features(df)
            
            if use_event_filter:
                status_text.text("事件過濾...")
                progress_bar.progress(35)
                event_filter = EventFilter(
                    use_strict_mode=use_strict,
                    min_volume_ratio=min_volume_ratio,
                    min_vsr=min_vsr,
                    bb_squeeze_threshold=bb_squeeze,
                    lookback_period=int(lookback_period)
                )
                df_filtered = event_filter.filter_events(df_features)
                st.info(f"過濾: {len(df_features)} → {len(df_filtered)} ({100*len(df_filtered)/len(df_features):.1f}%)")
            else:
                df_filtered = df_features
            
            status_text.text("生成預測...")
            progress_bar.progress(45)
            
            exclude_cols = [
                'open_time', 'close_time', 'htf_close_time',
                'label', 'label_return', 'hit_time', 'exit_type', 'exit_price', 'exit_bars', 'return',
                'sample_weight', 'mae_ratio', 'ignore'
            ]
            
            X_pred = pd.DataFrame(index=df_filtered.index)
            missing_features = []
            
            for feature_name in trainer.feature_names:
                if feature_name in df_filtered.columns and feature_name not in exclude_cols:
                    X_pred[feature_name] = df_filtered[feature_name]
                else:
                    if feature_name not in df_filtered.columns:
                        missing_features.append(feature_name)
                    X_pred[feature_name] = 0
            
            if len(missing_features) > 0:
                st.error(f"⚠️ 缺失特徵 ({len(missing_features)}): {', '.join(missing_features[:10])}...")
                st.info("建議: 重新訓練模型以移除非平稩特徵")
            
            X_pred = X_pred.fillna(0).replace([np.inf, -np.inf], 0)
            
            for col in X_pred.select_dtypes(include=['bool']).columns:
                X_pred[col] = X_pred[col].astype(int)
            
            probabilities = trainer.predict_proba(X_pred)
            df_filtered = df_filtered.copy()
            df_filtered['win_probability'] = probabilities
            
            prob_dist = df_filtered['win_probability'].describe()
            st.info(f"機率分布: min={prob_dist['min']:.3f}, mean={prob_dist['mean']:.3f}, max={prob_dist['max']:.3f}")
            
            signals = df_filtered[df_filtered['win_probability'] >= probability_threshold].copy()
            st.info(f"信號: {len(signals)} 個 (門檻: {probability_threshold})")
            
            if len(signals) == 0:
                st.warning("無信號,請降低門檻或增加回測天數")
                st.info(f"提示: 最高機率為 {prob_dist['max']:.3f}, 建議門檻 < {prob_dist['75%']:.3f}")
                return
            
            status_text.text("執行回測...")
            progress_bar.progress(60)
            
            backtester = Backtester(
                initial_capital=initial_capital,
                taker_fee=taker_fee,
                maker_fee=maker_fee,
                slippage=slippage,
                risk_per_trade=risk_per_trade / 100.0,
                leverage=int(leverage)
            )
            results = backtester.run_backtest(
                signals,
                tp_multiplier=tp_multiplier,
                sl_multiplier=sl_multiplier,
                direction=1
            )
            
            progress_bar.progress(100)
            status_text.text("完成")
            
            stats = results['statistics']
            trades_df = results['trades']
            
            if len(trades_df) == 0:
                st.warning("回測未產生交易,請調整參數")
                return
            
            st.success("回測完成")
            
            days_in_test = (trades_df.iloc[-1]['entry_time'] - trades_df.iloc[0]['entry_time']).days
            days_in_test = max(days_in_test, 1)
            annualized_return = stats['total_return'] * (365 / days_in_test)
            
            fee_to_profit_ratio = stats['total_commission'] / stats['net_pnl'] if stats['net_pnl'] > 0 else 0
            
            st.markdown("### 績效摘要")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("初始資金", f"${initial_capital:,.0f}")
            with col2:
                st.metric("最終資金", f"${stats['final_capital']:,.0f}")
            with col3:
                st.metric("淪損益", f"${stats['net_pnl']:,.0f}", 
                         delta=f"{stats['total_return']*100:.1f}%")
            with col4:
                st.metric("總手續費", f"${stats['total_commission']:,.0f}",
                         delta=f"{fee_to_profit_ratio*100:.1f}% 佔利潤",
                         delta_color="inverse")
            with col5:
                st.metric("年化報酬", f"{annualized_return*100:.1f}%")
            
            col1, col2 = st.columns(2)
            with col1:
                ev_theory = (stats['win_rate'] * tp_multiplier) - ((1 - stats['win_rate']) * sl_multiplier)
                st.metric("理論期望值", f"{ev_theory:.3f}R")
            with col2:
                avg_win_r = stats['avg_win'] / (initial_capital * risk_per_trade / 100) if stats['avg_win'] > 0 else 0
                avg_loss_r = abs(stats['avg_loss']) / (initial_capital * risk_per_trade / 100) if stats['avg_loss'] < 0 else 0
                ev_actual = (stats['win_rate'] * avg_win_r) - ((1 - stats['win_rate']) * avg_loss_r)
                st.metric("實際期望值", f"{ev_actual:.3f}R")
            
            st.markdown("### 績效指標")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("交易次數", stats['total_trades'])
                trades_per_week = stats['total_trades'] / (days_in_test / 7)
                st.metric("週均交易", f"{trades_per_week:.1f} 筆")
            with col2:
                st.metric("勝率", f"{stats['win_rate']*100:.1f}%")
                st.metric("獲利/虧損", f"{stats['winning_trades']}/{stats['losing_trades']}")
            with col3:
                st.metric("平均獲利", f"${stats['avg_win']:.0f}")
                st.metric("平均虧損", f"${stats['avg_loss']:.0f}")
            with col4:
                st.metric("盈虧比", f"{stats['profit_factor']:.2f}")
                st.metric("Sharpe", f"{stats['sharpe_ratio']:.2f}")
            
            st.markdown("### 風險指標")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("最大回撤", f"{stats['max_drawdown']*100:.1f}%")
            with col2:
                st.metric("平均持倉", f"{stats['avg_trade_duration']:.1f} 根")
            with col3:
                st.metric("總獲利", f"${stats['total_win']:,.0f}")
            with col4:
                st.metric("總虧損", f"${stats['total_loss']:,.0f}")
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("資金曲線", "回撤%"),
                vertical_spacing=0.15,
                row_heights=[0.7, 0.3]
            )
            
            fig.add_trace(
                go.Scatter(x=list(range(len(trades_df))), y=trades_df['capital'],
                          mode='lines', name='資金', line=dict(color='blue', width=2)),
                row=1, col=1
            )
            
            fig.add_hline(y=initial_capital, line_dash="dash", line_color="gray",
                         annotation_text="初始", row=1, col=1)
            
            fig.add_trace(
                go.Scatter(x=list(range(len(trades_df))), y=trades_df['drawdown_pct']*100,
                          mode='lines', name='回撤', fill='tozeroy', line=dict(color='red', width=1)),
                row=2, col=1
            )
            
            fig.update_xaxes(title_text="交易次數", row=2, col=1)
            fig.update_yaxes(title_text="$", row=1, col=1)
            fig.update_yaxes(title_text="%", row=2, col=1)
            fig.update_layout(height=700, showlegend=True)
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 退出原因")
            exit_counts = trades_df['exit_reason'].value_counts()
            col1, col2, col3 = st.columns(3)
            with col1:
                tp = exit_counts.get('TP', 0)
                st.metric("TP", tp, delta=f"{100*tp/len(trades_df):.1f}%")
            with col2:
                sl = exit_counts.get('SL', 0)
                st.metric("SL", sl, delta=f"{100*sl/len(trades_df):.1f}%")
            with col3:
                timeout = exit_counts.get('Timeout', 0)
                st.metric("Timeout", timeout, delta=f"{100*timeout/len(trades_df):.1f}%")
            
            st.markdown("### 交易紀錄")
            display_df = trades_df[['entry_time', 'entry_price', 'exit_price', 'required_margin',
                                   'exit_reason', 'pnl_dollar', 'total_commission', 'capital']].tail(50).copy()
            display_df['entry_time'] = display_df['entry_time'].dt.strftime('%Y-%m-%d %H:%M')
            display_df.columns = ['時間', '進場', '出場', '保證金', '原因', '損益', '費用', '累計']
            st.dataframe(display_df, use_container_width=True)
            
            csv = trades_df.to_csv(index=False)
            st.download_button(
                label="下載 CSV",
                data=csv,
                file_name=f"backtest_{symbol}_{timeframe}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # ===== [新增] 詳細回測報告 (可複製給 Gemini) =====
            st.markdown("---")
            st.markdown("### 📋 詳細回測報告 (可複製給 Gemini 查看)")
            
            report = f"""
# MTF 多時間框架交易系統回測報告

## 回測配置
- **模型**: {model_file}
- **交易對**: {symbol}
- **時間框架**: {timeframe} {'(MTF: 15m + 1h)' if is_mtf_model else ''}
- **回測時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **數據範圍**: {df_features['open_time'].min()} ~ {df_features['open_time'].max()}
- **回測天數**: {days_in_test} 天

## 風控參數
- **初始資金**: ${initial_capital:,.0f}
- **每筆風險**: {risk_per_trade}%
- **槓桿倍數**: {leverage}x
- **TP/SL 倍數**: {tp_multiplier:.1f} / {sl_multiplier:.1f} ATR
- **機率門檻**: {probability_threshold}

## 手續費與滑點
- **Taker 費率**: {taker_fee:.4f} ({taker_fee*100:.2f}%)
- **Maker 費率**: {maker_fee:.4f} ({maker_fee*100:.2f}%)
- **滑點**: {slippage:.4f} ({slippage*100:.2f}%)
- **總手續費**: ${stats['total_commission']:,.0f}
- **手續費佔利潤比**: {fee_to_profit_ratio*100:.1f}%

## 績效摘要
- **最終資金**: ${stats['final_capital']:,.0f}
- **淪損益**: ${stats['net_pnl']:,.0f} ({stats['total_return']*100:.1f}%)
- **年化報酬**: {annualized_return*100:.1f}%
- **理論期望值**: {ev_theory:.3f}R
- **實際期望值**: {ev_actual:.3f}R

## 交易統計
- **總交易次數**: {stats['total_trades']}
- **週均交易**: {trades_per_week:.1f} 筆
- **勝率**: {stats['win_rate']*100:.1f}%
- **獲利交易**: {stats['winning_trades']}
- **虧損交易**: {stats['losing_trades']}

## 損益分析
- **平均獲利**: ${stats['avg_win']:.0f}
- **平均虧損**: ${stats['avg_loss']:.0f}
- **總獲利**: ${stats['total_win']:,.0f}
- **總虧損**: ${stats['total_loss']:,.0f}
- **盈虧比**: {stats['profit_factor']:.2f}

## 風險指標
- **最大回撤**: {stats['max_drawdown']*100:.1f}%
- **Sharpe Ratio**: {stats['sharpe_ratio']:.2f}
- **平均持倉**: {stats['avg_trade_duration']:.1f} 根 ({stats['avg_trade_duration']/4:.1f} 小時)

## 退出原因分布
- **TP (止盈)**: {exit_counts.get('TP', 0)} ({100*exit_counts.get('TP', 0)/len(trades_df):.1f}%)
- **SL (止損)**: {exit_counts.get('SL', 0)} ({100*exit_counts.get('SL', 0)/len(trades_df):.1f}%)
- **Timeout (超時)**: {exit_counts.get('Timeout', 0)} ({100*exit_counts.get('Timeout', 0)/len(trades_df):.1f}%)

## 機率分布
- **最小機率**: {prob_dist['min']:.3f}
- **平均機率**: {prob_dist['mean']:.3f}
- **最大機率**: {prob_dist['max']:.3f}
- **75% 分位數**: {prob_dist['75%']:.3f}
- **信號數量**: {len(signals)} (門檻 {probability_threshold})

## 事件過濾配置
- **啟用**: {'Yes' if use_event_filter else 'No'}
- **最小成交量比率**: {min_volume_ratio if use_event_filter else 'N/A'}
- **最小波動率**: {min_vsr if use_event_filter else 'N/A'}
- **嚴格模式**: {'Yes' if use_strict and use_event_filter else 'No'}
- **突破回看週期**: {lookback_period if use_event_filter else 'N/A'}
- **過濾後比例**: {100*len(df_filtered)/len(df_features):.1f}%

## 缺失特徵
{', '.join(missing_features) if len(missing_features) > 0 else 'None'}
"""
            
            st.text_area("報告內容 (點擊右上角複製)", report, height=400)
            
            st.markdown("### 優化建議")
            
            suggestions = []
            
            if len(missing_features) > 0:
                suggestions.append(f"偵測到 {len(missing_features)} 個缺失特徵,強烈建議重新訓練模型")
            
            if fee_to_profit_ratio > 0.3:
                suggestions.append(f"手續費佔利潤 {fee_to_profit_ratio*100:.1f}% 過高,建議提高 TP 至 {tp_multiplier+0.5:.1f}")
            
            if trades_per_week < 2:
                suggestions.append(f"週均交易 {trades_per_week:.1f} 筆過少,建議降低門檻至 {probability_threshold-0.03:.2f}")
            
            if stats['avg_loss'] and abs(stats['avg_loss']) > stats['avg_win']:
                suggestions.append("平均虧損 > 平均獲利,考慮提高 TP")
            
            if stats['total_return'] > 0.3 and stats['sharpe_ratio'] > 2.0:
                suggestions.append("優秀的績效! 可以考慮實盤測試")
            
            if len(suggestions) > 0:
                for s in suggestions:
                    st.info(s)
            else:
                st.success("參數設定良好!")
            
        except Exception as e:
            st.error(f"錯誤: {str(e)}")
            import traceback
            st.code(traceback.format_exc())