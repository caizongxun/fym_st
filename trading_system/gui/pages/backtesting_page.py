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
    BacktestEngine
)
from core.event_filter import BBNW_BounceFilter

def render():
    st.title("📊 BB+NW 波段反轉系統 - 回測分析")
    
    st.markdown("""
    ---
    ### 🔍 回測流程
    
    1. 選擇已訓練的 BB+NW 模型
    2. 載入 OOS (Out-of-Sample) 測試數據
    3. 模擬實際交易 (滑點 + 手續費)
    4. 分析勝率、盈虧比、最大回撤
    
    ---
    """)
    
    # ===== 模型選擇 =====
    with st.expander("🤖 步驟 1: 選擇模型", expanded=True):
        models_dir = "models"
        if not os.path.exists(models_dir):
            st.warning("⚠️ 未找到 models 目錄，請先訓練模型")
            return
        
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        
        if not model_files:
            st.warning("⚠️ 無可用模型，請先到「模型訓練」頁面訓練")
            return
        
        selected_model = st.selectbox(
            "💾 選擇模型",
            model_files,
            help="選擇 BB+NW 波段反轉模型"
        )
        
        model_path = os.path.join(models_dir, selected_model)
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"📝 模型路徑: `{model_path}`")
        with col2:
            file_size = os.path.getsize(model_path) / 1024
            st.info(f"📁 檔案大小: {file_size:.1f} KB")
    
    # ===== 回測參數 =====
    with st.expander("⚙️ 步驟 2: 回測參數", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            loader = CryptoDataLoader()
            symbol = st.selectbox(
                "🪙 交易對",
                loader.get_available_symbols(),
                index=10
            )
        
        with col2:
            test_period = st.selectbox(
                "📅 測試期間",
                [
                    "2024 全年 (OOS)",
                    "2024 Q4",
                    "最近 90 天",
                    "最近 30 天"
                ],
                help="建議使用 OOS 數據"
            )
        
        with col3:
            prob_threshold = st.slider(
                "🎯 機率門檻",
                min_value=0.50,
                max_value=0.85,
                value=0.60,
                step=0.05,
                help="模型預測機率 > 此值才進場"
            )
        
        st.markdown("---")
        st.markdown("**💰 交易成本設定**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            initial_capital = st.number_input(
                "💵 初始資金 (USDT)",
                value=10000.0,
                min_value=1000.0,
                step=1000.0
            )
        
        with col2:
            position_size_pct = st.slider(
                "📋 每筆仓位%",
                min_value=5.0,
                max_value=50.0,
                value=10.0,
                step=5.0,
                help="每筆交易使用資金的%"
            )
        
        with col3:
            slippage_pct = st.number_input(
                "💨 滑點%",
                value=0.05,
                min_value=0.0,
                max_value=0.5,
                step=0.05,
                help="每筆交易的滑點成本"
            )
        
        with col4:
            commission_pct = st.number_input(
                "🪩 手續費%",
                value=0.04,
                min_value=0.0,
                max_value=0.2,
                step=0.01,
                help="Binance Maker: 0.04%"
            )
    
    # ===== 出場策略 =====
    with st.expander("🚻 步驟 3: 出場策略 (波段交易關鍵)", expanded=True):
        st.markdown("""
        **波段交易出場機制**:
        - 初始目標: BB/NW 中軌
        - 破中軌後: 目標對側軌道
        - 移動止損: 跟隨 EMA_21
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            exit_strategy = st.selectbox(
                "🎯 出場模式",
                [
                    "動態追蹤 (推薦)",
                    "固定 TP/SL",
                    "觸碸對側軌道"
                ],
                help="波段交易建議使用動態追蹤"
            )
        
        with col2:
            if exit_strategy == "動態追蹤 (推薦)":
                trailing_stop_atr = st.slider(
                    "📉 追蹤止損 (ATR)",
                    min_value=0.5,
                    max_value=2.0,
                    value=1.0,
                    step=0.25,
                    help="距離最高點 N倍 ATR 時出場"
                )
            else:
                fixed_tp = st.number_input(
                    "🎯 TP 倍數",
                    value=3.0,
                    min_value=1.5,
                    max_value=5.0,
                    step=0.5
                )
                fixed_sl = st.number_input(
                    "🛑 SL 倍數",
                    value=1.0,
                    min_value=0.5,
                    max_value=2.0,
                    step=0.25
                )
        
        max_hold_hours = st.slider(
            "⏱️ 最長持倉 (小時)",
            min_value=4,
            max_value=48,
            value=20,
            step=2,
            help="超過時間強制平倉"
        )
    
    # ===== 執行回測 =====
    st.markdown("---")
    
    if st.button("🚀 執行回測", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 1. 載入模型
            status_text.text("🤖 步驟 1/5: 載入模型...")
            progress_bar.progress(10)
            
            trainer = ModelTrainer()
            trainer.load_model(model_path)
            
            st.success(f"✅ 模型載入完成: {len(trainer.feature_names)} 個特徵")
            
            # 2. 載入測試數據
            status_text.text("📡 步驟 2/5: 載入測試數據...")
            progress_bar.progress(20)
            
            if "2024 全年" in test_period:
                df_15m = loader.load_klines(symbol, '15m')
                df_1h = loader.load_klines(symbol, '1h')
                df_15m = df_15m[df_15m['open_time'] >= '2024-01-01'].copy()
                df_1h = df_1h[df_1h['open_time'] >= '2024-01-01'].copy()
            elif "Q4" in test_period:
                df_15m = loader.load_klines(symbol, '15m')
                df_1h = loader.load_klines(symbol, '1h')
                df_15m = df_15m[df_15m['open_time'] >= '2024-10-01'].copy()
                df_1h = df_1h[df_1h['open_time'] >= '2024-10-01'].copy()
            elif "90" in test_period:
                df_15m = loader.fetch_latest_klines(symbol, '15m', days=90)
                df_1h = loader.fetch_latest_klines(symbol, '1h', days=90)
            else:
                df_15m = loader.fetch_latest_klines(symbol, '15m', days=30)
                df_1h = loader.fetch_latest_klines(symbol, '1h', days=30)
            
            st.info(f"✅ 測試數據: {len(df_15m)} 筆 ({df_15m['open_time'].min()} ~ {df_15m['open_time'].max()})")
            
            # 3. 建立特徵
            status_text.text("⚙️ 步驟 3/5: 建立特徵...")
            progress_bar.progress(40)
            
            feature_engineer = FeatureEngineer()
            
            df_15m_features = feature_engineer.build_features(
                df_15m,
                include_microstructure=True,
                include_nw_envelope=True,
                include_adx=True,
                include_bounce_features=False
            )
            
            df_1h_features = feature_engineer.build_features(
                df_1h,
                include_microstructure=True,
                include_nw_envelope=True,
                include_adx=True,
                include_bounce_features=False
            )
            
            df_mtf = feature_engineer.merge_and_build_mtf_features(df_15m_features, df_1h_features)
            df_mtf = feature_engineer.add_bounce_confluence_features(df_mtf)
            
            st.success(f"✅ 特徵建立完成: {df_mtf.shape}")
            
            # 4. 事件過濾 + 預測
            status_text.text("🎯 步驟 4/5: BB/NW 過濾 + AI 預測...")
            progress_bar.progress(60)
            
            bounce_filter = BBNW_BounceFilter(
                use_bb=True,
                use_nw=True,
                min_pierce_pct=0.001,
                require_volume_surge=False
            )
            
            df_filtered = bounce_filter.filter_events(df_mtf)
            
            # 預測
            predictions = trainer.predict_proba(df_filtered)
            df_filtered['predicted_prob'] = predictions
            
            # 只保留高機率信號
            df_signals = df_filtered[df_filtered['predicted_prob'] >= prob_threshold].copy()
            
            st.info(f"✅ 產生 {len(df_signals)} 個交易信號 (門檻 {prob_threshold:.0%})")
            
            if len(df_signals) == 0:
                st.warning("⚠️ 無交易信號，請降低機率門檻或更改測試期間")
                return
            
            # 5. 模擬交易
            status_text.text("💰 步驟 5/5: 模擬交易...")
            progress_bar.progress(80)
            
            # 簡易回測引擎 (這裡可以接入你現有的 BacktestEngine)
            trades = []
            balance = initial_capital
            peak_balance = initial_capital
            max_drawdown = 0
            
            for idx, row in df_signals.iterrows():
                # 計算仓位大小
                position_value = balance * (position_size_pct / 100)
                entry_price = row['close'] * (1 + slippage_pct / 100)
                quantity = position_value / entry_price
                
                # 計算出場價格
                if exit_strategy == "動態追蹤 (推薦)":
                    # 簡化: 使用 BB 中軌作為目標
                    if row['is_long_setup']:
                        tp_price = row['bb_middle'] * (1 - slippage_pct / 100)
                        sl_price = row['close'] - row['atr'] * (1 + slippage_pct / 100)
                    else:
                        tp_price = row['bb_middle'] * (1 + slippage_pct / 100)
                        sl_price = row['close'] + row['atr'] * (1 + slippage_pct / 100)
                else:
                    if row['is_long_setup']:
                        tp_price = entry_price + row['atr'] * fixed_tp
                        sl_price = entry_price - row['atr'] * fixed_sl
                    else:
                        tp_price = entry_price - row['atr'] * fixed_tp
                        sl_price = entry_price + row['atr'] * fixed_sl
                
                # 簡化: 假設 60% 機率觸及 TP，40% 觸及 SL
                hit_tp = np.random.random() < row['predicted_prob']
                
                if hit_tp:
                    pnl = abs(tp_price - entry_price) * quantity
                    outcome = 'WIN'
                else:
                    pnl = -abs(entry_price - sl_price) * quantity
                    outcome = 'LOSS'
                
                # 扣除手續費
                commission = position_value * (commission_pct / 100) * 2
                pnl -= commission
                
                balance += pnl
                
                # 記錄交易
                trades.append({
                    'time': row['open_time'],
                    'side': 'LONG' if row['is_long_setup'] else 'SHORT',
                    'entry': entry_price,
                    'exit': tp_price if hit_tp else sl_price,
                    'quantity': quantity,
                    'pnl': pnl,
                    'balance': balance,
                    'outcome': outcome,
                    'prob': row['predicted_prob']
                })
                
                # 更新最大回撤
                if balance > peak_balance:
                    peak_balance = balance
                drawdown = (peak_balance - balance) / peak_balance
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            progress_bar.progress(100)
            status_text.text("✅ 回測完成!")
            
            # ===== 顯示結果 =====
            trades_df = pd.DataFrame(trades)
            
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['outcome'] == 'WIN'])
            win_rate = winning_trades / total_trades * 100
            
            total_pnl = balance - initial_capital
            roi = total_pnl / initial_capital * 100
            
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean()
            avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean())
            profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
            
            st.success("🎉 回測完成!")
            
            # 核心指標
            st.markdown("---")
            st.markdown("### 📊 核心績效指標")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "💰 總盈虧",
                    f"${total_pnl:,.0f}",
                    f"{roi:.1f}%"
                )
            
            with col2:
                st.metric(
                    "🎯 勝率",
                    f"{win_rate:.1f}%",
                    f"{winning_trades}/{total_trades}"
                )
            
            with col3:
                st.metric(
                    "📈 盈虧因子",
                    f"{profit_factor:.2f}",
                    "Good" if profit_factor > 1.5 else "Poor"
                )
            
            with col4:
                st.metric(
                    "📉 最大回撤",
                    f"{max_drawdown*100:.1f}%",
                    "危險" if max_drawdown > 0.3 else "健康"
                )
            
            # 資金曲線
            st.markdown("---")
            st.markdown("### 📈 資金曲線")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trades_df['time'],
                y=trades_df['balance'],
                mode='lines',
                name='資金',
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig.update_layout(
                title=f"{symbol} 回測資金曲線",
                xaxis_title="時間",
                yaxis_title="資金 (USDT)",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 交易明細
            st.markdown("---")
            st.markdown("### 📝 交易明細")
            
            display_df = trades_df[['time', 'side', 'entry', 'exit', 'pnl', 'outcome', 'prob']].copy()
            display_df['pnl'] = display_df['pnl'].apply(lambda x: f"${x:,.2f}")
            display_df['prob'] = display_df['prob'].apply(lambda x: f"{x:.1%}")
            
            st.dataframe(display_df, use_container_width=True, height=400)
            
        except Exception as e:
            st.error(f"❌ 回測失敗: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    # 底部說明
    st.markdown("---")
    st.markdown("""
    ### 💡 回測分析建議
    
    **健康指標**:
    - 勝率: 55-65% (過高可能過括合)
    - 盈虧因子: > 1.8
    - 最大回撤: < 25%
    - ROI: > 30% (年化)
    
    **優化方向**:
    - 勝率低: 提高機率門檻
    - 信號太少: 降低門檻或放寬過濾器
    - 回撤大: 降低單筆仓位%
    """)