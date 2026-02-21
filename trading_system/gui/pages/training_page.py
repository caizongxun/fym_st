import streamlit as st
import pandas as pd
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core import (
    CryptoDataLoader, FeatureEngineer, ModelTrainer,
    TripleBarrierLabeling, EventFilter
)
from core.event_filter import BBNW_BounceFilter

def render():
    st.title("🧪 BB+NW 波段反轉系統 - 模型訓練")
    
    st.markdown("""
    ---
    ### 🎯 系統訓練流程
    
    這是一套專為 **15m 波段反轉** 設計的三層架構:
    
    1. **觸發層** (Event Trigger): 只在價格觸碸 BB/NW 軌道時啟動
    2. **特徵層** (Features): ADX 趨勢 + CVD 流動性 + VWWA 影線吸收
    3. **AI 層** (Meta-Label): LightGBM 判斷是否為「真反彈」
    
    ---
    """)
    
    # ===== 第一步: 數據準備 =====
    with st.expander("📂 步驟 1: 數據載入與範圍選擇", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            loader = CryptoDataLoader()
            symbol = st.selectbox(
                "🪙 交易對",
                loader.get_available_symbols(),
                index=10,
                help="建議選擇高流動性幣種 (BTC, ETH)"
            )
        
        with col2:
            data_source = st.radio(
                "📡 數據來源",
                ["HuggingFace (快速)", "Binance API (最新)"],
                help="HF: 2020-2024 歷史數據 | API: 即時數據"
            )
            
            if data_source == "Binance API (最新)":
                training_days = st.number_input(
                    "📅 訓練天數",
                    value=180,
                    min_value=90,
                    max_value=730,
                    step=30,
                    help="建議 180-365 天"
                )
            else:
                use_recent_only = st.checkbox(
                    "只使用 2024 數據 (OOS)",
                    value=True,
                    help="Out-of-Sample 測試"
                )
        
        with col3:
            st.info("""
            **🔒 時間框架鎖定**
            
            進場: 15m  
            趨勢: 1h (MTF)
            
            系統會自動載入雙時間框架數據
            """)
    
    # ===== 第二步: 特徵工程配置 =====
    with st.expander("⚙️ 步驟 2: 特徵工程配置 (核心)", expanded=True):
        st.markdown("""
        **必須啟用** (已預設):
        - ✅ Nadaraya-Watson 包絡線 (無未來函數)
        - ✅ ADX 趨勢強度指標
        - ✅ 波段反轉共振特徵 (CVD 背離, VWWA)
        - ✅ 訂單流微觀結構 (8 個核心特徵)
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**NW 指標參數**")
            nw_h = st.slider(
                "🌊 平滑度 (h)",
                min_value=4.0,
                max_value=12.0,
                value=8.0,
                step=0.5,
                help="越小越敏感"
            )
            
            nw_mult = st.slider(
                "📊 通道倍數 (mult)",
                min_value=2.0,
                max_value=4.0,
                value=3.0,
                step=0.5,
                help="MAE 倍數"
            )
        
        with col2:
            st.markdown("**進階特徵**")
            include_oi = st.checkbox(
                "📈 未平倉量 (OI) 特徵",
                value=False,
                help="需要 OI 數據"
            )
            
            include_funding = st.checkbox(
                "💰 資金費率特徵",
                value=False,
                help="需要 Funding Rate 數據"
            )
    
    # ===== 第三步: 事件過濾器 =====
    with st.expander("🎯 步驟 3: BB/NW 觸碸過濾器", expanded=True):
        st.markdown("""
        **觸發邏輯**: 只有當 K 線的 **Low 跌破 BB/NW 下軌** 或 **High 突破上軌** 時，才會被選入訓練集。
        
        這會將數據量激減至 **2-15%**，只保留極端反轉事件。
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            use_bb_trigger = st.checkbox(
                "✅ BB 通道觸發",
                value=True,
                help="Bollinger Bands"
            )
        
        with col2:
            use_nw_trigger = st.checkbox(
                "✅ NW 包絡線觸發",
                value=True,
                help="Nadaraya-Watson"
            )
        
        with col3:
            min_pierce = st.number_input(
                "🔬 刺穿容差%",
                value=0.1,
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                help="0.1% 的誤差範圍"
            )
        
        require_volume = st.checkbox(
            "📈 要求同時爆量",
            value=False,
            help="只保留有成交量爆增的觸碸事件"
        )
    
    # ===== 第四步: Triple Barrier 標註 =====
    with st.expander("🏷️ 步驟 4: 標籤設定 (Triple Barrier)", expanded=True):
        st.markdown("""
        **波段交易建議**:
        - TP/SL 比例: 2.5:1 ~ 4:1 (較大的盈虧比)
        - 持倉時間: 40-80 根 15m K線 (10-20 小時)
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tp_mult = st.number_input(
                "🎯 TP 倍數 (ATR)",
                value=3.0,
                min_value=2.0,
                max_value=5.0,
                step=0.5,
                help="建議 2.5-3.5"
            )
        
        with col2:
            sl_mult = st.number_input(
                "🛑 SL 倍數 (ATR)",
                value=1.0,
                min_value=0.5,
                max_value=2.0,
                step=0.25,
                help="建議 0.75-1.25"
            )
        
        with col3:
            max_hold = st.number_input(
                "⏱️ 最長持倉 (15m K線)",
                value=60,
                min_value=20,
                max_value=120,
                step=10,
                help="60 根 = 15 小時"
            )
    
    # ===== 第五步: 模型配置 =====
    with st.expander("🤖 步驟 5: 模型訓練參數", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "🏆 模型類型",
                ["LightGBM (推薦)", "XGBoost"],
                help="LightGBM 速度快且效果好"
            )
            
            cv_folds = st.slider(
                "🔁 交叉驗證折數",
                min_value=3,
                max_value=10,
                value=5,
                help="預設 5 折交叉驗證"
            )
        
        with col2:
            early_stop = st.number_input(
                "⏹️ 早停輪數",
                value=50,
                min_value=20,
                max_value=100,
                step=10,
                help="防止過括合"
            )
            
            model_name = st.text_input(
                "📝 模型名稱",
                value=f"{symbol}_15m_BB_NW_Bounce_v1",
                help="建議包含版本號"
            )
    
    # ===== 執行訓練 =====
    st.markdown("---")
    
    if st.button("🚀 開始訓練", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 步驟 1: 載入數據
            status_text.text("📡 步驟 1/6: 載入 15m + 1h 數據...")
            progress_bar.progress(10)
            
            if data_source == "Binance API (最新)":
                df_15m = loader.fetch_latest_klines(symbol, '15m', days=int(training_days))
                df_1h = loader.fetch_latest_klines(symbol, '1h', days=int(training_days))
            else:
                df_15m = loader.load_klines(symbol, '15m')
                df_1h = loader.load_klines(symbol, '1h')
                if use_recent_only:
                    df_15m = df_15m[df_15m['open_time'] >= '2024-01-01'].copy()
                    df_1h = df_1h[df_1h['open_time'] >= '2024-01-01'].copy()
            
            st.info(f"✅ 載入完成: 15m ({len(df_15m)} 筆) + 1h ({len(df_1h)} 筆)")
            st.info(f"📅 時間範圍: {df_15m['open_time'].min()} ~ {df_15m['open_time'].max()}")
            
            # 步驟 2: 建立特徵
            status_text.text("⚙️ 步驟 2/6: 建立 15m 特徵 (NW + ADX + CVD)...")
            progress_bar.progress(20)
            
            feature_engineer = FeatureEngineer()
            
            df_15m_features = feature_engineer.build_features(
                df_15m,
                include_microstructure=True,
                include_nw_envelope=True,
                include_adx=True,
                include_bounce_features=False,
                include_liquidity_features=include_oi or include_funding
            )
            
            st.success(f"✅ 15m 特徵完成: {df_15m_features.shape}")
            
            status_text.text("⚙️ 步驟 3/6: 建立 1h 特徵...")
            progress_bar.progress(30)
            
            df_1h_features = feature_engineer.build_features(
                df_1h,
                include_microstructure=True,
                include_nw_envelope=True,
                include_adx=True,
                include_bounce_features=False
            )
            
            st.success(f"✅ 1h 特徵完成: {df_1h_features.shape}")
            
            # 步驟 3: MTF 合併
            status_text.text("🔄 步驟 4/6: MTF 合併 + 波段反轉特徵...")
            progress_bar.progress(40)
            
            df_mtf = feature_engineer.merge_and_build_mtf_features(df_15m_features, df_1h_features)
            df_mtf = feature_engineer.add_bounce_confluence_features(df_mtf)
            
            st.success(f"✅ MTF 合併完成: {df_mtf.shape}")
            
            # 步驟 4: 事件過濾
            status_text.text("🎯 步驟 5/6: BB/NW 觸碸過濾...")
            progress_bar.progress(50)
            
            bounce_filter = BBNW_BounceFilter(
                use_bb=use_bb_trigger,
                use_nw=use_nw_trigger,
                min_pierce_pct=min_pierce / 100.0,
                require_volume_surge=require_volume,
                min_volume_ratio=1.2
            )
            
            df_filtered = bounce_filter.filter_events(df_mtf)
            
            filter_ratio = len(df_filtered) / len(df_mtf) * 100
            st.success(f"✅ 過濾完成: {len(df_mtf)} → {len(df_filtered)} ({filter_ratio:.1f}%)")
            
            if len(df_filtered) < 100:
                st.error("⚠️ 過濾後數據太少 (<100 筆)，請放寬參數或增加訓練天數")
                return
            
            # 步驟 5: Triple Barrier 標註
            status_text.text("🏷️ 步驟 6/6: Triple Barrier 標註...")
            progress_bar.progress(60)
            
            labeler = TripleBarrierLabeling(
                tp_multiplier=tp_mult,
                sl_multiplier=sl_mult,
                max_hold_bars=int(max_hold)
            )
            
            df_labeled = labeler.create_labels(df_filtered)
            
            # 統計標籤分布
            label_dist = df_labeled['label'].value_counts()
            win_rate = label_dist.get(1, 0) / len(df_labeled) * 100
            
            st.success(f"✅ 標註完成: {len(df_labeled)} 筆")
            st.info(f"🏆 勝: {label_dist.get(1, 0)} | 🛑 敗: {label_dist.get(-1, 0)} | ⌛ 超時: {label_dist.get(0, 0)}")
            st.info(f"📊 基礎勝率: {win_rate:.1f}%")
            
            if win_rate < 40:
                st.warning("⚠️ 基礎勝率 < 40%，建議調整 TP/SL 比例")
            
            # 步驟 6: 訓練模型
            status_text.text("🤖 步驟 7/7: 訓練 LightGBM 模型...")
            progress_bar.progress(70)
            
            trainer = ModelTrainer()
            
            model_type_key = 'lightgbm' if 'LightGBM' in model_type else 'xgboost'
            
            metrics = trainer.train(
                df_labeled,
                model_type=model_type_key,
                cv_folds=cv_folds,
                early_stopping_rounds=early_stop
            )
            
            progress_bar.progress(90)
            
            # 儲存模型
            model_path = f"{model_name}.pkl"
            trainer.save_model(model_path)
            
            progress_bar.progress(100)
            status_text.text("✅ 訓練完成!")
            
            # 顯示結果
            st.success(f"🎉 模型訓練成功! 已儲存為: {model_path}")
            
            st.markdown("---")
            st.markdown("### 📊 訓練指標")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🎯 CV AUC", f"{metrics.get('cv_auc_mean', 0):.3f}")
            with col2:
                st.metric("📈 CV Accuracy", f"{metrics.get('cv_accuracy_mean', 0):.3f}")
            with col3:
                st.metric("👑 特徵數量", len(trainer.feature_names))
            
            # 特徵重要性
            if hasattr(trainer.model, 'feature_importances_'):
                st.markdown("### 🔍 Top 20 重要特徵")
                
                importance_df = pd.DataFrame({
                    '特徵': trainer.feature_names,
                    '重要性': trainer.model.feature_importances_
                }).sort_values('重要性', ascending=False).head(20)
                
                st.dataframe(importance_df, use_container_width=True)
            
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ 訓練失敗: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    # 底部說明
    st.markdown("---")
    st.markdown("""
    ### 💡 訓練建議
    
    **初次訓練**:
    1. 使用 HuggingFace 數據 + 2024 OOS
    2. TP/SL = 3.0/1.0
    3. 啟用 BB + NW 雙觸發
    4. 目標勝率 55-65%
    
    **優化方向**:
    - 若勝率 > 70%: 提高 TP (追求更大盈虧比)
    - 若信號太少: 降低 min_pierce_pct
    - 若過括合: 縮短訓練天數或增加 early_stop
    """)