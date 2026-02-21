import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core import (
    CryptoDataLoader, FeatureEngineer, TripleBarrierLabeling,
    MetaLabeling, ModelTrainer, EventFilter
)

def render():
    st.title("模型訓練 (MTF 多時間框架版)")
    
    st.markdown("""
    使用進階機器學習訓練交易模型 (MTF Confluence System):
    - **架構**: 15m 進場與微觀結構 + 1h 環境與趨勢過濾
    - **特徵**: 包含 4 款 MTF 獨家 Alpha 特徵 (MVR, CVD Fractal, VWWA, HTF Trend)
    - **平穩性**: 全面封殺絕對值特徵，只使用比例與標準化指標
    """)
    
    with st.expander("⚠️ 重要:特徵平穩性說明", expanded=False):
        st.markdown("""
        ### 為什麼必須移除絕對值特徵?
        
        **致命問題**:
        - 訓練時 BTC = $30,000 → `bb_middle = 30000`
        - 回測時 BTC = $90,000 → `bb_middle = 90000`
        - 模型規則: `if bb_middle > 45000: ...` 完全失效!
        
        **已封殺的危險特徵**:
        - ✅ 絕對價格: open, high, low, close (及其 1h 版本)
        - ✅ 絕對 BB: bb_middle, bb_upper, bb_lower (及其 1h 版本)
        - ✅ 絕對成交量: volume, volume_ma_20 (及其 1h 版本)
        - ✅ 報價總量: quote_volume, taker_buy_quote_asset_volume (及其 1h 版本)
        - ✅ API 不穩定欄位: trades
        
        **保留的平穩特徵**:
        - ✓ MTF Alpha: MVR, CVD Fractal, VWWA, HTF Trend Age
        - ✓ 比例特徵: bb_width_pct, volume_ratio, taker_buy_ratio
        - ✓ 標準化: rsi_normalized, cvd_norm_10
        - ✓ 距離比: ema_9_dist, ema_21_dist
        - ✓ 影線比: upper_wick_ratio, lower_wick_ratio
        """)
    
    st.markdown("---")
    
    with st.expander("訓練配置", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            loader = CryptoDataLoader()
            symbol = st.selectbox("訓練交易對", loader.get_available_symbols(), index=10)
            st.info("系統架構: 15m (進場與微觀結構) + 1h (環境與趨勢過濾)")
            
            tp_multiplier = st.number_input("止盈倍數 (ATR)", value=3.0, step=0.5)
            sl_multiplier = st.number_input("止損倍數 (ATR)", value=1.0, step=0.25)
        
        with col2:
            max_holding_bars = st.number_input("最大持倉根數 (15m K線)", value=48, step=1, help="48根 = 12小時")
            n_splits = st.number_input("交叉驗證折數", value=5, step=1, min_value=3, max_value=10)
            embargo_pct = st.number_input("禁止區百分比", value=0.01, step=0.01, format="%.3f")
            use_calibration = st.checkbox("啟用機率校準", value=True)
    
    with st.expander("嚴格事件過濾", expanded=True):
        st.markdown("""
        **三重確認 (AND)**: 1)成交量爆發 2)價格突破20期 3)波動率爆發
        """)
        
        use_event_filter = st.checkbox("啟用嚴格過濾", value=True)
        
        col1, col2 = st.columns(2)
        with col1:
            min_volume_ratio = st.number_input("最小成交量比率 (15m)", value=2.0, step=0.1)
            use_strict = st.checkbox("嚴格模式 (AND)", value=True)
        with col2:
            min_vsr = st.number_input("最小波動率", value=1.0, step=0.1)
            bb_squeeze = st.number_input("BB壓縮門檻", value=0.5, step=0.1)
            lookback_period = st.number_input("突破回看週期 (K線)", value=40, step=10)
    
    with st.expander("進階配置", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            slippage = st.number_input("滑點", value=0.001, step=0.0001, format="%.4f")
            time_decay_lambda = st.number_input("時間衰減係數", value=2.0, step=0.5)
            quality_alpha = st.number_input("質量微調係數", value=0.5, step=0.1,
                                           help="微調範圍 1.0-1.5, 不再放大基礎權重")
        with col2:
            use_quality_weight = st.checkbox("啟用質量微調", value=False,
                                            help="關閉後所有樣本權重為1.0")
            use_class_weight = st.checkbox("使用類別權重平衡", value=False)
    
    with st.expander("模型超參數", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            max_depth = st.number_input("最大深度", value=4, min_value=3, max_value=6)
            learning_rate = st.number_input("學習率", value=0.02, step=0.01, format="%.3f")
            n_estimators = st.number_input("樹的數量", value=300, step=50)
        with col2:
            min_child_weight = st.number_input("最小子節點權重", value=5, min_value=3, max_value=10)
            subsample = st.number_input("子樣本比例", value=0.7, step=0.1, format="%.2f")
            colsample_bytree = st.number_input("特徵採樣比例", value=0.6, step=0.1, format="%.2f")
    
    model_name = st.text_input("模型名稱", value=f"{symbol}_MTF_15m_1h_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
    
    if st.button("開始訓練", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("載入 15m 與 1h 數據...")
            progress_bar.progress(5)
            df_15m = loader.load_klines(symbol, '15m')
            df_1h = loader.load_klines(symbol, '1h')
            
            st.info(f"載入完成: 15m ({len(df_15m)} 筆), 1h ({len(df_1h)} 筆)")
            st.info(f"數據範圏: {df_15m['open_time'].min()} 至 {df_15m['open_time'].max()}")
            
            status_text.text("建立單一週期特徵...")
            progress_bar.progress(10)
            feature_engineer = FeatureEngineer()
            
            df_15m_features = feature_engineer.build_features(df_15m, include_microstructure=True)
            df_1h_features = feature_engineer.build_features(df_1h, include_microstructure=True)
            
            status_text.text("合併多時間框架 (MTF) 特徵...")
            progress_bar.progress(15)
            
            df_mtf = feature_engineer.merge_and_build_mtf_features(df_15m_features, df_1h_features)
            st.success(f"MTF 特徵合併完成! 最終數據形狀: {df_mtf.shape}")
            
            df_features = df_mtf
            
            if use_event_filter:
                status_text.text("應用嚴格事件過濾...")
                progress_bar.progress(20)
                
                event_filter = EventFilter(
                    use_strict_mode=use_strict,
                    min_volume_ratio=min_volume_ratio,
                    min_vsr=min_vsr,
                    bb_squeeze_threshold=bb_squeeze,
                    lookback_period=int(lookback_period)
                )
                
                df_filtered = event_filter.filter_events(df_features)
                filter_ratio = len(df_filtered) / len(df_features)
                st.info(f"事件過濾: {len(df_features)} → {len(df_filtered)} ({100*filter_ratio:.1f}%)")
                
                if filter_ratio > 0.25:
                    st.warning(f"過濾後仍保留 {filter_ratio*100:.0f}% (建議調整參數以低於 25%)")
                
                df_features = df_filtered
            
            status_text.text("應用三重屏障標記...")
            progress_bar.progress(25)
            
            labeler = TripleBarrierLabeling(
                tp_multiplier=tp_multiplier,
                sl_multiplier=sl_multiplier,
                max_holding_bars=int(max_holding_bars),
                slippage=slippage,
                time_decay_lambda=time_decay_lambda,
                quality_weight_alpha=quality_alpha,
                use_quality_weight=use_quality_weight
            )
            df_labeled = labeler.apply_triple_barrier(df_features)
            
            positive_count = (df_labeled['label'] == 1).sum()
            negative_count = (df_labeled['label'] == 0).sum()
            positive_pct = positive_count / len(df_labeled) * 100
            
            avg_weight_pos = df_labeled[df_labeled['label'] == 1]['sample_weight'].mean()
            avg_weight_neg = df_labeled[df_labeled['label'] == 0]['sample_weight'].mean()
            
            st.info(f"標籤分布: {positive_pct:.1f}% 正樣本 ({positive_count} 勝, {negative_count} 負)")
            st.info(f"樣本權重 - 正類: {avg_weight_pos:.2f}, 負類: {avg_weight_neg:.2f}")
            
            status_text.text("準備訓練數據 (特徵大掃除)...")
            progress_bar.progress(35)
            
            st.warning("⚠️ 特徵大掃除:移除絕對值與 API 不穩定特徵")
            
            base_cols = [
                'open_time', 'close_time', 'htf_close_time',
                'label', 'label_return', 'hit_time', 'exit_type', 'sample_weight', 'mae_ratio',
                'exit_price', 'exit_bars', 'return', 'ignore'
            ]
            
            # ===== [修正] 封殺 taker_buy_quote_asset_volume (誤用非平稩特徵) =====
            forbidden_features = [
                # 絕對價格
                'open', 'high', 'low', 'close',
                'bb_middle', 'bb_upper', 'bb_lower', 'bb_std',
                
                # 絕對成交量
                'volume', 'volume_ma_20',
                'taker_buy_base_asset_volume',
                
                # 報價總量 (非平稩特徵，因價格波動而變動)
                'quote_asset_volume', 'quote_volume',
                'taker_buy_quote_asset_volume',  # <- 致命洩漏
                
                # API 不穩定欄位
                'number_of_trades', 'trades',
                'open_interest', 'atr',
                
                # 1h 絕對特徵
                'open_1h', 'high_1h', 'low_1h', 'close_1h',
                'bb_middle_1h', 'bb_upper_1h', 'bb_lower_1h', 'bb_std_1h',
                'volume_1h', 'volume_ma_20_1h',
                'taker_buy_base_asset_volume_1h',
                'quote_asset_volume_1h', 'quote_volume_1h',
                'taker_buy_quote_asset_volume_1h',  # <- 致命洩漏 (1h 版本)
                'number_of_trades_1h', 'trades_1h', 
                'open_interest_1h', 'atr_1h'
            ]
            
            exclude_all = base_cols + forbidden_features
            
            feature_cols = [col for col in df_labeled.columns if col not in exclude_all]
            feature_cols = [col for col in feature_cols 
                          if df_labeled[col].dtype in ['int64', 'float64', 'bool', 'int32', 'float32']]
            
            removed_features = [col for col in df_labeled.columns if col in forbidden_features]
            if len(removed_features) > 0:
                st.info(f"✅ 移除 {len(removed_features)} 個非平稩特徵 (含1h版本)")
            
            X = df_labeled[feature_cols].copy()
            y = df_labeled['label'].copy()
            sample_weights = df_labeled['sample_weight'].values
            
            X = X.fillna(0)
            X = X.replace([np.inf, -np.inf], 0)
            
            for col in X.select_dtypes(include=['bool']).columns:
                X[col] = X[col].astype(int)
            
            st.info(f"訓練數據: {len(X)} 樣本, {len(feature_cols)} 特徵")
            
            with st.expander("保留的平稩特徵 (點擊查看)", expanded=False):
                st.code('\n'.join(feature_cols))
            
            status_text.text("Purged K-Fold 訓練...")
            progress_bar.progress(50)
            
            trainer = ModelTrainer(use_calibration=use_calibration)
            
            scale_pos_weight = negative_count / positive_count if use_class_weight and positive_count > 0 else 1.0
            if scale_pos_weight != 1.0:
                st.warning(f"類別權重: {scale_pos_weight:.2f}")
            else:
                st.info("類別權重: 1.0")
            
            params = {
                'max_depth': int(max_depth),
                'learning_rate': float(learning_rate),
                'n_estimators': int(n_estimators),
                'min_child_weight': int(min_child_weight),
                'subsample': float(subsample),
                'colsample_bytree': float(colsample_bytree),
                'scale_pos_weight': scale_pos_weight,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0
            }
            
            cv_metrics = trainer.train_with_purged_kfold(
                X, y,
                sample_weights=sample_weights,
                n_splits=int(n_splits),
                embargo_pct=float(embargo_pct),
                params=params
            )
            
            progress_bar.progress(90)
            status_text.text("保存模型...")
            trainer.save_model(model_name)
            
            progress_bar.progress(100)
            status_text.text("訓練完成")
            
            st.success(f"模型已保存: {model_name}")
            
            st.markdown("### 交叉驗證結果")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("準確率", f"{cv_metrics.get('cv_val_accuracy', 0):.4f} ± {cv_metrics.get('cv_val_accuracy_std', 0):.4f}")
            with col2:
                auc_val = cv_metrics.get('cv_val_auc', 0)
                auc_delta = auc_val - 0.5
                st.metric("AUC", f"{auc_val:.4f}",
                         delta=f"+{auc_delta:.4f}")
            with col3:
                prec = cv_metrics.get('cv_val_precision', 0)
                st.metric("精確率", f"{prec:.4f}")
            with col4:
                recall = cv_metrics.get('cv_val_recall', 0)
                st.metric("召回率", f"{recall:.4f}")
            
            if prec > 0 and recall > 0:
                ev = (prec * tp_multiplier) - ((1 - prec) * sl_multiplier)
                st.info(f"期望值 (EV): {ev:.3f}R ({'positive' if ev > 0 else 'negative'})")
                if ev > 0.2:
                    st.success(f"模型優秀! AUC={auc_val:.3f}, EV={ev:.2f}R")
                elif ev > 0:
                    st.info(f"模型合格! EV={ev:.2f}R > 0")
            
            if auc_val < 0.55:
                st.error(f"AUC {auc_val:.3f} < 0.55")
            
            if recall > 0.70 and prec < 0.40:
                st.warning("召回率過高,模型濾發信號")
            
            st.markdown("### 特徵重要性 (前 20 名)")
            feature_importance = trainer.get_feature_importance()
            
            mtf_features = ['mvr', 'cvd_fractal', 'vwwa_buy', 'vwwa_sell', 'htf_trend_age_norm']
            
            top_15 = feature_importance.head(15)['feature'].tolist()
            mtf_in_top = [f for f in mtf_features if f in top_15]
            
            if len(mtf_in_top) > 0:
                st.success(f"✅ {len(mtf_in_top)} 個 MTF Alpha 特徵在 Top 15: {', '.join(mtf_in_top)}")
            else:
                st.info("MTF 特徵未進入前 15 名，請檢查數據量或參數")
                
            st.dataframe(feature_importance.head(20), use_container_width=True)
            
            # ===== [新增] 詳細訓練報告 (可複製給 Gemini) =====
            st.markdown("---")
            st.markdown("### 📋 詳細訓練報告 (可複製給 Gemini 查看)")
            
            report = f"""
# MTF 多時間框架交易模型訓練報告

## 模型配置
- **交易對**: {symbol}
- **時間框架**: 15m (進場) + 1h (環境過濾)
- **模型檔案**: {model_name}
- **訓練時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 數據概要
- **15m 數據**: {len(df_15m)} 筆
- **1h 數據**: {len(df_1h)} 筆
- **MTF 合併後**: {df_mtf.shape[0]} 筆, {df_mtf.shape[1]} 欄
- **事件過濾後**: {len(df_features)} 筆 ({100*len(df_features)/len(df_mtf):.1f}%)
- **數據範圏**: {df_15m['open_time'].min()} 至 {df_15m['open_time'].max()}

## 標籤配置
- **止盈 (TP)**: {tp_multiplier:.1f} ATR
- **止損 (SL)**: {sl_multiplier:.1f} ATR
- **最大持倉**: {max_holding_bars} 根 15m K線 ({max_holding_bars/4:.1f} 小時)
- **樣本權重**: {'Enabled' if use_quality_weight else 'Disabled'}
- **類別權重**: {scale_pos_weight:.2f}

## 標籤分布
- **正樣本 (勝)**: {positive_count} ({positive_pct:.1f}%)
- **負樣本 (負)**: {negative_count} ({100-positive_pct:.1f}%)
- **平均權重 (正)**: {avg_weight_pos:.2f}
- **平均權重 (負)**: {avg_weight_neg:.2f}

## 特徵工程
- **原始特徵數**: {len(df_labeled.columns)} 個
- **移除非平稩特徵**: {len(removed_features)} 個
- **最終保留特徵**: {len(feature_cols)} 個
- **MTF Alpha 特徵**: {', '.join(mtf_features)}

## 交叉驗證結果 (Purged K-Fold, {n_splits} Folds)
- **準確率**: {cv_metrics.get('cv_val_accuracy', 0):.4f} ± {cv_metrics.get('cv_val_accuracy_std', 0):.4f}
- **AUC**: {auc_val:.4f} (delta: +{auc_delta:.4f})
- **精確率 (Precision)**: {prec:.4f}
- **召回率 (Recall)**: {recall:.4f}
- **期望值 (EV)**: {ev:.3f}R

## 模型超參數
- **max_depth**: {max_depth}
- **learning_rate**: {learning_rate}
- **n_estimators**: {n_estimators}
- **min_child_weight**: {min_child_weight}
- **subsample**: {subsample}
- **colsample_bytree**: {colsample_bytree}

## 特徵重要性 Top 20
{feature_importance.head(20).to_string()}

## MTF Alpha 特徵在 Top 15
{', '.join(mtf_in_top) if len(mtf_in_top) > 0 else 'None'}

## 保留的平稩特徵列表
{chr(10).join(feature_cols)}
"""
            
            st.text_area("報告內容 (點擊右上角複製)", report, height=400)
            
            st.markdown("### 下一步")
            if auc_val >= 0.58:
                st.success("✅ 模型訓練成功!現在可以進行 **OOS 盲測**")
                st.info("""
                **第二階段檢查清單**:
                1. 前往 **回測分析**
                2. 選擇 MTF 訓練出的模型
                3. 數據來源: **Binance API (最新 90 天)**
                4. 設定風控: 機率門檻 0.55, 單筆風險 2%, TP 3.0 / SL 1.0
                5. 啟用嚴格事件過濾 (與訓練時完全一致)
                """)
            
        except Exception as e:
            st.error(f"訓練失敗: {str(e)}")
            import traceback
            st.code(traceback.format_exc())