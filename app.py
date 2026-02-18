import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from io import StringIO
import os

from data.binance_loader import BinanceDataLoader
from data.huggingface_loader import HuggingFaceKlineLoader
from utils.bb_reversal_detector import BBReversalDetector
from utils.bb_reversal_features import BBReversalFeatureExtractor
from models.train_bb_reversal_model import BBReversalModelTrainer
from backtesting.engine import BacktestEngine
from utils.signal_generator_bb_reversal import BBReversalSignalGenerator

st.set_page_config(page_title="AI 加密貨幣交易儀表板", layout="wide")
st.title("AI 加密貨幣交易儀表板 - BB反轉系統")

st.sidebar.title("設定")
data_source = st.sidebar.radio(
    "資料源",
    ["HuggingFace (38幣)", "Binance API (即時)"],
    help="HuggingFace: 離線資料,快速穩定\nBinance: 即時資料,需網絡"
)

if data_source == "HuggingFace (38幣)":
    loader = HuggingFaceKlineLoader()
    st.sidebar.success("使用HuggingFace離線資料")
else:
    loader = BinanceDataLoader()
    st.sidebar.info("使用Binance即時資料")

def display_metrics(metrics):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("交易次數", metrics.get('total_trades', 0))
        st.metric("勝率", f"{metrics.get('win_rate', 0):.2f}%")
    with col2:
        st.metric("最終權益", f"${metrics.get('final_equity', 0):.2f}")
        st.metric("總回報", f"{metrics.get('total_return_pct', 0):.2f}%")
    with col3:
        st.metric("獲利因子", f"{metrics.get('profit_factor', 0):.2f}")
        st.metric("夏普比率", f"{metrics.get('sharpe_ratio', 0):.2f}")
    with col4:
        st.metric("最大回撤", f"{metrics.get('max_drawdown_pct', 0):.2f}%")
        avg_duration = metrics.get('avg_duration_min', 0)
        st.metric("平均持倉(分)", f"{avg_duration:.0f}" if avg_duration else "N/A")

def symbol_selector(key_prefix: str, multi: bool = False, default_symbols: list = None):
    if data_source == "HuggingFace (38幣)":
        symbol_groups = HuggingFaceKlineLoader.get_symbol_groups()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selection_mode = st.radio(
                "選擇模式",
                ["熱門Top10", "按分類", "手動輸入"],
                key=f"{key_prefix}_mode"
            )
        
        with col2:
            if selection_mode == "熱門Top10":
                top_symbols = HuggingFaceKlineLoader.get_top_symbols(10)
                if multi:
                    selected = st.multiselect(
                        "選擇幣種",
                        top_symbols,
                        default=default_symbols or top_symbols[:2],
                        key=f"{key_prefix}_top"
                    )
                else:
                    selected = [st.selectbox(
                        "選擇幣種",
                        top_symbols,
                        key=f"{key_prefix}_top_single"
                    )]
            
            elif selection_mode == "按分類":
                category = st.selectbox(
                    "選擇分類",
                    list(symbol_groups.keys()),
                    key=f"{key_prefix}_category"
                )
                symbols_in_category = symbol_groups[category]
                
                if multi:
                    selected = st.multiselect(
                        f"{category} 幣種",
                        symbols_in_category,
                        default=default_symbols or symbols_in_category[:2],
                        key=f"{key_prefix}_cat_multi"
                    )
                else:
                    selected = [st.selectbox(
                        f"{category} 幣種",
                        symbols_in_category,
                        key=f"{key_prefix}_cat_single"
                    )]
            
            else:
                if multi:
                    text_input = st.text_area(
                        "輸入幣種 (逗號分隔)",
                        value=",".join(default_symbols) if default_symbols else "BTCUSDT,ETHUSDT",
                        key=f"{key_prefix}_manual",
                        height=100
                    )
                    selected = [s.strip().upper() for s in text_input.split(',') if s.strip()]
                else:
                    selected = [st.text_input(
                        "輸入幣種",
                        value="BTCUSDT",
                        key=f"{key_prefix}_manual_single"
                    ).strip().upper()]
        
        return selected
    
    else:
        if multi:
            text_input = st.text_area(
                "交易對 (逗號分隔)",
                value="BTCUSDT,ETHUSDT",
                key=f"{key_prefix}_binance"
            )
            return [s.strip().upper() for s in text_input.split(',') if s.strip()]
        else:
            return [st.text_input(
                "交易對",
                value="BTCUSDT",
                key=f"{key_prefix}_binance_single"
            ).strip().upper()]

tabs = st.tabs(["BB反轉視覺化", "BB反轉訓練(OOS)", "BB模型回測"])

with tabs[0]:
    st.header("BB反轉點視覺化")
    
    st.info("""
    **BB觸碰反轉定義**:
    1. 價格觸碰BB上軌/下軌
    2. 過濾走勢中的觸碰 (假突破)
    3. 隨後N根K線出現有效反轉
    4. 確認回到BB中軌附近
    
    **標記說明**:
    - 紅色三角: 上軌觸碰後下跌反轉 (做空機會)
    - 綠色三角: 下軌觸碰後上漨反轉 (做多機會)
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        viz_symbols = symbol_selector("bb_viz", multi=False)
        viz_symbol = viz_symbols[0]
        viz_days = st.slider("顯示天數", min_value=3, max_value=30, value=7, key="viz_days")
        viz_candles = viz_days * 96
    
    with col2:
        st.subheader("參數設定")
        bb_period_viz = st.number_input("BB周期", min_value=10, max_value=50, value=20, key="bb_period_viz")
        bb_std_viz = st.number_input("BB標準差", min_value=1.0, max_value=3.0, value=2.0, step=0.5, key="bb_std_viz")
        touch_threshold_viz = st.slider("觸碰閾值 (%)", min_value=0.0, max_value=0.5, value=0.1, step=0.05, key="touch_threshold_viz") / 100
        min_reversal_viz = st.slider("最小反轉幅度 (%)", min_value=0.1, max_value=1.0, value=0.5, step=0.1, key="min_reversal_viz") / 100
    
    if st.button("生成BB反轉點圖表", key="gen_bb_viz", type="primary"):
        with st.spinner(f"載入 {viz_symbol} 數據..."):
            try:
                if isinstance(loader, BinanceDataLoader):
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=viz_days+1)
                    df = loader.load_historical_data(viz_symbol, '15m', start_date, end_date)
                else:
                    df = loader.load_klines(viz_symbol, '15m')
                    
                df = df.tail(viz_candles)
                
                detector = BBReversalDetector(
                    bb_period=bb_period_viz,
                    bb_std=bb_std_viz,
                    touch_threshold=touch_threshold_viz,
                    reversal_confirm_candles=5,
                    min_reversal_pct=min_reversal_viz,
                    trend_filter_enabled=True,
                    trend_lookback=10,
                    require_middle_return=True
                )
                
                df_result = detector.detect_reversals(df)
                
                st.subheader(f"{viz_symbol} BB反轉點分析")
                
                stats = detector.get_statistics(df_result)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("總反轉點", stats['total_reversals'])
                with col2:
                    st.metric("上軌反轉", f"{stats['upper_reversals']} ({stats['upper_success_rate']:.1f}%)")
                with col3:
                    st.metric("下軌反轉", f"{stats['lower_reversals']} ({stats['lower_success_rate']:.1f}%)")
                
                col4, col5 = st.columns(2)
                with col4:
                    st.metric("平均反轉幅度", f"{stats['avg_reversal_pct']:.2%}")
                    st.metric("總拒絕數", stats['total_rejected'])
                with col5:
                    if 'rejection_reasons' in stats and stats['rejection_reasons']:
                        st.write("拒絕原因:")
                        for reason, count in stats['rejection_reasons'].items():
                            st.text(f"- {reason}: {count}")
                
                fig = detector.plot_reversals(df_result, n_candles=viz_candles, title=f"{viz_symbol} BB反轉點檢測")
                st.plotly_chart(fig, use_container_width=True)
                
                if len(detector.reversals) > 0:
                    st.subheader("反轉點明細")
                    reversals_df = pd.DataFrame(detector.reversals)
                    reversals_df['reversal_pct'] = reversals_df['reversal_pct'].apply(lambda x: f"{x:.2%}")
                    st.dataframe(reversals_df[['time', 'type', 'reversal_type', 'touch_price', 'target_price', 'reversal_pct']], use_container_width=True)
                else:
                    st.warning("沒有檢測到符合條件的反轉點，請降低最小反轉幅度或調整參數")
                    
            except Exception as e:
                st.error(f"錯誤: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

with tabs[1]:
    st.header("BB反轉點模型訓練 (OOS驗證)")
    
    st.success("""
    **OOS (Out-of-Sample) 驗證流程**:
    1. 載入全部數據
    2. 最後30天作OOS測試集
    3. OOS之前的20000根K棒作訓練集
    4. 訓練模型後在OOS上驗證泛化能力
    
    **標籤定義**:
    - 上軌反轉 -> 做空 (0)
    - 下軌反轉 -> 做多 (1)
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        train_symbols = symbol_selector("bb_train", multi=False)
        train_symbol = train_symbols[0]
        
        train_candles = st.number_input(
            "訓練K棒數量",
            min_value=10000,
            max_value=50000,
            value=20000,
            step=5000,
            key="train_candles",
            help="OOS之前的K棒數量用於訓練"
        )
        
        oos_days = st.number_input(
            "OOS測試天數",
            min_value=7,
            max_value=60,
            value=30,
            step=7,
            key="oos_days",
            help="最後N天作為OOS測試集"
        )
    
    with col2:
        st.subheader("反轉檢測參數")
        bb_period_train = st.number_input("BB周期", min_value=10, max_value=50, value=20, key="bb_period_train")
        bb_std_train = st.number_input("BB標準差", min_value=1.0, max_value=3.0, value=2.0, step=0.5, key="bb_std_train")
        touch_threshold_train = st.slider("觸碰閾值 (%)", min_value=0.0, max_value=0.5, value=0.1, step=0.05, key="touch_threshold_train") / 100
        min_reversal_train = st.slider("最小反轉幅度 (%)", min_value=0.1, max_value=1.0, value=0.5, step=0.1, key="min_reversal_train") / 100
    
    oos_candles = oos_days * 96
    st.caption(f"訓練: {train_candles}根 | OOS: {oos_candles}根({oos_days}天) | BB({bb_period_train},{bb_std_train}) | 觸碰:{touch_threshold_train*100:.2f}% | 反轉:{min_reversal_train*100:.1f}%")
    
    if st.button("開始OOS訓練+驗證", key="train_bb_oos_btn", type="primary"):
        with st.spinner(f"正在訓練 {train_symbol} BB反轉模型 (OOS模式)..."):
            try:
                # 載入全部數據
                if isinstance(loader, BinanceDataLoader):
                    # 加載足夠的數據用於訓練+OOS
                    # 估算需要的天數
                    total_candles = train_candles + oos_candles
                    days_needed = (total_candles / 96) + 5 # 緩衝
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=days_needed)
                    df_all = loader.load_historical_data(train_symbol, '15m', start_date, end_date)
                else:
                    df_all = loader.load_klines(train_symbol, '15m')
                
                # 分割OOS
                df_oos = df_all.tail(oos_candles).copy()
                df_train_full = df_all.iloc[:-oos_candles].copy()
                df_train = df_train_full.tail(train_candles).copy()
                
                st.info(f"數據分割: 訓練集={len(df_train)}根 | OOS={len(df_oos)}根({oos_days}天)")
                
                # ====== 訓練階段 ======
                st.subheader("階段 1: 訓練集處理")
                
                extractor = BBReversalFeatureExtractor(
                    bb_period=bb_period_train,
                    bb_std=bb_std_train,
                    rsi_period=14
                )
                
                extractor.detector = BBReversalDetector(
                    bb_period=bb_period_train,
                    bb_std=bb_std_train,
                    touch_threshold=touch_threshold_train,
                    reversal_confirm_candles=5,
                    min_reversal_pct=min_reversal_train,
                    trend_filter_enabled=True,
                    trend_lookback=10,
                    require_middle_return=True
                )
                
                df_train_processed = extractor.process(df_train, create_labels=True)
                train_stats = extractor.get_reversal_statistics()
                
                st.info(f"訓練集反轉點: {train_stats['total_reversals']} (上:{train_stats['upper_reversals']}, 下:{train_stats['lower_reversals']})")
                
                if train_stats['total_reversals'] < 50:
                    st.error(f"訓練集反轉點太少: {train_stats['total_reversals']}")
                    st.stop()
                
                X_train, y_train = extractor.get_training_data(df_train_processed)
                st.info(f"訓練樣本: {len(X_train)} (做多:{(y_train==1).sum()}, 做空:{(y_train==0).sum()})")
                
                # 訓練模型
                trainer = BBReversalModelTrainer(model_dir='models/saved')
                
                # 手動分割
                from sklearn.model_selection import train_test_split
                X_t, X_v, y_t, y_v = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
                )
                
                trainer.model = trainer.model or __import__('lightgbm').LGBMClassifier(**trainer.lgb_params)
                trainer.model.fit(
                    X_t, y_t,
                    eval_set=[(X_v, y_v)],
                    callbacks=[__import__('lightgbm').early_stopping(stopping_rounds=50, verbose=False)]
                )
                
                # 訓練集準確率
                from sklearn.metrics import accuracy_score
                y_pred_train = trainer.model.predict(X_v)
                train_accuracy = accuracy_score(y_v, y_pred_train)
                
                st.success(f"訓練集準確率: {train_accuracy:.2%}")
                
                # ====== OOS測試階段 ======
                st.subheader("階段 2: OOS測試集驗證")
                
                extractor_oos = BBReversalFeatureExtractor(
                    bb_period=bb_period_train,
                    bb_std=bb_std_train,
                    rsi_period=14
                )
                
                extractor_oos.detector = BBReversalDetector(
                    bb_period=bb_period_train,
                    bb_std=bb_std_train,
                    touch_threshold=touch_threshold_train,
                    reversal_confirm_candles=5,
                    min_reversal_pct=min_reversal_train,
                    trend_filter_enabled=True,
                    trend_lookback=10,
                    require_middle_return=True
                )
                
                df_oos_processed = extractor_oos.process(df_oos, create_labels=True)
                oos_stats = extractor_oos.get_reversal_statistics()
                
                st.info(f"OOS反轉點: {oos_stats['total_reversals']} (上:{oos_stats['upper_reversals']}, 下:{oos_stats['lower_reversals']})")
                
                if oos_stats['total_reversals'] < 10:
                    st.warning(f"OOS反轉點太少: {oos_stats['total_reversals']}")
                
                X_oos, y_oos = extractor_oos.get_training_data(df_oos_processed)
                
                # OOS預測
                y_pred_oos = trainer.model.predict(X_oos)
                oos_accuracy = accuracy_score(y_oos, y_pred_oos)
                
                # ====== 結果展示 ======
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("訓練集準確率", f"{train_accuracy:.2%}")
                with col2:
                    st.metric("OOS準確率", f"{oos_accuracy:.2%}")
                with col3:
                    gap = train_accuracy - oos_accuracy
                    st.metric("泛化差距", f"{gap:.2%}")
                
                # 保存模型
                trainer.save_model(prefix=f"{train_symbol}_oos")
                st.success(f"模型已保存: models/saved/{train_symbol}_oos_bb_reversal_lgb.pkl")
                
                if oos_accuracy >= 0.60 and gap < 0.15:
                    st.balloons()
                    st.success("模型表現優異且泛化良好! 可以進行回測")
                
            except Exception as e:
                st.error(f"訓練失敗: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

with tabs[2]:
    st.header("BB模型回測")
    
    st.info("""
    **回測邏輯**:
    1. 載入訓練好的模型 ({SYMBOL}_oos_bb_reversal_lgb.pkl)
    2. 檢測BB觸碰 (上軌/下軌)
    3. 模型預測信號 (0=做空, 1=做多)
    4. 規則過濾:
       - 觸碰上軌 + 預測做空 -> 進場做空
       - 觸碰下軌 + 預測做多 -> 進場做多
    5. 下一根K線開盤價進場
    6. 動態ATR止盈止損
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        backtest_symbol = st.selectbox("回測幣種", ["BTCUSDT", "ETHUSDT", "SOLUSDT"], key="bt_symbol")
        backtest_days = st.slider("回測天數", 30, 365, 90, key="bt_days")
    
    with col2:
        st.subheader("風險管理")
        initial_capital = st.number_input("初始資金", 100, 10000, 1000, key="bt_capital")
        leverage = st.number_input("槓桿倍數", 1, 50, 5, key="bt_leverage")
        tp_mult = st.number_input("ATR止盈倍數", 1.0, 5.0, 2.0, key="bt_tp")
        sl_mult = st.number_input("ATR止損倍數", 1.0, 5.0, 1.5, key="bt_sl")
    
    if st.button("開始回測", type="primary"):
        model_path = f"models/saved/{backtest_symbol}_oos_bb_reversal_lgb.pkl"
        if not os.path.exists(model_path):
            st.error(f"找不到模型文件: {model_path}，請先進行訓練")
        else:
            with st.spinner(f"正在回測 {backtest_symbol}..."):
                try:
                    # 載入數據
                    if isinstance(loader, BinanceDataLoader):
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=backtest_days)
                        df = loader.load_historical_data(backtest_symbol, '15m', start_date, end_date)
                    else:
                        df = loader.load_klines(backtest_symbol, '15m')
                        df = df.tail(backtest_days * 96)
                    
                    # 生成信號
                    generator = BBReversalSignalGenerator(
                        model_path=model_path,
                        bb_period=bb_period_train,  # 使用OOS頁面設定的參數
                        bb_std=bb_std_train,
                        touch_threshold=touch_threshold_train
                    )
                    
                    df_signals = generator.generate_signals(df)
                    
                    # 執行回測
                    engine = BacktestEngine(
                        initial_capital=initial_capital,
                        leverage=leverage,
                        tp_atr_mult=tp_mult,
                        sl_atr_mult=sl_mult
                    )
                    
                    signals_dict = {backtest_symbol: df_signals}
                    metrics = engine.run_backtest(signals_dict)
                    
                    st.subheader("回測結果")
                    display_metrics(metrics)
                    
                    # 繪製權益曲線
                    fig = engine.plot_equity_curve()
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 交易明細
                    trades_df = engine.get_trades_dataframe()
                    if not trades_df.empty:
                        st.subheader("交易明細")
                        st.dataframe(trades_df, use_container_width=True)
                    else:
                        st.warning("沒有產生任何交易")
                        
                except Exception as e:
                    st.error(f"回測錯誤: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())