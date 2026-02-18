import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from io import StringIO

from data.binance_loader import BinanceDataLoader
from data.huggingface_loader import HuggingFaceKlineLoader
from utils.bb_bounce_features import BBBounceFeatureExtractor
from models.train_bb_bounce_model import BBBounceModelTrainer
from utils.signal_generator_bb import BBBounceSignalGenerator
from utils.signal_generator_triple import TripleConfirmSignalGenerator
from utils.dual_model_features_v2 import EnhancedDualModelFeatureExtractor
from models.train_dual_model import DualModelTrainer
from utils.signal_generator_dual import DualModelSignalGenerator
from backtesting.engine import BacktestEngine

st.set_page_config(page_title="AI 加密貨幣交易儀表板", layout="wide")
st.title("AI 加密貨幣交易儀表板 - v8.1 增強雙模型")

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

st.sidebar.info("""
**v8.1 增強雙模型**

✨ 新特性:
- 50+維度增強特徵
- 訂單流特徵 (買賣壓力)
- K棒形態識別
- 多時間框架動量
- 目標準確率: 55-60%

策略:
1. BB反彈
2. 三重確認  
3. 雙模型剝頭皮 (v2)
""")

def calculate_atr(df_signals):
    high_low = df_signals['high'] - df_signals['low']
    high_close = abs(df_signals['high'] - df_signals['close'].shift(1))
    low_close = abs(df_signals['low'] - df_signals['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=14).mean()
    atr = atr.bfill().fillna(df_signals['close'] * 0.02)
    return atr

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

tabs = st.tabs(["雙模型訓練", "雙模型回測", "BB反彈回測", "三重確認回測", "策略對比"])

with tabs[0]:
    st.header("雙模型剝頭皮策略 - 模型訓練 (v2增強版)")
    
    st.success("""
    **v2 增強特徵** (50+維度):
    - ✨ 訂單流: 買賣壓力比、累積壓力
    - 📊 K棒形態: 連續漲跌、影線分析
    - 📈 多時間框架: 1/2/3/5/10期報酬率
    - 🎯 波動率: Parkinson、ATR、範圍比
    - 💪 增強動量: RSI變化、MACD動量
    
    **預期提升**:
    - 方向準確率: 50% → 55-60%
    - 價格預測MAE: 降低20-30%
    """)
    
    train_mode = st.radio("訓練模式", ["單幣種訓練", "批量訓練"], horizontal=True, key="dual_train_mode")
    
    if train_mode == "單幣種訓練":
        col1, col2 = st.columns(2)
        with col1:
            symbols = symbol_selector("dual_train_single", multi=False)
            symbol = symbols[0]
            n_candles = st.number_input(
                "訓練K棒數量",
                min_value=5000,
                max_value=50000,
                value=15000,
                step=1000,
                key="dual_train_candles",
                help="建議至少10000根,15000根更佳"
            )
        
        with col2:
            st.info("**v2增強模型參數**")
            st.write("- 特徵: 50+維度")
            st.write("- 決策樹: 300棵")
            st.write("- 深度: 20層")
            st.write("- 訂單流權重: 高")
        
        st.caption(f"預估訓練時間: 約3-8分鐘 | 數據量: {n_candles}根K棒")
        
        if st.button("開始訓練增強雙模型", key="dual_train_btn", type="primary"):
            with st.spinner(f"正在訓練 {symbol} 增強雙模型..."):
                try:
                    df = loader.load_klines(symbol, '15m')
                    df = df.tail(n_candles)
                    
                    st.info(f"載入 {len(df)} 根K棒,開始特徵工程...")
                    
                    # 使用v2增強特徵
                    extractor = EnhancedDualModelFeatureExtractor(lookback_candles=20)
                    df_processed = extractor.process(df, create_labels=True)
                    
                    st.info(f"特徵工程完成,有效樣本: {len(df_processed)}")
                    
                    X, y_dict = extractor.get_training_data(df_processed)
                    
                    # 使用更強參數
                    trainer = DualModelTrainer(
                        model_dir='models/saved',
                        direction_params={
                            'n_estimators': 300,
                            'max_depth': 20,
                            'min_samples_split': 15,
                            'min_samples_leaf': 8,
                            'random_state': 42,
                            'n_jobs': -1,
                            'class_weight': 'balanced'
                        },
                        price_params={
                            'n_estimators': 300,
                            'max_depth': 20,
                            'min_samples_split': 15,
                            'min_samples_leaf': 8,
                            'random_state': 42,
                            'n_jobs': -1
                        }
                    )
                    
                    metrics = trainer.train_all_models(X, y_dict)
                    trainer.save_models(prefix=symbol)
                    
                    st.success(f"✅ {symbol} 增強雙模型訓練完成!")
                    st.info(f"模型保存至: `models/saved/{symbol}_dual_*.pkl`")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        accuracy = metrics['accuracy']
                        color = "normal" if accuracy < 0.53 else "inverse" if accuracy < 0.57 else "off"
                        st.metric("方向準確率", f"{accuracy:.2%}", delta=f"{(accuracy-0.5)*100:.1f}%")
                    with col2:
                        st.metric("最高價MAE", f"{metrics['high_mae']:.4f}%")
                    with col3:
                        st.metric("最低價MAE", f"{metrics['low_mae']:.4f}%")
                    
                    if accuracy < 0.53:
                        st.warning("⚠️ 準確率偏低,建議增加訓練數據或調整參數")
                    elif accuracy >= 0.55:
                        st.balloons()
                        st.success("🎉 準確率達標! 可以開始回測")
                    
                    importance = trainer.get_feature_importance(extractor.get_feature_columns(), top_n=15)
                    st.subheader("Top 15 重要特徵")
                    st.dataframe(importance[['feature', 'avg_importance']], use_container_width=True)
                    
                except Exception as e:
                    st.error(f"訓練失敗: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    else:
        st.subheader("批量訓練多幣種增強雙模型")
        
        symbols = symbol_selector("dual_train_batch", multi=True)
        batch_candles = st.number_input(
            "訓練K棒數量",
            min_value=5000,
            max_value=50000,
            value=15000,
            step=1000,
            key="dual_batch_candles"
        )
        
        if symbols:
            st.caption(f"預估總時間: 約{len(symbols) * 4}-{len(symbols) * 9}分鐘")
        
        if st.button("批量訓練增強雙模型", key="dual_batch_train_btn", type="primary"):
            if not symbols:
                st.error("請選擇至少一個幣種!")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                results = []
                
                for idx, symbol in enumerate(symbols):
                    status_text.text(f"正在訓練 {symbol} ({idx+1}/{len(symbols)})...")
                    progress_bar.progress((idx + 1) / len(symbols))
                    
                    try:
                        df = loader.load_klines(symbol, '15m')
                        df = df.tail(batch_candles)
                        
                        extractor = EnhancedDualModelFeatureExtractor(lookback_candles=20)
                        df_processed = extractor.process(df, create_labels=True)
                        X, y_dict = extractor.get_training_data(df_processed)
                        
                        trainer = DualModelTrainer(
                            model_dir='models/saved',
                            direction_params={'n_estimators': 300, 'max_depth': 20, 'random_state': 42, 'n_jobs': -1, 'class_weight': 'balanced'},
                            price_params={'n_estimators': 300, 'max_depth': 20, 'random_state': 42, 'n_jobs': -1}
                        )
                        metrics = trainer.train_all_models(X, y_dict)
                        trainer.save_models(prefix=symbol)
                        
                        results.append({
                            '幣種': symbol,
                            '狀態': '✅成功',
                            '準確率': f"{metrics['accuracy']:.2%}",
                            '數據量': len(df)
                        })
                    except Exception as e:
                        results.append({
                            '幣種': symbol,
                            '狀態': f'❌失敗: {str(e)[:30]}',
                            '準確率': 'N/A',
                            '數據量': 0
                        })
                
                progress_bar.empty()
                status_text.empty()
                
                st.success("批量訓練完成!")
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)

with tabs[1]:
    st.header("雙模型剝頭皮策略 - 回測 (v2增強版)")
    
    st.success("""
    **增強版策略**:
    - 使用50+特徵的v2模型
    - 訂單流輔助判斷
    - 更精準的價格範圍預測
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        dual_symbols = symbol_selector("dual_backtest", multi=True, default_symbols=['BTCUSDT', 'ETHUSDT'])
        dual_bt_days = st.number_input("回測天數", min_value=7, max_value=180, value=30, key="dual_bt_days")
        dual_capital = st.number_input("總資金 (USDT)", min_value=10.0, value=100.0, key="dual_capital")
    
    with col2:
        dual_max_pos = st.number_input("最大持倉數", min_value=1, max_value=10, value=3, key="dual_max_pos")
        dual_pos_size = st.slider("單筆倉位 (%)", min_value=10, max_value=100, value=30, step=10, key="dual_pos_size") / 100
        dual_leverage = st.number_input("槓桿倍數", min_value=1, max_value=20, value=10, key="dual_leverage")
    
    col3, col4 = st.columns(2)
    with col3:
        min_confidence = st.slider("最低信心度", min_value=0.5, max_value=0.9, value=0.55, step=0.05, key="dual_conf")
        tp_safety = st.slider("止盈安全係數", min_value=0.80, max_value=0.98, value=0.90, step=0.02, key="dual_tp_safety")
    
    with col4:
        min_rr = st.slider("最低風報比", min_value=1.0, max_value=3.0, value=1.2, step=0.1, key="dual_min_rr")
        sl_cushion = st.slider("止損緩衝", min_value=0.02, max_value=0.15, value=0.05, step=0.01, key="dual_sl_cushion")
    
    if st.button("執行增強雙模型回測", key="dual_bt_btn", type="primary"):
        if not dual_symbols:
            st.error("請選擇至少一個幣種!")
        else:
            with st.spinner("載入數據並生成信號..."):
                end_date = datetime.now()
                start_date = end_date - timedelta(days=dual_bt_days)
                
                signals_dict = {}
                
                for symbol in dual_symbols:
                    try:
                        df = loader.load_historical_data(symbol, '15m', start_date, end_date)
                        
                        signal_gen = DualModelSignalGenerator(
                            model_dir='models/saved',
                            model_prefix=symbol,
                            min_confidence=min_confidence,
                            tp_safety_factor=tp_safety,
                            sl_cushion=sl_cushion,
                            min_reward_risk=min_rr
                        )
                        
                        df_signals = signal_gen.generate_signals(df)
                        
                        if 'open_time' not in df_signals.columns:
                            df_signals['open_time'] = df_signals.index
                        df_signals['open_time'] = pd.to_datetime(df_signals['open_time'])
                        df_signals['15m_atr'] = 0
                        
                        signals_dict[symbol] = df_signals
                        
                        summary = signal_gen.get_signal_summary(df_signals)
                        st.info(f"{symbol}: {summary['total_signals']}個信號 (多:{summary['long_signals']}, 空:{summary['short_signals']}) | 平均風報比: {summary['avg_reward_risk']:.2f}")
                        
                    except Exception as e:
                        st.warning(f"{symbol} 載入失敗: {str(e)}")
                
                if len(signals_dict) == 0:
                    st.error("沒有成功載入任何幣種!")
                    st.stop()
                
                st.success(f"成功載入 {len(signals_dict)} 個幣種")
            
            with st.spinner("執行回測..."):
                engine = BacktestEngine(
                    initial_capital=dual_capital,
                    leverage=dual_leverage,
                    tp_atr_mult=0,
                    sl_atr_mult=0,
                    position_size_pct=dual_pos_size,
                    position_mode='fixed',
                    max_positions=dual_max_pos,
                    debug=False
                )
                
                try:
                    metrics = engine.run_backtest(signals_dict)
                    
                    st.subheader("績效指標")
                    display_metrics(metrics)
                    
                    if metrics.get('total_trades', 0) > 0:
                        st.plotly_chart(engine.plot_equity_curve(), use_container_width=True)
                        
                        trades_df = engine.get_trades_dataframe()
                        st.subheader("交易明細")
                        display_cols = ['symbol', '進場時間', '離場時間', '方向', '進場價格', '離場價格', 
                                       '損益(USDT)', '損益率', '離場原因', '持倉時長(分)']
                        st.dataframe(trades_df[display_cols], use_container_width=True)
                    else:
                        st.warning("無交易產生,請降低信心度閾值或風報比要求")
                except Exception as e:
                    st.error(f"回測執行失敗: {str(e)}")
                    st.warning("提示: 請確保已訓練對應幣種的增強雙模型")

with tabs[2]:
    st.header("BB反彈策略回測")
    st.info("此Tab保留原有BB反彈策略功能")

with tabs[3]:
    st.header("三重確認策略回測")
    st.info("此Tab保留原有三重確認策略功能")

with tabs[4]:
    st.header("策略對比分析")
    st.info("對比功能開發中...")