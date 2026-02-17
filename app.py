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
from backtesting.engine import BacktestEngine

st.set_page_config(page_title="AI 加密貨幣交易儀表板", layout="wide")
st.title("AI 加密貨幣交易儀表板 - BB反彈策略 v6")

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
**BB反彈策略 v6**

核心概念:
- BB上軌/下軌反彈預測
- ADX趨勢過濾
- 雙重確認機制

適合市場:
- 震盪市、弱趨勢市
- 均值回歸特性強的幣種
""")

def calculate_atr(df_signals):
    """Calculate ATR using True Range method"""
    high_low = df_signals['high'] - df_signals['low']
    high_close = abs(df_signals['high'] - df_signals['close'].shift(1))
    low_close = abs(df_signals['low'] - df_signals['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=14).mean()
    atr = atr.bfill().fillna(df_signals['close'] * 0.02)
    return atr

def symbol_selector(key_prefix: str, multi: bool = False, default_symbols: list = None):
    """幣種選擇器"""
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

tabs = st.tabs(["BB模型訓練", "多幣種回測", "參數優化", "Walk-Forward"])

with tabs[0]:
    st.header("BB反彈模型訓練")
    
    st.info("""
    **訓練流程**:
    1. **單幣種訓練**: 訓練特定幣種的BB模型
    2. **批量訓練**: 一鍵訓練多個幣種的模型
    
    訓練後的模型會保存到 `models/saved/{SYMBOL}_bb_*.pkl`
    """)
    
    train_mode = st.radio("訓練模式", ["單幣種訓練", "批量訓練"], horizontal=True)
    
    if train_mode == "單幣種訓練":
        col1, col2 = st.columns(2)
        with col1:
            symbols = symbol_selector("train_single", multi=False)
            symbol = symbols[0]
            days = st.number_input("訓練天數", min_value=30, max_value=365, value=60, key="train_days")
        
        with col2:
            bb_period = st.number_input("BB週期", min_value=10, max_value=30, value=20)
            bb_std = st.number_input("BB標準差", min_value=1.0, max_value=3.0, value=2.0, step=0.5)
        
        if st.button("開始訓練", key="train_btn", type="primary"):
            with st.spinner(f"正在訓練 {symbol}..."):
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                df = loader.load_historical_data(symbol, '15m', start_date, end_date)
                
                extractor = BBBounceFeatureExtractor(bb_period=bb_period, bb_std=bb_std)
                df_processed = extractor.process(df, create_labels=True)
                
                trainer = BBBounceModelTrainer(model_dir='models/saved')
                trainer.train_both_models(df_processed)
                trainer.save_models(prefix=symbol)
                
                st.success(f"{symbol} BB模型訓練完成!")
                st.info(f"模型保存至: `models/saved/{symbol}_bb_*.pkl`")
    
    else:
        st.subheader("批量訓練多幣種模型")
        
        symbols = symbol_selector("train_batch", multi=True)
        batch_days = st.number_input("訓練天數", min_value=30, max_value=365, value=60, key="batch_days")
        
        if st.button("一鍵訓練所有幣種", key="batch_train_btn", type="primary"):
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
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=batch_days)
                        df = loader.load_historical_data(symbol, '15m', start_date, end_date)
                        
                        extractor = BBBounceFeatureExtractor(bb_period=20, bb_std=2.0)
                        df_processed = extractor.process(df, create_labels=True)
                        
                        trainer = BBBounceModelTrainer(model_dir='models/saved')
                        trainer.train_both_models(df_processed)
                        trainer.save_models(prefix=symbol)
                        
                        results.append({'幣種': symbol, '狀態': '成功', '數據量': len(df)})
                    except Exception as e:
                        results.append({'幣種': symbol, '狀態': f'失敗: {str(e)[:30]}', '數據量': 0})
                
                progress_bar.empty()
                status_text.empty()
                
                st.success("批量訓練完成!")
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)

with tabs[1]:
    st.header("多幣種BB反彈策略回測")
    
    st.info("""
    **多幣種交易說明**:
    - 總資金分配到多個幣種
    - 每個幣種獨立產生信號
    - 可設置最大同時持倉數
    - 資金動態管理
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        symbols = symbol_selector("backtest", multi=True, default_symbols=['BTCUSDT', 'ETHUSDT'])
        bt_days = st.number_input("回測天數", min_value=7, max_value=365, value=30, key="bt_days")
        initial_capital = st.number_input("總資金 (USDT)", min_value=10.0, value=100.0, key="capital")
    
    with col2:
        max_positions = st.number_input(
            "最大同時持倉數",
            min_value=1,
            max_value=10,
            value=2,
            key="max_pos"
        )
        position_size_pct = st.slider(
            "單筆倉位 (%)",
            min_value=10,
            max_value=100,
            value=50,
            step=10,
            key="pos_size"
        ) / 100
        leverage = st.number_input("槓桿倍數", min_value=1, max_value=20, value=10, key="leverage")
    
    col3, col4 = st.columns(2)
    with col3:
        tp_atr_mult = st.number_input("止盈 ATR倍數", min_value=0.5, max_value=5.0, value=2.0, step=0.5, key="tp")
        bb_threshold = st.slider("BB反彈閾值 (%)", min_value=50, max_value=90, value=60, step=5, key="bb_th") / 100
    
    with col4:
        sl_atr_mult = st.number_input("止損 ATR倍數", min_value=0.5, max_value=3.0, value=1.5, step=0.5, key="sl")
        adx_threshold = st.number_input("ADX強趨勢閾值", min_value=20, max_value=40, value=30, key="adx_th")
    
    if st.button("執行多幣種回測", key="bt_btn", type="primary"):
        if not symbols:
            st.error("請選擇至少一個幣種!")
        else:
            with st.spinner("載入數據並生成信號..."):
                end_date = datetime.now()
                start_date = end_date - timedelta(days=bt_days)
                
                signals_dict = {}
                
                for symbol in symbols:
                    try:
                        df = loader.load_historical_data(symbol, '15m', start_date, end_date)
                        
                        signal_gen = BBBounceSignalGenerator(
                            bb_model_dir='models/saved',
                            bb_bounce_threshold=bb_threshold,
                            adx_strong_trend_threshold=adx_threshold,
                            model_prefix=symbol
                        )
                        
                        df_signals = signal_gen.generate_signals(df)
                        
                        if 'open_time' not in df_signals.columns:
                            df_signals['open_time'] = df_signals.index
                        df_signals['open_time'] = pd.to_datetime(df_signals['open_time'])
                        df_signals['15m_atr'] = calculate_atr(df_signals)
                        
                        signals_dict[symbol] = df_signals
                        
                    except Exception as e:
                        st.warning(f"{symbol} 載入失敗: {str(e)}")
                
                if len(signals_dict) == 0:
                    st.error("沒有成功載入任何幣種!")
                    st.stop()
                
                st.success(f"成功載入 {len(signals_dict)} 個幣種")
            
            with st.spinner("執行回測..."):
                engine = BacktestEngine(
                    initial_capital=initial_capital,
                    leverage=leverage,
                    tp_atr_mult=tp_atr_mult,
                    sl_atr_mult=sl_atr_mult,
                    position_size_pct=position_size_pct,
                    position_mode='fixed',
                    max_positions=max_positions,
                    debug=False
                )
                
                metrics = engine.run_backtest(signals_dict)
                
                st.subheader("績效指標")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("交易次數", metrics['total_trades'])
                    st.metric("勝率", f"{metrics['win_rate']:.2f}%")
                with col2:
                    st.metric("最終權益", f"${metrics['final_equity']:.2f}")
                    st.metric("總回報", f"{metrics['total_return_pct']:.2f}%")
                with col3:
                    st.metric("獲利因子", f"{metrics['profit_factor']:.2f}")
                    st.metric("夏普比率", f"{metrics['sharpe_ratio']:.2f}")
                with col4:
                    st.metric("最大回撤", f"{metrics['max_drawdown_pct']:.2f}%")
                    st.metric("平均持倉(分)", f"{metrics['avg_duration_min']:.0f}")
                
                if 'trades_per_symbol' in metrics and metrics['trades_per_symbol']:
                    st.subheader("各幣種交易統計")
                    symbol_stats = pd.DataFrame([
                        {'幣種': k, '交易數': v}
                        for k, v in metrics['trades_per_symbol'].items()
                    ]).sort_values('交易數', ascending=False)
                    st.dataframe(symbol_stats, use_container_width=True)
                
                if metrics['total_trades'] > 0:
                    st.plotly_chart(engine.plot_equity_curve(), use_container_width=True)
                    
                    trades_df = engine.get_trades_dataframe()
                    st.subheader("交易明細")
                    display_cols = ['symbol', '進場時間', '離場時間', '方向', '進場價格', '離場價格', 
                                   '損益(USDT)', '損益率', '離場原因', '持倉時長(分)']
                    st.dataframe(trades_df[display_cols], use_container_width=True)
                    
                    st.subheader("離場原因分布")
                    exit_reasons = trades_df['離場原因'].value_counts()
                    col1, col2 = st.columns(2)
                    with col1:
                        st.bar_chart(exit_reasons)
                    with col2:
                        st.dataframe(exit_reasons, use_container_width=True)
                else:
                    st.warning("無交易產生,請調整參數")

with tabs[2]:
    st.header("參數優化")
    st.info("此功能正在開發中...")

with tabs[3]:
    st.header("Walk-Forward測試")
    st.info("此功能正在開發中...")