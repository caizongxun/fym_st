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
from backtesting.engine import BacktestEngine

st.set_page_config(page_title="AI 加密貨幣交易儀表板", layout="wide")
st.title("AI 加密貨幣交易儀表板 - BB策略 v7")

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
**BB策略 v7 - 新增**

策略1: BB反彈 (原有)
- BB上軌/下軌反彈預測
- ADX趨勢過濾

策略2: 三重確認 (新)
- BB + RSI + MACD
- 三個指標同時確認
- 預期勝率 70-75%
- 預期報酬 +100%
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

def display_metrics(metrics):
    """顯示績效指標 (安全處理缺失值)"""
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

tabs = st.tabs(["BB模型訓練", "BB反彈回測", "三重確認回測", "策略對比", "參數優化"])

with tabs[0]:
    st.header("BB反彈模型訓練")
    
    st.info("""
    **訓練流程**:
    1. **單幣種訓練**: 訓練特定幣種的BB模型
    2. **批量訓練**: 一鍵訓練多個幣種的模型
    3. **全部訓練**: 訓練所有38個幣種的模型
    
    訓練後的模型會保存到 `models/saved/{SYMBOL}_bb_*.pkl`
    
    **推薦K棒數量** (15分鐘):
    - 快速測試: 5000根 (約52天)
    - 標準訓練: 10000根 (約104天)
    - 完整訓練: 15000根 (約156天)
    - 超大資料: 20000根 (約208天)
    
    **注意**: 三重確認策略不需要訓練模型,可直接使用!
    """)
    
    train_mode = st.radio("訓練模式", ["單幣種訓練", "批量訓練", "全部訓練(38幣)"], horizontal=True)
    
    if train_mode == "單幣種訓練":
        col1, col2 = st.columns(2)
        with col1:
            symbols = symbol_selector("train_single", multi=False)
            symbol = symbols[0]
            n_candles = st.number_input(
                "訓練K棒數量",
                min_value=1000,
                max_value=50000,
                value=10000,
                step=1000,
                key="train_candles",
                help="15分鐘: 10000根約104天"
            )
        
        with col2:
            bb_period = st.number_input("BB週期", min_value=10, max_value=30, value=20)
            bb_std = st.number_input("BB標準差", min_value=1.0, max_value=3.0, value=2.0, step=0.5)
        
        st.caption(f"預估訓練時間: 約1-3分鐘 | 數據量: {n_candles}根K棒")
        
        if st.button("開始訓練", key="train_btn", type="primary"):
            with st.spinner(f"正在訓練 {symbol}..."):
                df = loader.load_klines(symbol, '15m')
                df = df.tail(n_candles)
                
                extractor = BBBounceFeatureExtractor(bb_period=bb_period, bb_std=bb_std)
                df_processed = extractor.process(df, create_labels=True)
                
                trainer = BBBounceModelTrainer(model_dir='models/saved')
                trainer.train_both_models(df_processed)
                trainer.save_models(prefix=symbol)
                
                st.success(f"{symbol} BB模型訓練完成!")
                st.info(f"模型保存至: `models/saved/{symbol}_bb_*.pkl`")
                st.write(f"實際使用數據: {len(df_processed)}根K棒")
    
    elif train_mode == "批量訓練":
        st.subheader("批量訓練多幣種模型")
        
        symbols = symbol_selector("train_batch", multi=True)
        batch_candles = st.number_input(
            "訓練K棒數量",
            min_value=1000,
            max_value=50000,
            value=10000,
            step=1000,
            key="batch_candles",
            help="15分鐘: 10000根約104天"
        )
        
        if symbols:
            st.caption(f"預估總時間: 約{len(symbols) * 2}-{len(symbols) * 4}分鐘 | 每幣{batch_candles}根K棒")
        
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
                        df = loader.load_klines(symbol, '15m')
                        df = df.tail(batch_candles)
                        
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
    
    else:  # 全部訓練
        st.subheader("全部訓練 - 38個幣種")
        
        all_symbols = HuggingFaceKlineLoader.get_supported_symbols()
        
        st.info(f"將訓練所有38個幣種: {', '.join(all_symbols[:5])}...")
        
        all_candles = st.number_input(
            "訓練K棒數量",
            min_value=1000,
            max_value=50000,
            value=10000,
            step=1000,
            key="all_candles",
            help="15分鐘: 10000根約104天"
        )
        
        st.warning(f"預估總時間: 約76-152分鐘 (1.3-2.5小時) | 總計{len(all_symbols)}個幣種")
        
        if st.button("開始訓練所有38個幣種", key="all_train_btn", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []
            
            start_time = datetime.now()
            
            for idx, symbol in enumerate(all_symbols):
                elapsed = (datetime.now() - start_time).total_seconds() / 60
                avg_time = elapsed / max(idx, 1)
                remaining = avg_time * (len(all_symbols) - idx)
                
                status_text.text(
                    f"正在訓練 {symbol} ({idx+1}/{len(all_symbols)}) | "
                    f"已耗時: {elapsed:.1f}分 | 預估剩餘: {remaining:.1f}分"
                )
                progress_bar.progress((idx + 1) / len(all_symbols))
                
                try:
                    df = loader.load_klines(symbol, '15m')
                    df = df.tail(all_candles)
                    
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
            
            total_time = (datetime.now() - start_time).total_seconds() / 60
            
            st.success(f"全部訓練完成! 總耗時: {total_time:.1f}分鐘")
            
            results_df = pd.DataFrame(results)
            success_count = (results_df['狀態'] == '成功').sum()
            st.metric("成功訓練", f"{success_count}/{len(all_symbols)}")
            
            st.dataframe(results_df, use_container_width=True)

with tabs[1]:
    st.header("BB反彈策略回測 (原有策略)")
    
    st.info("""
    **BB反彈策略**:
    - 使用機器學習預測BB上下軌反彈
    - ADX趨勢過濾
    - 適合震盪市場
    - 需要先訓練模型
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        symbols = symbol_selector("bb_backtest", multi=True, default_symbols=['BTCUSDT', 'ETHUSDT'])
        bt_days = st.number_input("回測天數", min_value=7, max_value=365, value=30, key="bb_bt_days")
        initial_capital = st.number_input("總資金 (USDT)", min_value=10.0, value=100.0, key="bb_capital")
    
    with col2:
        max_positions = st.number_input(
            "最大同時持倉數",
            min_value=1,
            max_value=10,
            value=2,
            key="bb_max_pos"
        )
        position_size_pct = st.slider(
            "單筆倉位 (%)",
            min_value=10,
            max_value=100,
            value=50,
            step=10,
            key="bb_pos_size"
        ) / 100
        leverage = st.number_input("槓桿倍數", min_value=1, max_value=20, value=10, key="bb_leverage")
    
    col3, col4 = st.columns(2)
    with col3:
        tp_atr_mult = st.number_input("止盈 ATR倍數", min_value=0.5, max_value=5.0, value=2.0, step=0.5, key="bb_tp")
        bb_threshold = st.slider("BB反彈閾值 (%)", min_value=50, max_value=90, value=60, step=5, key="bb_th") / 100
    
    with col4:
        sl_atr_mult = st.number_input("止損 ATR倍數", min_value=0.5, max_value=3.0, value=1.5, step=0.5, key="bb_sl")
        adx_threshold = st.number_input("ADX強趨勢閾值", min_value=20, max_value=40, value=30, key="bb_adx_th")
    
    if st.button("執行BB反彈回測", key="bb_bt_btn", type="primary"):
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
                display_metrics(metrics)
                
                if metrics.get('total_trades', 0) > 0:
                    st.plotly_chart(engine.plot_equity_curve(), use_container_width=True)
                else:
                    st.warning("無交易產生,請調整參數")

with tabs[2]:
    st.header("BB+RSI+MACD 三重確認策略回測 (新策略)")
    
    st.success("""
    **三重確認策略特點**:
    - 無需訓練模型,直接可用!
    - 做多: BB下軌 + RSI<30回升 + MACD金叉 + ADX<30
    - 做空: BB上軌 + RSI>70下降 + MACD死叉 + ADX<30
    - 預期勝率提升到 70-75%
    - 預期報酬提升 100%
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        tri_symbols = symbol_selector("tri_backtest", multi=True, default_symbols=['BTCUSDT', 'ETHUSDT'])
        tri_bt_days = st.number_input("回測天數", min_value=7, max_value=365, value=30, key="tri_bt_days")
        tri_capital = st.number_input("總資金 (USDT)", min_value=10.0, value=100.0, key="tri_capital")
    
    with col2:
        tri_max_pos = st.number_input(
            "最大同時持倉數",
            min_value=1,
            max_value=10,
            value=2,
            key="tri_max_pos"
        )
        tri_pos_size = st.slider(
            "單筆倉位 (%)",
            min_value=10,
            max_value=100,
            value=50,
            step=10,
            key="tri_pos_size"
        ) / 100
        tri_leverage = st.number_input("槓桿倍數", min_value=1, max_value=20, value=10, key="tri_leverage")
    
    col3, col4 = st.columns(2)
    with col3:
        tri_tp = st.number_input("止盈 ATR倍數", min_value=0.5, max_value=5.0, value=2.0, step=0.5, key="tri_tp")
        tri_rsi_oversold = st.slider("RSI超賣閾值", min_value=20, max_value=40, value=30, step=5, key="tri_rsi_os")
    
    with col4:
        tri_sl = st.number_input("止損 ATR倍數", min_value=0.5, max_value=3.0, value=1.5, step=0.5, key="tri_sl")
        tri_rsi_overbought = st.slider("RSI超買閾值", min_value=60, max_value=80, value=70, step=5, key="tri_rsi_ob")
    
    if st.button("執行三重確認回測", key="tri_bt_btn", type="primary"):
        if not tri_symbols:
            st.error("請選擇至少一個幣種!")
        else:
            with st.spinner("載入數據並生成三重確認信號..."):
                end_date = datetime.now()
                start_date = end_date - timedelta(days=tri_bt_days)
                
                signals_dict = {}
                
                for symbol in tri_symbols:
                    try:
                        df = loader.load_historical_data(symbol, '15m', start_date, end_date)
                        
                        signal_gen = TripleConfirmSignalGenerator(
                            bb_period=20,
                            bb_std=2.0,
                            rsi_period=14,
                            rsi_oversold=tri_rsi_oversold,
                            rsi_overbought=tri_rsi_overbought,
                            adx_threshold=30
                        )
                        
                        df_signals = signal_gen.generate_signals(df)
                        
                        if 'open_time' not in df_signals.columns:
                            df_signals['open_time'] = df_signals.index
                        df_signals['open_time'] = pd.to_datetime(df_signals['open_time'])
                        df_signals['15m_atr'] = calculate_atr(df_signals)
                        
                        signals_dict[symbol] = df_signals
                        
                        summary = signal_gen.get_signal_summary(df_signals)
                        st.info(f"{symbol}: {summary['total_signals']}個信號 (多:{summary['long_signals']}, 空:{summary['short_signals']})")
                        
                    except Exception as e:
                        st.warning(f"{symbol} 載入失敗: {str(e)}")
                
                if len(signals_dict) == 0:
                    st.error("沒有成功載入任何幣種!")
                    st.stop()
                
                st.success(f"成功載入 {len(signals_dict)} 個幣種")
            
            with st.spinner("執行回測..."):
                engine = BacktestEngine(
                    initial_capital=tri_capital,
                    leverage=tri_leverage,
                    tp_atr_mult=tri_tp,
                    sl_atr_mult=tri_sl,
                    position_size_pct=tri_pos_size,
                    position_mode='fixed',
                    max_positions=tri_max_pos,
                    debug=False
                )
                
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
                    st.warning("無交易產生,請調整參數")

with tabs[3]:
    st.header("策略對比分析")
    st.info("""
    **對比說明**:
    使用相同的回測參數,比較兩個策略的表現差異
    - 策略1: BB反彈 (機器學習)
    - 策略2: BB+RSI+MACD三重確認
    
    **使用方法**: 先在Tab2和Tab3分別執行回測,然後在這裡查看對比結果
    """)
    
    st.warning("此功能正在開發中,請先分別在Tab2和Tab3執行回測查看結果...")

with tabs[4]:
    st.header("參數優化")
    st.info("此功能正在開發中...")