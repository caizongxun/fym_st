import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from io import StringIO

from data.binance_loader import BinanceDataLoader
from utils.bb_bounce_features import BBBounceFeatureExtractor
from models.train_bb_bounce_model import BBBounceModelTrainer
from utils.signal_generator_bb import BBBounceSignalGenerator
from backtesting.engine import BacktestEngine

st.set_page_config(page_title="AI 加密貨幣交易儀表板", layout="wide")
st.title("AI 加密貨幣交易儀表板 - BB反彈策略 v6")

st.sidebar.info("""
**BB反彈策略 v6**

核心概念:
- 在BB上軌/下軌觸碰點預測反彈
- ADX趨勢過濾
- 雙重確認: BB模型 + 反轉模型

適合市場:
- 震盪市、弱趨勢市
- 均值回歸特性強的幣種
""")

# 共用函數: 計算ATR
def calculate_atr(df_signals):
    """Calculate ATR using True Range method"""
    high_low = df_signals['high'] - df_signals['low']
    high_close = abs(df_signals['high'] - df_signals['close'].shift(1))
    low_close = abs(df_signals['low'] - df_signals['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=14).mean()
    atr = atr.bfill().fillna(df_signals['close'] * 0.02)
    return atr

tabs = st.tabs(["BB模型訓練", "多幣種回測", "參數優化", "Walk-Forward測試"])

# ============ TAB 1: 模型訓練 ============
with tabs[0]:
    st.header("BB反彈模型訓練")
    
    st.info("""
    **訓練流程**:
    1. 單幣種訓練: 訓練特定幣種的BB模型
    2. 批量訓練: 一鍵訓練多個幣種的模型
    
    訓練後的模型會保存到 `models/saved/{SYMBOL}_bb_*.pkl`
    """)
    
    train_mode = st.radio("訓練模式", ["單幣種訓練", "批量訓練"], horizontal=True)
    
    if train_mode == "單幣種訓練":
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.text_input("交易對", value="BTCUSDT", key="bb_train_symbol")
            days = st.number_input("訓練天數", min_value=30, max_value=180, value=60, key="bb_train_days")
        
        with col2:
            bb_period = st.number_input("BB週期", min_value=10, max_value=30, value=20)
            bb_std = st.number_input("BB標準差", min_value=1.0, max_value=3.0, value=2.0, step=0.5)
        
        if st.button("開始訓練BB模型", key="bb_train_btn"):
            with st.spinner("訓練中..."):
                loader = BinanceDataLoader()
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                df = loader.load_historical_data(symbol, '15m', start_date, end_date)
                
                extractor = BBBounceFeatureExtractor(bb_period=bb_period, bb_std=bb_std)
                df_processed = extractor.process(df, create_labels=True)
                
                trainer = BBBounceModelTrainer(model_dir=f'models/saved')
                trainer.train_both_models(df_processed)
                trainer.save_models(prefix=symbol)
                
                st.success(f"✅ {symbol} BB模型訓練完成!")
                st.write(f"模型保存至: `models/saved/{symbol}_bb_upper_bounce_model.pkl`")
    
    else:  # 批量訓練
        st.subheader("批量訓練多幣種模型")
        
        default_symbols = "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,ADAUSDT"
        symbols_input = st.text_area(
            "幣種列表 (逗號分隔)",
            value=default_symbols,
            help="輸入多個幣種,用英文逗號分隔"
        )
        
        batch_days = st.number_input("訓練天數", min_value=30, max_value=180, value=60, key="batch_days")
        
        if st.button("🚀 一鍵訓練所有幣種", key="batch_train_btn"):
            symbols_list = [s.strip().upper() for s in symbols_input.split(',')]
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []
            
            for idx, symbol in enumerate(symbols_list):
                status_text.text(f"正在訓練 {symbol} ({idx+1}/{len(symbols_list)})...")
                progress_bar.progress((idx + 1) / len(symbols_list))
                
                try:
                    loader = BinanceDataLoader()
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=batch_days)
                    df = loader.load_historical_data(symbol, '15m', start_date, end_date)
                    
                    extractor = BBBounceFeatureExtractor(bb_period=20, bb_std=2.0)
                    df_processed = extractor.process(df, create_labels=True)
                    
                    trainer = BBBounceModelTrainer(model_dir='models/saved')
                    trainer.train_both_models(df_processed)
                    trainer.save_models(prefix=symbol)
                    
                    results.append({'幣種': symbol, '狀態': '✅ 成功', '數據量': len(df)})
                except Exception as e:
                    results.append({'幣種': symbol, '狀態': f'❌ 失敗: {str(e)[:30]}', '數據量': 0})
            
            progress_bar.empty()
            status_text.empty()
            
            st.success("批量訓練完成!")
            st.dataframe(pd.DataFrame(results))

# ============ TAB 2: 多幣種回測 ============
with tabs[1]:
    st.header("多幣種BB反彈策略回測")
    
    st.info("""
    **多幣種交易說明**:
    - 總資金會分配到多個幣種
    - 每個幣種獨立產生信號
    - 可設置最大同時持倉數量
    - 資金動態管理,優先執行高信號質量的交易
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        symbols_input = st.text_area(
            "交易幣種 (逗號分隔)",
            value="BTCUSDT,ETHUSDT",
            key="bt_symbols",
            help="支持多幣種同時回測"
        )
        bt_days = st.number_input("回測天數", min_value=7, max_value=90, value=30, key="bb_bt_days")
        initial_capital = st.number_input("總資金 (USDT)", min_value=10.0, value=100.0, key="bb_capital")
    
    with col2:
        max_positions = st.number_input(
            "最大同時持倉數",
            min_value=1,
            max_value=10,
            value=2,
            key="max_positions",
            help="限制同時開倉的最大數量"
        )
        position_size_pct = st.slider(
            "單筆倉位 (%)",
            min_value=10,
            max_value=100,
            value=50,
            step=10,
            key="bb_position",
            help="每筆交易使用總資金的百分比"
        ) / 100
        
        leverage = st.number_input("槓桿倍數", min_value=1, max_value=20, value=10, key="leverage")
    
    col3, col4 = st.columns(2)
    with col3:
        tp_atr_mult = st.number_input("止盈 ATR倍數", min_value=0.5, max_value=5.0, value=2.0, step=0.5, key="bb_tp")
        bb_threshold = st.slider("BB反彈閾值 (%)", min_value=50, max_value=90, value=60, step=5, key="bb_threshold") / 100
    
    with col4:
        sl_atr_mult = st.number_input("止損 ATR倍數", min_value=0.5, max_value=3.0, value=1.5, step=0.5, key="bb_sl")
        adx_threshold = st.number_input("ADX強趨勢閾值", min_value=20, max_value=40, value=30, key="bb_adx_threshold")
    
    if st.button("執行多幣種回測", key="bb_bt_btn"):
        symbols_list = [s.strip().upper() for s in symbols_input.split(',')]
        
        with st.spinner("載入數據並生成信號..."):
            loader = BinanceDataLoader()
            end_date = datetime.now()
            start_date = end_date - timedelta(days=bt_days)
            
            signals_dict = {}
            
            for symbol in symbols_list:
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
            
            st.write(f"✅ 成功載入 {len(signals_dict)} 個幣種")
        
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
                st.metric("總交易次數", metrics['total_trades'])
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
            
            # 各幣種統計
            if 'trades_per_symbol' in metrics:
                st.subheader("各幣種交易統計")
                symbol_stats = pd.DataFrame([
                    {'幣種': k, '交易數': v}
                    for k, v in metrics['trades_per_symbol'].items()
                ]).sort_values('交易數', ascending=False)
                st.dataframe(symbol_stats)
            
            if metrics['total_trades'] > 0:
                st.plotly_chart(engine.plot_equity_curve(), use_container_width=True)
                
                trades_df = engine.get_trades_dataframe()
                st.subheader("交易明細")
                display_cols = ['symbol', '進場時間', '離場時間', '方向', '進場價格', '離場價格', 
                               '損益(USDT)', '損益率', '離場原因', '持倉時長(分)']
                st.dataframe(trades_df[display_cols])
                
                # 離場原因統計
                st.subheader("離場原因分布")
                exit_reasons = trades_df['離場原因'].value_counts()
                col1, col2 = st.columns(2)
                with col1:
                    st.bar_chart(exit_reasons)
                with col2:
                    st.dataframe(exit_reasons)

# ============ TAB 3: 參數優化 ============
with tabs[2]:
    st.header("參數優化")
    
    st.info("""
    **目標**: 找到最佳參數組合
    
    優化參數:
    1. BB反彈閾值 (50%-70%)
    2. ADX閾值 (25-35)
    3. 止盈/止損倍數
    
    使用訓練期優化,驗證期測試避免過擬合
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        opt_symbol = st.text_input("優化交易對", value="BTCUSDT", key="opt_symbol")
        opt_train_days = st.number_input("訓練期天數", min_value=30, max_value=90, value=45, key="opt_train")
    with col2:
        opt_test_days = st.number_input("驗證期天數", min_value=15, max_value=45, value=15, key="opt_test")
    
    if st.button("開始參數優化", key="opt_btn"):
        with st.spinner("執行參數優化..."):
            loader = BinanceDataLoader()
            end_date = datetime.now()
            train_end = end_date - timedelta(days=opt_test_days)
            train_start = train_end - timedelta(days=opt_train_days)
            
            df_train = loader.load_historical_data(opt_symbol, '15m', train_start, train_end)
            df_test = loader.load_historical_data(opt_symbol, '15m', train_end, end_date)
            
            results = []
            
            bb_thresholds = [0.50, 0.55, 0.60, 0.65]
            adx_thresholds = [25, 30, 35]
            tp_mults = [1.5, 2.0, 2.5]
            sl_mults = [1.0, 1.5]
            
            total_combinations = len(bb_thresholds) * len(adx_thresholds) * len(tp_mults) * len(sl_mults)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            idx = 0
            for bb_th in bb_thresholds:
                for adx_th in adx_thresholds:
                    for tp_mult in tp_mults:
                        for sl_mult in sl_mults:
                            idx += 1
                            status_text.text(f"測試組合 {idx}/{total_combinations}...")
                            progress_bar.progress(idx / total_combinations)
                            
                            signal_gen = BBBounceSignalGenerator(
                                bb_model_dir='models/saved',
                                bb_bounce_threshold=bb_th,
                                adx_strong_trend_threshold=adx_th,
                                model_prefix=opt_symbol
                            )
                            
                            # 訓練期
                            df_train_signals = signal_gen.generate_signals(df_train.copy())
                            if 'open_time' not in df_train_signals.columns:
                                df_train_signals['open_time'] = df_train_signals.index
                            df_train_signals['open_time'] = pd.to_datetime(df_train_signals['open_time'])
                            df_train_signals['15m_atr'] = calculate_atr(df_train_signals)
                            
                            engine_train = BacktestEngine(
                                initial_capital=100.0,
                                leverage=10.0,
                                tp_atr_mult=tp_mult,
                                sl_atr_mult=sl_mult,
                                position_size_pct=1.0,
                                position_mode='fixed',
                                max_positions=1,
                                debug=False
                            )
                            train_metrics = engine_train.run_backtest({opt_symbol: df_train_signals})
                            
                            # 驗證期
                            df_test_signals = signal_gen.generate_signals(df_test.copy())
                            if 'open_time' not in df_test_signals.columns:
                                df_test_signals['open_time'] = df_test_signals.index
                            df_test_signals['open_time'] = pd.to_datetime(df_test_signals['open_time'])
                            df_test_signals['15m_atr'] = calculate_atr(df_test_signals)
                            
                            engine_test = BacktestEngine(
                                initial_capital=100.0,
                                leverage=10.0,
                                tp_atr_mult=tp_mult,
                                sl_atr_mult=sl_mult,
                                position_size_pct=1.0,
                                position_mode='fixed',
                                max_positions=1,
                                debug=False
                            )
                            test_metrics = engine_test.run_backtest({opt_symbol: df_test_signals})
                            
                            results.append({
                                'BB閾值': bb_th,
                                'ADX閾值': adx_th,
                                '止盈倍數': tp_mult,
                                '止損倍數': sl_mult,
                                '訓練_交易數': train_metrics['total_trades'],
                                '訓練_勝率': train_metrics['win_rate'],
                                '訓練_獲利因子': train_metrics['profit_factor'],
                                '訓練_回報': train_metrics['total_return_pct'],
                                '驗證_交易數': test_metrics['total_trades'],
                                '驗證_勝率': test_metrics['win_rate'],
                                '驗證_獲利因子': test_metrics['profit_factor'],
                                '驗證_回報': test_metrics['total_return_pct'],
                            })
            
            progress_bar.empty()
            status_text.empty()
            
            results_df = pd.DataFrame(results)
            
            st.write(f"總共測試了 {len(results_df)} 組參數")
            st.write(f"訓練期交易數 >= 5: {(results_df['訓練_交易數'] >= 5).sum()} 組")
            st.write(f"驗證期交易數 >= 3: {(results_df['驗證_交易數'] >= 3).sum()} 組")
            
            filtered_df = results_df[
                (results_df['訓練_交易數'] >= 5) & 
                (results_df['驗證_交易數'] >= 3) &
                (results_df['驗證_獲利因子'] > 0)
            ]
            
            st.write(f"\n過濾後剩餘: {len(filtered_df)} 組")
            
            if len(filtered_df) == 0:
                st.warning("沒有符合條件的參數組合!")
                st.subheader("所有參數組合 (未過濾)")
                st.dataframe(results_df.sort_values('驗證_獲利因子', ascending=False).head(20).round(2))
            else:
                filtered_df = filtered_df.sort_values('驗證_獲利因子', ascending=False)
                
                st.success("優化完成!")
                st.subheader("Top 10 參數組合 (按驗證期獲利因子排序)")
                st.dataframe(filtered_df.head(10).round(2))
                
                best = filtered_df.iloc[0]
                st.info(f"""
                **推薦參數**:
                - BB反彈閾值: {best['BB閾值']:.0%}
                - ADX閾值: {best['ADX閾值']:.0f}
                - 止盈倍數: {best['止盈倍數']:.1f}
                - 止損倍數: {best['止損倍數']:.1f}
                
                驗證期績效:
                - 獲利因子: {best['驗證_獲利因子']:.2f}
                - 勝率: {best['驗證_勝率']:.1f}%
                - 回報: {best['驗證_回報']:.1f}%
                - 交易數: {int(best['驗證_交易數'])}
                """)

# ============ TAB 4: Walk-Forward測試 ============
with tabs[3]:
    st.header("Walk-Forward測試")
    
    st.info("""
    **Walk-Forward測試**避免過擬合:
    
    將數據分為多個時間窗口:
    1. 在窗口1訓練參數 → 在窗口2測試
    2. 在窗口2訓練參數 → 在窗口3測試
    3. ...
    
    最終評估所有測試窗口的綜合表現
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        wf_symbol = st.text_input("測試交易對", value="BTCUSDT", key="wf_symbol")
        wf_total_days = st.number_input("總測試天數", min_value=60, max_value=180, value=90, key="wf_days")
    with col2:
        wf_window_days = st.number_input("每個窗口天數", min_value=15, max_value=30, value=20, key="wf_window")
    
    if st.button("執行Walk-Forward測試", key="wf_btn"):
        with st.spinner("執行Walk-Forward測試..."):
            loader = BinanceDataLoader()
            end_date = datetime.now()
            start_date = end_date - timedelta(days=wf_total_days)
            
            df_full = loader.load_historical_data(wf_symbol, '15m', start_date, end_date)
            
            window_size = wf_window_days * 96
            n_windows = len(df_full) // window_size
            
            st.write(f"總共 {n_windows} 個窗口,每個窗口約{wf_window_days}天")
            
            all_trades = []
            window_results = []
            
            for i in range(n_windows - 1):
                train_start_idx = i * window_size
                train_end_idx = (i + 1) * window_size
                test_end_idx = min((i + 2) * window_size, len(df_full))
                
                df_train_window = df_full.iloc[train_start_idx:train_end_idx]
                df_test_window = df_full.iloc[train_end_idx:test_end_idx]
                
                signal_gen = BBBounceSignalGenerator(
                    bb_model_dir='models/saved',
                    bb_bounce_threshold=0.60,
                    adx_strong_trend_threshold=30,
                    model_prefix=wf_symbol
                )
                
                df_test_signals = signal_gen.generate_signals(df_test_window.copy())
                if 'open_time' not in df_test_signals.columns:
                    df_test_signals['open_time'] = df_test_signals.index
                df_test_signals['open_time'] = pd.to_datetime(df_test_signals['open_time'])
                df_test_signals['15m_atr'] = calculate_atr(df_test_signals)
                
                engine = BacktestEngine(
                    initial_capital=100.0,
                    leverage=10.0,
                    tp_atr_mult=2.0,
                    sl_atr_mult=1.5,
                    position_size_pct=1.0,
                    position_mode='fixed',
                    max_positions=1,
                    debug=False
                )
                
                metrics = engine.run_backtest({wf_symbol: df_test_signals})
                
                window_results.append({
                    '窗口': i+1,
                    '交易數': metrics['total_trades'],
                    '勝率': metrics['win_rate'],
                    '獲利因子': metrics['profit_factor'],
                    '回報': metrics['total_return_pct']
                })
                
                if metrics['total_trades'] > 0:
                    trades = engine.get_trades_dataframe()
                    all_trades.append(trades)
            
            results_df = pd.DataFrame(window_results)
            st.subheader("各窗口績效")
            st.dataframe(results_df.round(2))
            
            st.subheader("綜合統計")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("平均獲利因子", f"{results_df['獲利因子'].mean():.2f}")
            with col2:
                st.metric("平均勝率", f"{results_df['勝率'].mean():.1f}%")
            with col3:
                st.metric("平均回報", f"{results_df['回報'].mean():.1f}%")
            with col4:
                st.metric("總交易數", f"{results_df['交易數'].sum():.0f}")
            
            # 穩定性評估
            st.subheader("策略穩定性評估")
            profitable_windows = (results_df['獲利因子'] > 1.0).sum()
            total_windows = len(results_df)
            consistency_rate = profitable_windows / total_windows * 100
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("獲利窗口比例", f"{consistency_rate:.1f}%")
                if consistency_rate >= 75:
                    st.success("策略非常穩定!")
                elif consistency_rate >= 60:
                    st.info("策略表現良好")
                else:
                    st.warning("策略穩定性待改善")
            
            with col2:
                pf_std = results_df['獲利因子'].std()
                st.metric("獲利因子標準差", f"{pf_std:.2f}")
                if pf_std < 2.0:
                    st.success("績效波動小")
                else:
                    st.warning("績效波動較大")
            
            if len(all_trades) > 0:
                combined_trades = pd.concat(all_trades, ignore_index=True)
                st.write(f"\n總交易數: {len(combined_trades)}")
                st.dataframe(combined_trades[['進場時間', '方向', '損益(USDT)', '損益率', '離場原因']].head(20))