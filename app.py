import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.binance_loader import BinanceDataLoader
from data.feature_engineer import FeatureEngineer
from training.train_trend import TrendModelTrainer
from training.train_reversal import ReversalModelTrainer
from utils.signal_generator import SignalGenerator
from backtesting.engine import BacktestEngine

# BB Bounce imports
from utils.bb_bounce_features import BBBounceFeatureExtractor
from models.train_bb_bounce_model import BBBounceModelTrainer
from utils.signal_generator_bb import BBBounceSignalGenerator

st.set_page_config(page_title="AI 加密貨幣交易儀表板", layout="wide")

st.title("AI 加密貨幣交易儀表板")

# Strategy selector
strategy = st.sidebar.selectbox(
    "選擇策略",
    options=['反轉策略 (v1-v5)', 'BB反彈策略 (v6)'],
    index=1
)

if strategy == 'BB反彈策略 (v6)':
    st.sidebar.info("""
    **BB反彈策略 v6**
    
    核心概念:
    - 在BB上軌/下軌觸碰點預測反彈
    - ADX趨勢過濾(避免強趨勢中交易)
    - 雙重確認: BB模型 + 反轉模型
    
    適合市場:
    - 震盪市、弱趨勢市
    - 均值回歸特性強的幣種
    """)
    
    tabs = st.tabs(["BB模型訓練", "BB回測", "BB即時分析"])
    
    with tabs[0]:
        st.header("BB反彈模型訓練")
        
        st.info("""
        訓練兩個獨立模型:
        1. **上軌反彈模型**: 預測觸碰上軌後是否下跌(做空機會)
        2. **下軌反彈模型**: 預測觸碰下軌後是否上漲(做多機會)
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            symbol = st.text_input("交易對", value="BTCUSDT", key="bb_train_symbol")
            days = st.number_input("訓練天數", min_value=30, max_value=180, value=60, key="bb_train_days")
        
        with col2:
            bb_period = st.number_input("BB週期", min_value=10, max_value=30, value=20)
            bb_std = st.number_input("BB標準差", min_value=1.0, max_value=3.0, value=2.0, step=0.5)
        
        col3, col4 = st.columns(2)
        with col3:
            adx_period = st.number_input("ADX週期", min_value=7, max_value=21, value=14)
        
        with col4:
            touch_threshold = st.number_input("觸碰閾值(σ)", min_value=0.1, max_value=1.0, value=0.3, step=0.1,
                                            help="距離軌道多少標準差內算觸碰")
        
        if st.button("開始訓練BB模型", key="bb_train_btn"):
            with st.spinner("載入數據中..."):
                loader = BinanceDataLoader()
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                df = loader.load_historical_data(symbol, '15m', start_date, end_date)
                st.write(f"載入 {len(df)} 根K線")
            
            with st.spinner("提取BB特徵..."):
                extractor = BBBounceFeatureExtractor(
                    bb_period=bb_period,
                    bb_std=bb_std,
                    adx_period=adx_period,
                    touch_threshold=touch_threshold
                )
                
                df_processed = extractor.process(df, create_labels=True)
                
                upper_touches = (df_processed['touch_upper'] == 1).sum()
                lower_touches = (df_processed['touch_lower'] == 1).sum()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("觸碰上軌次數", upper_touches)
                with col2:
                    st.metric("觸碰下軌次數", lower_touches)
                
                st.write("趨勢狀態分布:")
                trend_dist = df_processed['trend_state'].value_counts()
                st.bar_chart(trend_dist)
                
                if upper_touches < 30 or lower_touches < 30:
                    st.warning("觸碰樣本數偏少,建議增加訓練天數或放寬touch_threshold")
            
            with st.spinner("訓練XGBoost模型..."):
                trainer = BBBounceModelTrainer(model_dir='models/saved')
                trainer.train_both_models(df_processed)
                trainer.save_models()
                
                st.success("BB模型訓練完成!")
                st.info("""
                模型已保存至:
                - models/saved/bb_upper_bounce_model.pkl (上軌反彈)
                - models/saved/bb_lower_bounce_model.pkl (下軌反彈)
                """)
    
    with tabs[1]:
        st.header("BB反彈策略回測")
        
        st.info("""
        **交易邏輯**:
        
        做空條件 (全部滿足):
        - 觸碰BB上軌
        - BB模型預測反彈機率 > 60%
        - 趨勢不是強多頭 (ADX<30)
        - RSI > 60 (超買)
        
        做多條件 (全部滿足):
        - 觸碰BB下軌
        - BB模型預測反彈機率 > 60%
        - 趨勢不是強空頭 (ADX<30)
        - RSI < 40 (超賣)
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            bt_symbol = st.text_input("回測交易對", value="BTCUSDT", key="bb_bt_symbol")
            bt_days = st.number_input("回測天數", min_value=7, max_value=90, value=30, key="bb_bt_days")
        
        with col2:
            initial_capital = st.number_input("初始資金 (USDT)", min_value=10.0, value=100.0, key="bb_capital")
            position_size_pct = st.slider("倉位大小 (%)", min_value=5, max_value=100, value=100, step=5, key="bb_position") / 100
        
        col3, col4 = st.columns(2)
        with col3:
            tp_atr_mult = st.number_input("止盈 ATR倍數", min_value=0.5, max_value=5.0, value=2.0, step=0.5, key="bb_tp")
        
        with col4:
            sl_atr_mult = st.number_input("止損 ATR倍數", min_value=0.5, max_value=3.0, value=1.5, step=0.5, key="bb_sl")
        
        col5, col6 = st.columns(2)
        with col5:
            bb_threshold = st.slider("BB反彈閾值 (%)", min_value=50, max_value=90, value=60, step=5, key="bb_threshold") / 100
        
        with col6:
            adx_threshold = st.number_input("ADX強趨勢閾值", min_value=20, max_value=40, value=30, key="bb_adx_threshold")
        
        if st.button("執行BB回測", key="bb_bt_btn"):
            with st.spinner("載入數據..."):
                loader = BinanceDataLoader()
                end_date = datetime.now()
                start_date = end_date - timedelta(days=bt_days)
                
                df = loader.load_historical_data(bt_symbol, '15m', start_date, end_date)
                st.write(f"回測期間: {df.index[0]} 至 {df.index[-1]}")
            
            with st.spinner("生成BB信號..."):
                try:
                    signal_gen = BBBounceSignalGenerator(
                        bb_model_dir='models/saved',
                        bb_bounce_threshold=bb_threshold,
                        adx_strong_trend_threshold=adx_threshold
                    )
                    
                    df_signals = signal_gen.generate_signals(df)
                    
                    total_signals = (df_signals['signal'] != 0).sum()
                    long_signals = (df_signals['signal'] == 1).sum()
                    short_signals = (df_signals['signal'] == -1).sum()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("總信號數", total_signals)
                    with col2:
                        st.metric("做多信號", long_signals)
                    with col3:
                        st.metric("做空信號", short_signals)
                    
                    if total_signals == 0:
                        st.warning("未產生任何信號,請降低bb_threshold或增加回測天數")
                        st.stop()
                    
                except FileNotFoundError:
                    st.error("BB模型未找到,請先訓練模型")
                    st.stop()
            
            with st.spinner("執行回測..."):
                # 計算ATR (使用True Range方法)
                high_low = df_signals['high'] - df_signals['low']
                high_close = abs(df_signals['high'] - df_signals['close'].shift(1))
                low_close = abs(df_signals['low'] - df_signals['close'].shift(1))
                
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                df_signals['15m_atr'] = true_range.rolling(window=14).mean()
                
                # 填ATR空值
                df_signals['15m_atr'] = df_signals['15m_atr'].fillna(method='bfill').fillna(df_signals['close'] * 0.02)
                
                # DEBUG: 顯示ATR統計
                st.write("### Debug Info")
                st.write(f"ATR範圍: {df_signals['15m_atr'].min():.2f} - {df_signals['15m_atr'].max():.2f}")
                st.write(f"ATR平均: {df_signals['15m_atr'].mean():.2f}")
                st.write(f"ATR為0的數量: {(df_signals['15m_atr'] == 0).sum()}")
                
                # 顯示有信號的幾筆資料
                signal_rows = df_signals[df_signals['signal'] != 0].head(3)
                st.write("\n前3筆信號:")
                st.dataframe(signal_rows[['open_time', 'signal', '15m_atr', 'close']].round(2))
                
                engine = BacktestEngine(
                    initial_capital=initial_capital,
                    leverage=10.0,
                    tp_atr_mult=tp_atr_mult,
                    sl_atr_mult=sl_atr_mult,
                    position_size_pct=position_size_pct,
                    position_mode='fixed',
                    maker_fee=0.0002,
                    taker_fee=0.0006
                )
                
                signals_dict = {bt_symbol: df_signals}
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
                    pf_color = 'green' if metrics['profit_factor'] > 1.5 else ('orange' if metrics['profit_factor'] > 1.0 else 'red')
                    st.markdown(f":{pf_color}[{'優秀' if metrics['profit_factor'] > 1.5 else ('尚可' if metrics['profit_factor'] > 1.0 else '需優化')}]")
                
                with col4:
                    st.metric("夏普比率", f"{metrics['sharpe_ratio']:.2f}")
                    st.metric("最大回撤", f"{metrics['max_drawdown_pct']:.2f}%")
                
                # 有fallback
                avg_duration = metrics.get('avg_duration_min', 0)
                if avg_duration > 0:
                    st.metric("平均持倉時長", f"{avg_duration:.0f}分鐘")
                
                # 顯示權益曲線
                if metrics['total_trades'] > 0:
                    st.plotly_chart(engine.plot_equity_curve(), use_container_width=True)
                
                    # 離場原因分析
                    trades_df = engine.get_trades_dataframe()
                    
                    st.subheader("離場原因分布")
                    exit_reasons = trades_df['離場原因'].value_counts()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.bar_chart(exit_reasons)
                    
                    with col2:
                        st.write("各原因績效:")
                        for reason in exit_reasons.index:
                            subset = trades_df[trades_df['離場原因'] == reason]
                            win_rate = (subset['pnl'] > 0).sum() / len(subset) * 100
                            avg_pnl = subset['pnl'].mean()
                            st.write(f"{reason}: 勝率{win_rate:.1f}% | 平均{avg_pnl:.2f}U")
                    
                    st.subheader("交易明細")
                    display_cols = [
                        '進場時間', '離場時間', '方向', 
                        '進場價格', '離場價格', '損益(USDT)', '損益率',
                        '離場原因', '持倉時長(分)'
                    ]
                    st.dataframe(trades_df[display_cols])
                    
                    # 下載按鈕
                    csv = trades_df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="下載交易記錄",
                        data=csv,
                        file_name=f'bb_backtest_{bt_symbol}_{datetime.now().strftime("%Y%m%d")}.csv',
                        mime='text/csv'
                    )
                else:
                    st.warning("""
                    ### 無交易產生
                    
                    可能原因:
                    1. ATR值過小或為0,導致TP/SL無效
                    2. 信號時間戳與價格資料不匹配
                    3. 信號未正確傳遞backtest engine
                    
                    請檢查上方Debug Info
                    """)
                
                # 優化建議
                st.subheader("優化建議")
                if metrics['profit_factor'] < 1.0:
                    st.error("""
                    獲利因子 < 1.0,策略虧損
                    
                    建議:
                    - 提高bb_threshold至70%
                    - 增加ADX過濾強度(35)
                    - 調整風險回報比為1:2 (止盈ATR倍數>2.5)
                    - 只在ranging/weak_trend狀態交易
                    """)
                elif metrics['profit_factor'] > 1.5:
                    st.success("""
                    獲利因子 > 1.5,策略表現優秀!
                    
                    下一步:
                    - 測試更長時間(60-90天)
                    - 測試其他幣種
                    - 測試1小時週期
                    - 準備實盤測試
                    """)
                else:
                    st.info("""
                    獲利因子 1.0-1.5,策略有潛力
                    
                    建議:
                    - 微調閾值參數
                    - 加入更多過濾條件
                    - 優化出場策略
                    """)
    
    with tabs[2]:
        st.header("BB即時分析")
        
        live_symbol = st.text_input("交易對", value="BTCUSDT", key="bb_live_symbol")
        
        if st.button("分析當前市場", key="bb_live_btn"):
            with st.spinner("載入數據..."):
                loader = BinanceDataLoader()
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                
                df = loader.load_historical_data(live_symbol, '15m', start_date, end_date)
            
            with st.spinner("生成BB信號..."):
                try:
                    signal_gen = BBBounceSignalGenerator(
                        bb_model_dir='models/saved',
                        bb_bounce_threshold=0.60,
                        adx_strong_trend_threshold=30
                    )
                    
                    df_signals = signal_gen.generate_signals(df)
                    
                    latest = df_signals.iloc[-1]
                    
                    st.subheader(f"當前市場狀態 - {live_symbol}")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("當前價格", f"${latest['close']:.2f}")
                        st.metric("BB上軌", f"${latest['bb_upper']:.2f}")
                        st.metric("BB下軌", f"${latest['bb_lower']:.2f}")
                    
                    with col2:
                        st.metric("BB位置", f"{latest['bb_position']:.2%}")
                        st.metric("BB寬度", f"{latest['bb_width']:.2f}%")
                        trend_map = {'strong_uptrend': '強多頭', 'weak_uptrend': '弱多頭', 
                                   'ranging': '盤整', 'weak_downtrend': '弱空頭', 'strong_downtrend': '強空頭'}
                        st.metric("趨勢狀態", trend_map.get(latest['trend_state'], latest['trend_state']))
                    
                    with col3:
                        st.metric("ADX", f"{latest['adx']:.1f}")
                        st.metric("RSI", f"{latest['rsi']:.1f}")
                        
                        touch_status = "無"
                        if latest['touch_upper'] == 1:
                            touch_status = "上軌"
                        elif latest['touch_lower'] == 1:
                            touch_status = "下軌"
                        st.metric("觸碰狀態", touch_status)
                    
                    with col4:
                        if latest['bb_upper_bounce_prob'] > 0:
                            st.metric("上軌反彈機率", f"{latest['bb_upper_bounce_prob']:.1%}")
                        if latest['bb_lower_bounce_prob'] > 0:
                            st.metric("下軌反彈機率", f"{latest['bb_lower_bounce_prob']:.1%}")
                        
                        signal_map = {1: '做多', -1: '做空', 0: '觀望'}
                        signal_name = signal_map.get(int(latest['signal']), '觀望')
                        color = 'green' if latest['signal'] == 1 else ('red' if latest['signal'] == -1 else 'gray')
                        st.markdown(f"### :{color}[{signal_name}]")
                    
                    if latest['signal'] != 0:
                        st.info(f"**信號原因**: {latest['signal_reason']}")
                    
                    st.subheader("最近信號")
                    recent_signals = df_signals[df_signals['signal'] != 0].tail(10)
                    if not recent_signals.empty:
                        st.dataframe(recent_signals[[
                            'open_time', 'close', 'signal', 'bb_upper_bounce_prob',
                            'bb_lower_bounce_prob', 'trend_state', 'adx', 'rsi'
                        ]].round(2))
                    else:
                        st.info("近期無交易信號")
                    
                except FileNotFoundError:
                    st.error("BB模型未找到,請先訓練模型")

else:  # 原有反轉策略
    st.sidebar.info("""
    **反轉策略 v1-v5**
    
    15分鐘趨勢 + 反轉偵測 + ATR止盈止損
    """)
    
    tabs = st.tabs(["模型訓練", "回測", "即時分析"])
    
    with tabs[0]:
        st.header("模型訓練 (15分鐘趨勢 + 反轉)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            symbol = st.text_input("交易對", value="BTCUSDT")
        
        with col2:
            days = st.number_input("訓練天數", min_value=30, max_value=365, value=180)
        
        with col3:
            oos_size = st.number_input("樣本外數量", min_value=500, max_value=3000, value=1500)
        
        if st.button("開始訓練"):
            with st.spinner("載入數據中..."):
                loader = BinanceDataLoader()
                
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                df_15m = loader.load_historical_data(symbol, '15m', start_date, end_date)
                
                feature_engineer = FeatureEngineer()
                df_15m = feature_engineer.create_features(df_15m, timeframe='15m')
                
                st.write(f"15分鐘數據: {len(df_15m)} 筆")
            
            st.subheader("訓練 15分鐘趨勢偵測模型")
            with st.spinner("訓練趨勢模型中..."):
                trend_trainer = TrendModelTrainer()
                train_df_trend, oos_df_trend = trend_trainer.prepare_data(df_15m, oos_size=oos_size)
                
                st.write(f"訓練樣本: {len(train_df_trend)}, 樣本外: {len(oos_df_trend)}")
                
                metrics = trend_trainer.train(train_df_trend)
                
                if not oos_df_trend.empty:
                    oos_metrics = trend_trainer.evaluate_oos(oos_df_trend)
                
                trend_trainer.save_models(symbol)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("分類準確率", f"{metrics['classification_accuracy']*100:.2f}%")
                    if not oos_df_trend.empty:
                        st.metric("樣本外分類準確率", f"{oos_metrics['oos_classification_accuracy']*100:.2f}%")
                
                with col2:
                    st.metric("回歸RMSE", f"{metrics['regression_rmse']:.2f}")
                    if not oos_df_trend.empty:
                        st.metric("樣本外回歸RMSE", f"{oos_metrics['oos_regression_rmse']:.2f}")
            
            st.subheader("訓練 15分鐘反轉偵測模型 (二元分類)")
            with st.spinner("訓練反轉模型中..."):
                reversal_trainer = ReversalModelTrainer()
                train_df_rev, oos_df_rev = reversal_trainer.prepare_data(df_15m, oos_size=oos_size)
                
                st.write(f"訓練樣本: {len(train_df_rev)}, 樣本外: {len(oos_df_rev)}")
                
                metrics = reversal_trainer.train(train_df_rev)
                
                if not oos_df_rev.empty:
                    oos_metrics = reversal_trainer.evaluate_oos(oos_df_rev)
                
                reversal_trainer.save_models(symbol)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("反轉信號準確率", f"{metrics['signal_accuracy']*100:.2f}%")
                    if not oos_df_rev.empty:
                        st.metric("樣本外反轉信號準確率", f"{oos_metrics['oos_signal_accuracy']*100:.2f}%")
                
                with col2:
                    st.metric("概率RMSE", f"{metrics['probability_rmse']:.2f}")
                    if not oos_df_rev.empty:
                        st.metric("樣本外概率RMSE", f"{oos_metrics['oos_probability_rmse']:.2f}")
                
                with col3:
                    st.metric("支撐MAE", f"{metrics['support_mae']:.2f}")
                    if not oos_df_rev.empty:
                        st.metric("支撐MAE %", f"{oos_metrics['oos_support_mae_pct']:.2f}%")
            
            st.success("訓練完成")
    
    # 原有回測和即時分析標籤頁...
    # (保持原有代碼)