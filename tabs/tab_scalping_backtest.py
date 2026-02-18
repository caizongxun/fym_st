import streamlit as st
import os
import pandas as pd
from datetime import datetime, timedelta

from data.binance_loader import BinanceDataLoader
from data.huggingface_loader import HuggingFaceKlineLoader
from utils.signal_generator_scalping import ScalpingSignalGenerator
from backtesting.limit_order_engine import LimitOrderBacktestEngine
from ui.components import display_metrics

def render_scalping_backtest_tab(loader):
    """
    剝頭皮回測 Tab (Limit Order)
    
    Args:
        loader: 數據加載器
    """
    st.header("剝頭皮模型回測 (Limit Order)")
    
    st.info("""
    **Limit Order 掛單回測**:
    1. 模型預測信號 + 置信度 > 65%
    2. 在當前價 ± 0.1% 掛限價單 (等回調)
    3. 只有價格觸碰限價才成交
    4. 止盈: Limit Order (Maker 0.02%)
    5. 止損: Stop Market (Taker 0.06%)
    
    **優勢**:
    - 雙Maker結構，大幅降低手續費
    - 不追高殺低，等最佳進場點
    - 模擬真實掛單成交邏輯
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        scalp_bt_symbol = st.selectbox(
            "回測幣種",
            ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            key="scalp_bt_symbol"
        )
        
        scalp_bt_days = st.slider(
            "回測天數",
            30, 180, 60,
            key="scalp_bt_days"
        )
        
        confidence_threshold = st.slider(
            "置信度閑值",
            0.5, 0.9, 0.65,
            step=0.05,
            key="scalp_confidence",
            help="只有模型置信度 > 此值才進場"
        )
    
    with col2:
        st.subheader("風險管理")
        
        scalp_capital = st.number_input(
            "初始資金",
            100, 10000, 1000,
            key="scalp_capital"
        )
        
        scalp_leverage = st.number_input(
            "桓桿倍數",
            1, 20, 5,
            key="scalp_leverage"
        )
        
        scalp_position_pct = st.slider(
            "單筆倉位 (%)",
            10, 50, 20,
            key="scalp_position_pct"
        ) / 100
    
    st.caption(
        f"回測: {scalp_bt_days}天 | 資金: ${scalp_capital} | "
        f"桓桿: {scalp_leverage}x | 單筆: {scalp_position_pct*100:.0f}% | "
        f"置信度: {confidence_threshold:.0%}"
    )
    
    if st.button("開始剝頭皮回測", key="scalp_bt_btn", type="primary"):
        model_path = f"models/saved/{scalp_bt_symbol}_scalping_lightgbm_oos_scalping_lightgbm.pkl"
        
        if not os.path.exists(model_path):
            st.error(f"找不到模型: {model_path}，請先訓練模型")
        else:
            with st.spinner(f"正在回測 {scalp_bt_symbol}..."):
                try:
                    # 載入數據
                    if isinstance(loader, BinanceDataLoader):
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=scalp_bt_days)
                        df = loader.load_historical_data(
                            scalp_bt_symbol, '15m', start_date, end_date
                        )
                    else:
                        df = loader.load_klines(scalp_bt_symbol, '15m')
                        df = df.tail(scalp_bt_days * 96)
                    
                    st.info(f"載入 {len(df)} 根K線")
                    
                    # 生成信號
                    generator = ScalpingSignalGenerator(
                        model_path=model_path,
                        confidence_threshold=confidence_threshold,
                        entry_offset_pct=0.001,
                        tp_pct=0.003,
                        sl_pct=0.002
                    )
                    
                    df_signals = generator.generate_signals(df)
                    
                    signal_count = (df_signals['signal'] != 0).sum()
                    long_count = (df_signals['signal'] == 1).sum()
                    short_count = (df_signals['signal'] == -1).sum()
                    
                    st.info(
                        f"生成 {signal_count} 個信號 "
                        f"(做多: {long_count}, 做空: {short_count})"
                    )
                    
                    # 執行回測
                    engine = LimitOrderBacktestEngine(
                        initial_capital=scalp_capital,
                        leverage=scalp_leverage,
                        position_size_pct=scalp_position_pct,
                        maker_fee=0.0002,
                        taker_fee=0.0006
                    )
                    
                    signals_dict = {scalp_bt_symbol: df_signals}
                    metrics = engine.run_backtest(signals_dict)
                    
                    # 顯示結果
                    st.subheader("回測結果")
                    display_metrics(metrics)
                    
                    # 權益曲線
                    fig = engine.plot_equity_curve()
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 交易明細
                    trades_df = engine.get_trades_dataframe()
                    if not trades_df.empty:
                        st.subheader("交易明細")
                        display_cols = [
                            '進場時間', '離場時間', '方向',
                            '進場價格', '離場價格',
                            '損益(USDT)', '損益率',
                            '離場原因', '持倉時長(分)'
                        ]
                        st.dataframe(
                            trades_df[display_cols],
                            use_container_width=True
                        )
                        
                        # 統計分析
                        st.subheader("交易統計")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            tp_count = (trades_df['exit_reason'] == 'TP').sum()
                            st.metric("止盈交易", tp_count)
                        
                        with col2:
                            sl_count = (trades_df['exit_reason'] == 'SL').sum()
                            st.metric("止損交易", sl_count)
                        
                        with col3:
                            if 'signal_data' in trades_df.columns:
                                avg_conf = trades_df['signal_data'].apply(
                                    lambda x: x.get('confidence', 0) 
                                    if isinstance(x, dict) else 0
                                ).mean()
                                st.metric("平均置信度", f"{avg_conf:.2%}")
                            else:
                                st.metric("平均置信度", "N/A")
                    else:
                        st.warning("沒有產生任何交易，請降低置信度閑值")
                        
                except Exception as e:
                    st.error(f"回測錯誤: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())