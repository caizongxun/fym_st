import streamlit as st
import os
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

from data.binance_loader import BinanceDataLoader
from data.huggingface_loader import HuggingFaceKlineLoader
from utils.signal_generator_nextbar import NextBarSignalGenerator
from backtest.nextbar_backtest_engine import NextBarBacktestEngine
from ui.selectors import symbol_selector

def render_nextbar_backtest_tab(loader):
    """
    下一根K棒預測回測 Tab
    """
    st.header("下一根K棒預測 - 回測")
    
    st.info("""
    **回測策略**: 雙向限價單
    
    1. 模型預測下一根K棒的 high/low
    2. 在預測 low 附近掛做多限價單 (Maker 0.02%)
    3. 在預測 high 附近掛做空限價單 (Maker 0.02%)
    4. 成交後:
       - 止盈 = 另一邊預測價
       - 止損 = 預測邊界 - 0.2%
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("模型與數據")
        
        # 選擇幣種
        nextbar_symbols = symbol_selector(loader, "nextbar_backtest", multi=False)
        nextbar_symbol = nextbar_symbols[0]
        
        # 選擇模型
        model_dir = 'models/saved'
        if os.path.exists(model_dir):
            model_files = [
                f for f in os.listdir(model_dir) 
                if f.endswith('.pkl') and 'nextbar' in f and nextbar_symbol in f
            ]
            
            if len(model_files) > 0:
                selected_model = st.selectbox(
                    "選擇模型",
                    model_files,
                    key="nextbar_backtest_model"
                )
                model_path = os.path.join(model_dir, selected_model)
            else:
                st.warning(f"未找到 {nextbar_symbol} 的模型，請先訓練")
                return
        else:
            st.error("模型目錄不存在")
            return
        
        # 設定回測天數
        if isinstance(loader, BinanceDataLoader):
            backtest_days = st.slider(
                "回測天數",
                min_value=7,
                max_value=90,
                value=30,
                key="nextbar_backtest_days"
            )
        else:
            st.info("使用HuggingFace最近30天數據")
            backtest_days = 30
    
    with col2:
        st.subheader("策略參數")
        
        entry_offset = st.slider(
            "掛單偏移 (%)",
            min_value=0.0,
            max_value=0.5,
            value=0.1,
            step=0.05,
            key="nextbar_bt_entry_offset",
            help="在預測價位留的空間"
        ) / 100
        
        sl_buffer = st.slider(
            "止損緩衝 (%)",
            min_value=0.0,
            max_value=0.5,
            value=0.2,
            step=0.05,
            key="nextbar_bt_sl_buffer",
            help="止損距離預測邊界的額外空間"
        ) / 100
        
        col_a, col_b = st.columns(2)
        with col_a:
            min_range = st.number_input(
                "最小區間 (%)",
                min_value=0.1,
                max_value=1.0,
                value=0.2,
                step=0.1,
                key="nextbar_bt_min_range"
            ) / 100
        
        with col_b:
            max_range = st.number_input(
                "最大區間 (%)",
                min_value=0.5,
                max_value=2.0,
                value=0.8,
                step=0.1,
                key="nextbar_bt_max_range"
            ) / 100
        
        initial_capital = st.number_input(
            "初始資金 (USDT)",
            min_value=100,
            max_value=100000,
            value=10000,
            step=1000,
            key="nextbar_bt_capital"
        )
    
    st.caption(
        f"回測: {nextbar_symbol} | 天數: {backtest_days} | "
        f"掛單偏移: {entry_offset*100:.1f}% | 止損緩衝: {sl_buffer*100:.1f}%"
    )
    
    if st.button("開始回測", key="run_nextbar_backtest", type="primary"):
        with st.spinner(f"正在回測 {nextbar_symbol}..."):
            try:
                # 載入數據
                if isinstance(loader, BinanceDataLoader):
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=backtest_days)
                    df = loader.load_historical_data(
                        nextbar_symbol, '15m', start_date, end_date
                    )
                else:
                    df = loader.load_klines(nextbar_symbol, '15m')
                    df = df.tail(backtest_days * 96)
                
                st.info(f"載入 {len(df)} 根K線")
                
                # 生成信號
                signal_gen = NextBarSignalGenerator(
                    model_path=model_path,
                    entry_offset_pct=entry_offset,
                    sl_buffer_pct=sl_buffer,
                    max_range_filter=max_range,
                    min_range_filter=min_range
                )
                
                df_signals = signal_gen.generate_signals(df)
                
                # 回測
                engine = NextBarBacktestEngine(
                    initial_capital=initial_capital,
                    maker_fee=0.0002,
                    taker_fee=0.0004
                )
                
                results = engine.run_backtest(df_signals)
                
                # 顯示結果
                st.subheader("回測結果")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "總交易次數",
                        results['total_trades']
                    )
                    st.metric(
                        "最終資金",
                        f"${results['final_capital']:.2f}"
                    )
                
                with col2:
                    st.metric(
                        "總報酬率",
                        f"{results['total_return_pct']*100:.2f}%"
                    )
                    st.metric(
                        "勝率",
                        f"{results['win_rate']*100:.1f}%"
                    )
                
                with col3:
                    st.metric(
                        "平均盈利",
                        f"{results['avg_win']*100:.2f}%"
                    )
                    st.metric(
                        "平均虧損",
                        f"{results['avg_loss']*100:.2f}%"
                    )
                
                with col4:
                    st.metric(
                        "盈虧因子",
                        f"{results['profit_factor']:.2f}" if results['profit_factor'] != float('inf') else "Inf"
                    )
                    st.metric(
                        "最大回撤",
                        f"{results['max_drawdown_pct']*100:.2f}%"
                    )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("盈利交易", results['winning_trades'])
                with col2:
                    st.metric("虧損交易", results['losing_trades'])
                
                # 評估
                st.subheader("策略評估")
                
                if results['win_rate'] >= 0.55 and results['profit_factor'] >= 1.5:
                    st.success(
                        f"優秀! 勝率 {results['win_rate']*100:.1f}% >= 55% 且 "
                        f"盈虧因子 {results['profit_factor']:.2f} >= 1.5, 可以考慮實盤"
                    )
                elif results['win_rate'] >= 0.50 and results['profit_factor'] >= 1.2:
                    st.info(
                        f"合格! 勝率 {results['win_rate']*100:.1f}% >= 50% 且 "
                        f"盈虧因子 {results['profit_factor']:.2f} >= 1.2, 可以繼續優化"
                    )
                else:
                    st.warning(
                        f"需要優化! 勝率 {results['win_rate']*100:.1f}% 或 "
                        f"盈虧因子 {results['profit_factor']:.2f} 偏低"
                    )
                    st.write("建議:")
                    st.write("1. 調整掛單偏移和止損緩衝")
                    st.write("2. 過濾更大或更小的預測區間")
                    st.write("3. 重新訓練模型")
                
                # 權益曲線
                if len(results['equity_df']) > 0:
                    st.subheader("權益曲線")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=results['equity_df']['equity'],
                        mode='lines',
                        name='資金',
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig.add_hline(
                        y=initial_capital,
                        line_dash="dash",
                        line_color="gray",
                        annotation_text="初始資金"
                    )
                    
                    fig.update_layout(
                        title="權益曲線",
                        xaxis_title="K線數",
                        yaxis_title="資金 (USDT)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # 交易明細
                if len(results['trades_df']) > 0:
                    st.subheader("交易明細 (最近 20 筆)")
                    trades_display = results['trades_df'][[
                        'type', 'entry_price', 'exit_price', 'pnl_pct', 'reason'
                    ]].tail(20).copy()
                    
                    trades_display['pnl_pct'] = trades_display['pnl_pct'].apply(
                        lambda x: f"{x*100:.2f}%"
                    )
                    
                    st.dataframe(trades_display, use_container_width=True)
                
            except Exception as e:
                st.error(f"回測失敗: {str(e)}")
                import traceback
                st.code(traceback.format_exc())