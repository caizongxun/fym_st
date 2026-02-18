import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from data.binance_loader import BinanceDataLoader
from utils.bb_reversal_detector import BBReversalDetector
from ui.selectors import symbol_selector

def render_bb_visualization_tab(loader):
    """
    BB反轉點視覺化 Tab
    
    Args:
        loader: 數據加載器
    """
    st.header("BB反轉點視覺化")
    
    st.info("""
    **BB觸碰反轉定義**:
    1. 價格觸碰BB上軒/下軒
    2. 過濾走勢中的觸碰 (假突破)
    3. 隨後N根K線出現有效反轉
    4. 確認回到BB中軒附近
    
    **標記說明**:
    - 紅色三角: 上軒觸碰後下跌反轉 (做空機會)
    - 綠色三角: 下軒觸碰後上漨反轉 (做多機會)
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        viz_symbols = symbol_selector(loader, "bb_viz", multi=False)
        viz_symbol = viz_symbols[0]
        viz_days = st.slider(
            "顯示天數",
            min_value=3,
            max_value=30,
            value=7,
            key="viz_days"
        )
        viz_candles = viz_days * 96
    
    with col2:
        st.subheader("參數設定")
        bb_period = st.number_input(
            "BB周期",
            min_value=10,
            max_value=50,
            value=20,
            key="bb_period_viz"
        )
        bb_std = st.number_input(
            "BB標準差",
            min_value=1.0,
            max_value=3.0,
            value=2.0,
            step=0.5,
            key="bb_std_viz"
        )
        touch_threshold = st.slider(
            "觸碰閑值 (%)",
            min_value=0.0,
            max_value=0.5,
            value=0.1,
            step=0.05,
            key="touch_threshold_viz"
        ) / 100
        min_reversal = st.slider(
            "最小反轉幅度 (%)",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
            key="min_reversal_viz"
        ) / 100
    
    if st.button("生成BB反轉點圖表", key="gen_bb_viz", type="primary"):
        with st.spinner(f"載入 {viz_symbol} 數據..."):
            try:
                if isinstance(loader, BinanceDataLoader):
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=viz_days+1)
                    df = loader.load_historical_data(
                        viz_symbol, '15m', start_date, end_date
                    )
                else:
                    df = loader.load_klines(viz_symbol, '15m')
                    
                df = df.tail(viz_candles)
                
                detector = BBReversalDetector(
                    bb_period=bb_period,
                    bb_std=bb_std,
                    touch_threshold=touch_threshold,
                    reversal_confirm_candles=5,
                    min_reversal_pct=min_reversal,
                    trend_filter_enabled=True,
                    trend_lookback=10,
                    require_middle_return=True
                )
                
                df_result = detector.detect_reversals(df)
                stats = detector.get_statistics(df_result)
                
                st.subheader(f"{viz_symbol} BB反轉點分析")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("總反轉點", stats['total_reversals'])
                with col2:
                    st.metric(
                        "上軒反轉",
                        f"{stats['upper_reversals']} ({stats['upper_success_rate']:.1f}%)"
                    )
                with col3:
                    st.metric(
                        "下軒反轉",
                        f"{stats['lower_reversals']} ({stats['lower_success_rate']:.1f}%)"
                    )
                
                # 繪圖
                fig = detector.plot_reversals(
                    df_result,
                    n_candles=viz_candles,
                    title=f"{viz_symbol} BB反轉點檢測"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 反轉點明細
                if len(detector.reversals) > 0:
                    st.subheader("反轉點明細")
                    reversals_df = pd.DataFrame(detector.reversals)
                    reversals_df['reversal_pct'] = reversals_df['reversal_pct'].apply(
                        lambda x: f"{x:.2%}"
                    )
                    st.dataframe(
                        reversals_df[[
                            'time', 'type', 'reversal_type',
                            'touch_price', 'target_price', 'reversal_pct'
                        ]],
                        use_container_width=True
                    )
                else:
                    st.warning("沒有檢測到符合條件的反轉點")
                    
            except Exception as e:
                st.error(f"錯誤: {str(e)}")
                import traceback
                st.code(traceback.format_exc())