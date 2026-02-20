import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.pages import (
    training_page, backtesting_page, live_prediction_page, 
    dashboard_page, optimization_page, calibration_page,
    liquidity_sweep_page
)

st.set_page_config(
    page_title="加密貨幣交易系統",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.sidebar.title("導航選單")
    
    page = st.sidebar.radio(
        "選擇頁面",
        [
            "控制台", 
            "模型訓練", 
            "機率校準分析",
            "策略優化", 
            "回測分析",
            "流動性掃蕩分析",  # 新增
            "即時預測"
        ]
    )
    
    # 側邊欄資訊
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 系統功能
    
    **核心模組**:
    - 控制台: 系統概覽
    - 模型訓練: ML 模型訓練
    - 機率校準: 模型校準分析
    - 策略優化: 參數優化
    - 回測分析: 歷史回測
    - **流動性掃蕩**: 機構級分析 (NEW)
    - 即時預測: 實時交易信號
    
    **新功能**: 流動性掃蕩系統
    - OI (未平倉量) 分析
    - CVD (成交量差) 偵測
    - Smart Money 追蹤
    - 機構級進場點
    """)
    
    # 路由頁面
    if page == "控制台":
        dashboard_page.render()
    elif page == "模型訓練":
        training_page.render()
    elif page == "機率校準分析":
        calibration_page.render()
    elif page == "策略優化":
        optimization_page.render()
    elif page == "回測分析":
        backtesting_page.render()
    elif page == "流動性掃蕩分析":
        liquidity_sweep_page.render()
    elif page == "即時預測":
        live_prediction_page.render()

if __name__ == "__main__":
    main()