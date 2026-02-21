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
    page_title="BB+NW 波段反轉交易系統",
    page_icon="■",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # 主標題
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h2 style='margin: 5px 0;'>BB+NW</h2>
        <h3 style='color: #7f7f7f; margin: 0; font-weight: normal;'>波段反轉系統</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # 頁面選擇
    page = st.sidebar.radio(
        "主選單",
        [
            "控制台", 
            "模型訓練", 
            "回測分析",
            "機率校準",
            "策略優化", 
            "流動性分析",
            "即時預測"
        ],
        label_visibility="collapsed"
    )
    
    # 系統狀態
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 系統核心
    
    **觸發層**: BB + NW 雙通道  
    **特徵層**: ADX + CVD + VWWA  
    **AI 層**: LightGBM Meta-Label  
    **出場**: 動態移動止損  
    
    ---
    
    ### 防禁機制
    
    - 單邊趨勢輾壓過濾 (ADX + HTF EMA)  
    - 獵取流動性辨識 (CVD 背離)  
    - BB 壓縮突破偵測  
    - 無未來函數 (No Repaint)  
    
    ---
    
    ### 適用市場
    
    **時間框架**: 15m (进场) + 1h (趨勢)  
    **交易風格**: 波段反轉 (Swing Reversal)  
    **持倉時間**: 4-20 小時  
    **勝率目標**: 55-65%  
    **盈虧比**: 2.5:1 ~ 4:1  
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.caption("v2.0 - Swing Reversal Edition")
    
    # 路由頁面
    if "控制台" in page:
        dashboard_page.render()
    elif "訓練" in page:
        training_page.render()
    elif "回測" in page:
        backtesting_page.render()
    elif "校準" in page:
        calibration_page.render()
    elif "優化" in page:
        optimization_page.render()
    elif "流動性" in page:
        liquidity_sweep_page.render()
    elif "預測" in page:
        live_prediction_page.render()

if __name__ == "__main__":
    main()