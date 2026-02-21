import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.pages import (
    training_page, backtesting_page, live_prediction_page, 
    dashboard_page, optimization_page, calibration_page,
    liquidity_sweep_page
)
from config import SystemConfig, UIConfig

st.set_page_config(
    page_title=SystemConfig.SYSTEM_NAME,
    page_icon="[BB]",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Sidebar header
    st.sidebar.markdown(f"""
    <div style='text-align: center; padding: 20px 0;'>
        <h2 style='margin: 5px 0;'>BB+NW</h2>
        <h3 style='color: #7f7f7f; margin: 0; font-weight: normal;'>Swing Reversal System</h3>
        <p style='color: #999; font-size: 0.9em;'>v{SystemConfig.VERSION}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Page navigation
    page = st.sidebar.radio(
        "Navigation",
        [
            UIConfig.PAGE_TITLE_DASHBOARD,
            UIConfig.PAGE_TITLE_TRAINING,
            UIConfig.PAGE_TITLE_BACKTEST,
            UIConfig.PAGE_TITLE_CALIBRATION,
            UIConfig.PAGE_TITLE_OPTIMIZATION,
            UIConfig.PAGE_TITLE_LIQUIDITY,
            UIConfig.PAGE_TITLE_LIVE
        ],
        label_visibility="collapsed"
    )
    
    # System info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### System Core
    
    **Trigger Layer**: BB + NW Dual Channels  
    **Feature Layer**: ADX + CVD + VWWA  
    **AI Layer**: LightGBM Meta-Label  
    **Exit**: Dynamic Trailing Stop  
    
    ---
    
    ### Protection Mechanisms
    
    - Trend Crush Filter (ADX + HTF EMA)  
    - Liquidity Sweep Detection (CVD Divergence)  
    - BB Squeeze Breakout Detection  
    - No Repaint (Zero Future Function)  
    
    ---
    
    ### Target Markets
    
    **Timeframe**: 15m (Entry) + 1h (Trend)  
    **Style**: Swing Reversal  
    **Hold Time**: 4-20 hours  
    **Win Rate Target**: 55-65%  
    **Risk/Reward**: 2.5:1 ~ 4:1  
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.caption(f"{SystemConfig.SYSTEM_NAME} | {SystemConfig.VERSION}")
    
    # Route to pages
    if page == UIConfig.PAGE_TITLE_DASHBOARD:
        dashboard_page.render()
    elif page == UIConfig.PAGE_TITLE_TRAINING:
        training_page.render()
    elif page == UIConfig.PAGE_TITLE_BACKTEST:
        backtesting_page.render()
    elif page == UIConfig.PAGE_TITLE_CALIBRATION:
        calibration_page.render()
    elif page == UIConfig.PAGE_TITLE_OPTIMIZATION:
        optimization_page.render()
    elif page == UIConfig.PAGE_TITLE_LIQUIDITY:
        liquidity_sweep_page.render()
    elif page == UIConfig.PAGE_TITLE_LIVE:
        live_prediction_page.render()

if __name__ == "__main__":
    main()