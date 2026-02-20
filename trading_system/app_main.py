import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.pages import training_page, backtesting_page, live_prediction_page, dashboard_page

st.set_page_config(
    page_title="Crypto Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.sidebar.title("Navigation")
    
    page = st.sidebar.radio(
        "Select Page",
        ["Dashboard", "Model Training", "Backtesting", "Live Prediction"]
    )
    
    if page == "Dashboard":
        dashboard_page.render()
    elif page == "Model Training":
        training_page.render()
    elif page == "Backtesting":
        backtesting_page.render()
    elif page == "Live Prediction":
        live_prediction_page.render()

if __name__ == "__main__":
    main()