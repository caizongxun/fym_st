import streamlit as st

def render_live_monitor_tab(loader):
    st.header("步驟 5: 實時監控")
    st.warning("實盤交易有風險!")
    st.info("開發中... 將每秒更新市場資訊")
    st.write("功能:")
    st.write("- 即時計算 BB 通道")
    st.write("- 價格接近軌道時調用模型")
    st.write("- 顯示反轉機率和趨勢強度")
    st.write("- 自動下單 (可選)")