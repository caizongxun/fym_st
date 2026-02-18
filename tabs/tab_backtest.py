import streamlit as st

def render_backtest_tab(loader):
    st.header("步驟 4: 歷史回測")
    st.info("開發中... 將測試完整策略")
    st.write("回測邏輯:")
    st.write("- 價格接近上軌 + 反轉機率 > 70% + 趨勢強度 < 30% = 做空")
    st.write("- 價格接近下軌 + 反轉機率 > 70% + 趨勢強度 < 30% = 做多")
    st.write("- 止盈: 中軌")
    st.write("- 止損: 突破軌道外 0.2%")