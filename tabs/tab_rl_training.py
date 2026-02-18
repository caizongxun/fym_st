import streamlit as st

def render_rl_training_tab(loader):
    """
    Tab 5: 強化學習訓練
    """
    st.header("步驟 5: 強化學習 (PPO)")
    st.info("正在開發中... 將訓練 RL 智能體")
    st.write("功能:")
    st.write("- 輸入: Ensemble 預測 + 當前狀態")
    st.write("- 行動: 買入/賣出/持有 + 個位大小")
    st.write("- 獎勵: 利潤 - 風險 - 手續費")
    st.write("- 目標: 每天 5-10 筆交易, 月報酬 15-40%")