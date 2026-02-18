import streamlit as st
from data.binance_loader import BinanceDataLoader
from data.huggingface_loader import HuggingFaceKlineLoader

# 導入所有 Tab 模組
from tabs import (
    render_bb_visualization_tab,
    render_bb_training_tab,
    render_bb_backtest_tab,
    render_scalping_training_tab,
    render_scalping_backtest_tab,
    render_nextbar_training_tab
)

# 頁面配置
st.set_page_config(
    page_title="AI 加密貨幣交易儀表板",
    layout="wide"
)

st.title("AI 加密貨幣交易儀表板 - 多策略系統")

# 側邊欄設定
st.sidebar.title("設定")
data_source = st.sidebar.radio(
    "資料源",
    ["HuggingFace (38幣)", "Binance API (即時)"],
    help="HuggingFace: 離線資料,快速穩定\nBinance: 即時資料,需網絡"
)

# 初始化數據加載器
if data_source == "HuggingFace (38幣)":
    loader = HuggingFaceKlineLoader()
    st.sidebar.success("使用HuggingFace離線資料")
else:
    loader = BinanceDataLoader()
    st.sidebar.info("使用Binance即時資料")

# 創建 Tabs
tabs = st.tabs([
    "BB反轉視覺化",
    "BB反轉訓練(OOS)",
    "BB模型回測",
    "剝頭皮訓練",
    "剝頭皮回測",
    "下一根K棒預測"
])

# 渲染各 Tab
with tabs[0]:
    render_bb_visualization_tab(loader)

with tabs[1]:
    render_bb_training_tab(loader)

with tabs[2]:
    render_bb_backtest_tab(loader)

with tabs[3]:
    render_scalping_training_tab(loader)

with tabs[4]:
    render_scalping_backtest_tab(loader)

with tabs[5]:
    render_nextbar_training_tab(loader)