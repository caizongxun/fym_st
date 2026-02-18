import streamlit as st
from data.binance_loader import BinanceDataLoader
from data.huggingface_loader import HuggingFaceKlineLoader

# 導入所有 Tab 模組
from tabs import (
    render_data_analysis_tab,
    render_feature_engineering_tab,
    render_transformer_training_tab,
    render_ensemble_training_tab,
    render_rl_training_tab,
    render_backtest_tab,
    render_live_trading_tab
)

# 頁面配置
st.set_page_config(
    page_title="Ensemble RL-Transformer 交易系統",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 Ensemble RL-Transformer 交易系統")
st.caption("目標: 10U 一個月翻倉 | 每天 5-10 筆交易")

# 側邊欄設定
st.sidebar.title("⚙️ 系統設定")

st.sidebar.markdown("""
### 系統架構

**第 1 層**: 多時間框架特徵
- 5m / 15m / 1h K線
- 技術指標 (不篩選)
- 波動率特徵

**第 2 層**: Ensemble 預測
- Transformer (40%)
- LSTM (30%)
- XGBoost (20%)
- Attention-GRU (10%)

**第 3 層**: RL 智能體
- 自主決策進場時機
- 動態個位管理
- 風險控制
---
""")

data_source = st.sidebar.radio(
    "資料源",
    ["HuggingFace (38幣)", "Binance API (即時)"],
    help="HuggingFace: 離線資料,快速穩定\nBinance: 即時資料,需網路"
)

# 初始化數據加載器
if data_source == "HuggingFace (38幣)":
    loader = HuggingFaceKlineLoader()
    st.sidebar.success("✅ 使用 HuggingFace 離線資料")
else:
    loader = BinanceDataLoader()
    st.sidebar.info("✅ 使用 Binance 即時資料")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 使用流程

1️⃣ **數據分析**: 了解多時間框架  
2️⃣ **特徵工程**: 提取特徵  
3️⃣ **Transformer**: 訓練核心模型  
4️⃣ **Ensemble**: 整合多模型  
5️⃣ **RL 訓練**: 智能決策  
6️⃣ **歷史回測**: 驗證策略  
7️⃣ **實盤交易**: 自動交易  
""")

# 建立 Tabs
tabs = st.tabs([
    "📊 1. 數據分析",
    "🔧 2. 特徵工程",
    "🧠 3. Transformer",
    "🤝 4. Ensemble",
    "🎯 5. RL 訓練",
    "📊 6. 歷史回測",
    "🚀 7. 實盤交易"
])

# 渲染各 Tab
with tabs[0]:
    render_data_analysis_tab(loader)

with tabs[1]:
    render_feature_engineering_tab(loader)

with tabs[2]:
    render_transformer_training_tab(loader)

with tabs[3]:
    render_ensemble_training_tab(loader)

with tabs[4]:
    render_rl_training_tab(loader)

with tabs[5]:
    render_backtest_tab(loader)

with tabs[6]:
    render_live_trading_tab(loader)

# 底部資訊
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 目標設定
- 初始資金: **10 USDT**
- 目標報酬: **1000% (30天)**
- 每天交易: **5-10 筆**
- 時間框架: **15分鐘**
- 預期勝率: **55-65%**
- 目標 Sharpe: **1.5-2.5**
""")