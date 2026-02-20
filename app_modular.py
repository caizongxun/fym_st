import streamlit as st
from data.binance_loader import BinanceDataLoader
from data.huggingface_loader import HuggingFaceKlineLoader
from tabs.tab_strategy_a import render_strategy_a_tab
from tabs.tab_strategy_b import render_strategy_b_tab
from tabs.tab_strategy_c import render_strategy_c_tab
from tabs.tab_strategy_d import render_strategy_d_tab
from tabs.tab_strategy_e import render_strategy_e_tab
from tabs.tab_strategy_f import render_strategy_f_tab
from tabs.tab_strategy_g import render_strategy_g_tab
from tabs.tab_strategy_h import render_strategy_h_tab
from tabs.tab_strategy_i import render_strategy_i_tab
from tabs.tab_strategy_j import render_strategy_j_tab
from tabs.tab_strategy_k import render_strategy_k_tab
from tabs.tab_strategy_l import render_strategy_l_tab

st.set_page_config(
    page_title="多策略交易系統",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("多策略交易系統")
st.caption("策略A-H:基礎 | I-K:激進 | L:終極 (10年數據) 🏆")

st.sidebar.title("系統設定")

strategy_choice = st.sidebar.radio(
    "選擇策略",
    [
        "A: SMC (Smart Money)",
        "B: SSL Hybrid + AI",
        "C: 斐波那契回調",
        "D: AI動態網格",
        "E: K棒影線 AI",
        "F: 動量趨勢 AI",
        "G: 強化學習 Agent 🤖",
        "H: 混合智能系統 🚀",
        "---",
        "I: 極致激進H (10x) 🔥",
        "J: 網格+趨勢雙引擎 🎯",
        "K: RL Agent 激進版 🤖🔥",
        "---",
        "L: 終極系統 (10年數據) 🏆"
    ]
)

st.sidebar.markdown("---")

if strategy_choice.startswith("L"):
    st.sidebar.markdown("""
### 策略L: 終極系統 🏆

**利用 10 年完整數據**:
- 2016-2026 全部歷史
- 3 個牛熊週期
- 自動識別每個幣開始時間

**智能系統**:
1. 環境分類器 (牛/熊/震盪)
2. 分環境訓練 (專屬策略)
3. 參數優化 (最佳組合)
4. Walk-Forward 驗證

**優勢**:
- 每種市場用最佳策略
- 避免過擬合
- 參數經大量驗證

**目標**: +100%+ / 30天

---
    """)
elif strategy_choice.startswith("K"):
    st.sidebar.markdown("""
### 策略K: RL 激進版 🤖🔥

**目標**: 30天 +100-150%

**改造點**:
- 10x 槓桿 (放大2倍)
- 允許多倉重疊
- 最大倉位 200%
- Reward = 日報酬率

**風險**:
- 可能爆倉 (-100%)
- 不可預測
- 最高潛力

---
    """)
elif strategy_choice.startswith("J"):
    st.sidebar.markdown("""
### 策略J: 雙引擎 🎯

**目標**: 30天 +80-100%

**引擎 1** (50%): 網格
- 日交易 20-30 次
- 日均 +1-2%

**引擎 2** (50%): 趨勢
- 抓大行情
- 週均 +10-20%

**優勢**:
- 震盪靠網格
- 趨勢靠突破
- 風險分散

---
    """)
elif strategy_choice.startswith("I"):
    st.sidebar.markdown("""
### 策略I: 激進H 🔥

**目標**: 30天 +100%

**設置**:
- 10x 槓桿 + 80% 倉位
- 快進快出 (ATR*2/0.8)
- ADX>35 最強趨勢
- 高頻交易

**風險**:
- 最大回撤 -40%
- 連續虧損可能爆倉

---
    """)
elif strategy_choice.startswith("H"):
    st.sidebar.markdown("""
### 策略H: 混合智能 🚀

**三層架構**:
- 🧠 市場狀態識別 (ML)
- 🎯 自適應信號
- ⚡ 智能風控

**優勢**:
- 多/空自動切換
- 多時間框架共振
- 白盒可解釋

---
    """)
elif strategy_choice.startswith("G"):
    st.sidebar.markdown("""
### 策略G: RL Agent 🤖

**革命性**:
- 直接學習賺錢
- 自主決策
- 無需 TP/SL

---
    """)
elif strategy_choice.startswith("F"):
    st.sidebar.markdown("""
### 策略F: 動量 AI ⭐⭐⭐

**特徵**:
- 連續陽/陰線
- 高低點突破
- EMA排列
- ADX + ROC

---
    """)
elif strategy_choice.startswith("E"):
    st.sidebar.markdown("""
### 策略E: K棒 AI ⭐⭐

**學習**:
- 影線模式
- K棒形態
- RSI/MACD/BB

---
    """)
elif strategy_choice.startswith("D"):
    st.sidebar.markdown("""
### 策略D: AI網格

**AI增強**:
- 預測波動
- 動態調整

---
    """)
elif strategy_choice.startswith("C"):
    st.sidebar.markdown("""
### 策略C: 斐波那契

**Fibonacci**:
- 38.2%/50%/61.8%

---
    """)
elif strategy_choice.startswith("B"):
    st.sidebar.markdown("""
### 策略B: SSL+AI

**SSL + XGBoost**:
- 過濾假信號

---
    """)
else:
    st.sidebar.markdown("""
### 策略A: SMC

**Smart Money**:
- Order Block
- FVG

---
    """)

data_source = st.sidebar.radio(
    "資料源",
    ["HuggingFace (38幣種)", "Binance API (即時)"]
)

if data_source == "HuggingFace (38幣種)":
    loader = HuggingFaceKlineLoader()
    st.sidebar.success("使用 HuggingFace 離線資料")
else:
    loader = BinanceDataLoader()
    st.sidebar.info("使用 Binance 即時資料")

st.sidebar.markdown("---")

def symbol_selector(key_prefix: str, multi: bool = False, default_symbols: list = None):
    if isinstance(loader, HuggingFaceKlineLoader):
        symbol_groups = HuggingFaceKlineLoader.get_symbol_groups()
        col1, col2 = st.columns([1, 2])
        with col1:
            selection_mode = st.radio(
                "選擇模式",
                ["熱門Top10", "按分類", "手動輸入"],
                key=f"{key_prefix}_mode"
            )
        with col2:
            if selection_mode == "熱門Top10":
                top_symbols = HuggingFaceKlineLoader.get_top_symbols(10)
                if multi:
                    selected = st.multiselect("選擇幣種", top_symbols,
                        default=default_symbols or top_symbols[:2], key=f"{key_prefix}_top")
                else:
                    selected = [st.selectbox("選擇幣種", top_symbols, key=f"{key_prefix}_top_single")]
            elif selection_mode == "按分類":
                category = st.selectbox("選擇分類", list(symbol_groups.keys()), key=f"{key_prefix}_category")
                symbols_in_category = symbol_groups[category]
                if multi:
                    selected = st.multiselect(f"{category} 幣種", symbols_in_category,
                        default=default_symbols or symbols_in_category[:2], key=f"{key_prefix}_cat_multi")
                else:
                    selected = [st.selectbox(f"{category} 幣種", symbols_in_category, key=f"{key_prefix}_cat_single")]
            else:
                if multi:
                    text_input = st.text_area("輸入幣種(逗號)",
                        value=",".join(default_symbols) if default_symbols else "BTCUSDT,ETHUSDT",
                        key=f"{key_prefix}_manual", height=100)
                    selected = [s.strip().upper() for s in text_input.split(',') if s.strip()]
                else:
                    selected = [st.text_input("輸入幣種", value="BTCUSDT",
                        key=f"{key_prefix}_manual_single").strip().upper()]
        return selected
    else:
        if multi:
            text_input = st.text_area("交易對(逗號)", value="BTCUSDT,ETHUSDT", key=f"{key_prefix}_binance")
            return [s.strip().upper() for s in text_input.split(',') if s.strip()]
        else:
            return [st.text_input("交易對", value="BTCUSDT", key=f"{key_prefix}_binance_single").strip().upper()]

# 渲染策略
if strategy_choice.startswith("A"):
    render_strategy_a_tab(loader, symbol_selector)
elif strategy_choice.startswith("B"):
    render_strategy_b_tab(loader, symbol_selector)
elif strategy_choice.startswith("C"):
    render_strategy_c_tab(loader, symbol_selector)
elif strategy_choice.startswith("D"):
    render_strategy_d_tab(loader, symbol_selector)
elif strategy_choice.startswith("E"):
    render_strategy_e_tab(loader, symbol_selector)
elif strategy_choice.startswith("F"):
    render_strategy_f_tab(loader, symbol_selector)
elif strategy_choice.startswith("G"):
    render_strategy_g_tab(loader, symbol_selector)
elif strategy_choice.startswith("H"):
    render_strategy_h_tab(loader, symbol_selector)
elif strategy_choice.startswith("I"):
    render_strategy_i_tab(loader, symbol_selector)
elif strategy_choice.startswith("J"):
    render_strategy_j_tab(loader, symbol_selector)
elif strategy_choice.startswith("K"):
    render_strategy_k_tab(loader, symbol_selector)
else:  # L
    render_strategy_l_tab(loader, symbol_selector)

st.sidebar.markdown("---")
st.sidebar.info("""
### 策略分類

**基礎版 (A-H)**:
- 穩健路線
- 適合入門

**激進版 (I-K)** 🔥:
- 目標 +100% / 30天
- 10x 槓桿
- 高風險高報酬

**終極版 (L)** 🏆:
- 利用 10 年完整數據
- 環境分類 + 參數優化
- Walk-Forward 驗證
- 最高穩健性

**推薦順序**:
1. 先試 L (終極版)
2. 再試 J (雙引擎)
3. 最後 K (極致)
""")
