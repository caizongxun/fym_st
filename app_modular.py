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

st.set_page_config(
    page_title="多策略交易系統",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("多策略交易系統")
st.caption("策略A:SMC | B:SSL+AI | C:斐波那契 | D:AI網格 | E:K棒AI影線 | F:AI動量 | G:RL Agent")

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
        "G: 強化學習 Agent 🤖"
    ]
)

st.sidebar.markdown("---")

if strategy_choice.startswith("G"):
    st.sidebar.markdown("""
### 策略G: 強化學習 Agent 🤖💥

**革命性方法**:
- ❌ 不預測漲跌
- ✅ 直接學習賺錢

**Agent 自主決策**:
- 何時開倉（多/空）
- 何時平倉
- 最優持倉時間

**核心優勢**:
- Reward = 實際 PnL
- 無需手動 TP/SL
- 自動適應市場

---
    """)
elif strategy_choice.startswith("F"):
    st.sidebar.markdown("""
### 策略F: 動量趨勢 AI ⭐⭐⭐

**新特徵**:
- 連續陽/陰線計數
- 高低點突破追蹤
- EMA排列強度 (8/20/50)
- 成交量趨勢斜率
- ADX + ROC 動量組合

**目標**:
- 解決v5做多失敗
- 讓做多模型有效
- 做空維持優勢

---
    """)
elif strategy_choice.startswith("E"):
    st.sidebar.markdown("""
### 策略E: K棒影線 AI ⭐⭐

**學習內容**:
- 前10根K棒影線模式
- 上影線/下影線/實體比例
- K棒形態 (錘子/鐓錘/流星)
- RSI/MACD/BB/Stoch

**預測目標**:
- 下1根K棒方向 (+1/0/-1)
- ATR訊號門檻
- 信心度過濾

**流程**:
1. 用前90天訓練
2. 用前30天測試
3. 顯示特徵重要度
4. 執行回測

---
    """)
elif strategy_choice.startswith("D"):
    st.sidebar.markdown("""
### 策略D: AI動態網格

**AI增強網格**:
- AI預測波動範圍
- AI判斷市場狀態
- 動態調整網格

---
    """)
elif strategy_choice.startswith("C"):
    st.sidebar.markdown("""
### 策略C: 斐波那契回調

**Fibonacci**:
- 38.2%/50%/61.8%回調位
- 等待反轉確認

---
    """)
elif strategy_choice.startswith("B"):
    st.sidebar.markdown("""
### 策略B: SSL Hybrid + AI

**SSL指標 + XGBoost過濾**:
- Baseline/SSL1/SSL2/Exit
- 過濾假信號

---
    """)
else:
    st.sidebar.markdown("""
### 策略A: SMC v2

**Smart Money Concepts**:
- Order Block
- Fair Value Gap
- Market Structure

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
else:
    render_strategy_g_tab(loader, symbol_selector)

st.sidebar.markdown("---")
st.sidebar.info("""
### 預期表現

**A (SMC)**: 趨勢市場
**B (SSL+AI)**: 通用
**C (斐波)**: 趨勢市場
**D (AI網格)**: 震盪市場
**E (K棒AI)**: 所有市場 ⭐⭐
**F (動量AI)**: 所有市場 ⭐⭐⭐
**G (RL Agent)**: 自主學習 🤖💥
""")
