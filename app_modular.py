import streamlit as st
from data.binance_loader import BinanceDataLoader
from data.huggingface_loader import HuggingFaceKlineLoader
from tabs.tab_strategy_a import render_strategy_a_tab
from tabs.tab_strategy_b import render_strategy_b_tab
from tabs.tab_strategy_c import render_strategy_c_tab

st.set_page_config(
    page_title="多策略交易系統",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("多策略交易系統")
st.caption("策略A: SMC | 策略B: SSL+AI | 策略C: 斐波那契 | Tick級別回測")

st.sidebar.title("系統設定")

# 策略選擇
strategy_choice = st.sidebar.radio(
    "選擇策略",
    ["A: SMC (Smart Money)", "B: SSL Hybrid + AI", "C: 斐波那契回調"],
    help="A: 機構交易\nB: SSL指標+AI\nC: 斐波那契回調"
)

st.sidebar.markdown("---")

# 策略說明
if strategy_choice.startswith("A"):
    st.sidebar.markdown("""
### 策略A: SMC v2

**Smart Money Concepts**:
- Order Block (機構訂單區)
- Fair Value Gap (價格缺口)
- Market Structure (趨勢結構)
- Liquidity Zones (流動性區)

**改進**:
- 簡化進場条件
- 動態ATR止損
- 3:1風報比
- 只做趨勢交易

---
    """)
elif strategy_choice.startswith("B"):
    st.sidebar.markdown("""
### 策略B: SSL Hybrid + AI

**SSL指標系統**:
- Baseline: 趨勢方向
- SSL1/2: 主確認信號
- Exit: 出場信號

**AI過濾**:
- XGBoost分類器
- 過濾震盪假信號
- 只保留高品質信號

**優勢**:
- 結合經典指標
- AI智能過濾
- 適合趨勢市場

---
    """)
else:
    st.sidebar.markdown("""
### 策略C: 斐波那契回調

**Fibonacci Retracement**:
- 識別波段高低點
- 計算回調位 (38.2%, 50%, 61.8%)
- 等待反轉確認信號
- 精準進場點

**優勢**:
- 經典技術分析
- 進場點精準
- 止損止盈明確
- 適合趨勢市場

**理論**:
- 市場常在關鍵位反轉
- 斐波數列反映自然規律
- 高勝率進場點

---
    """)

data_source = st.sidebar.radio(
    "資料源",
    ["HuggingFace (38幣種)", "Binance API (即時)"],
    help="HuggingFace: 離線資料\nBinance: 即時資料"
)

if data_source == "HuggingFace (38幣種)":
    loader = HuggingFaceKlineLoader()
    st.sidebar.success("使用 HuggingFace 離線資料")
else:
    loader = BinanceDataLoader()
    st.sidebar.info("使用 Binance 即時資料")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 使用流程

1. **選擇策略** - A/B/C
2. **選擇幣種** - 要交易的幣種
3. **調整參數** - 根據喜好
4. **執行回測** - 點擊按鈕
5. **查看結果** - 分析績效

**自動完成**:
- 載入數據
- 生成信號
- Tick級別回測
- 顯示結果
""")

st.sidebar.markdown("---")

# symbol_selector helper function
def symbol_selector(key_prefix: str, multi: bool = False, default_symbols: list = None):
    """幫助函數: 幣種選擇"""
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
                    selected = st.multiselect(
                        "選擇幣種",
                        top_symbols,
                        default=default_symbols or top_symbols[:2],
                        key=f"{key_prefix}_top"
                    )
                else:
                    selected = [st.selectbox(
                        "選擇幣種",
                        top_symbols,
                        key=f"{key_prefix}_top_single"
                    )]
            
            elif selection_mode == "按分類":
                category = st.selectbox(
                    "選擇分類",
                    list(symbol_groups.keys()),
                    key=f"{key_prefix}_category"
                )
                symbols_in_category = symbol_groups[category]
                
                if multi:
                    selected = st.multiselect(
                        f"{category} 幣種",
                        symbols_in_category,
                        default=default_symbols or symbols_in_category[:2],
                        key=f"{key_prefix}_cat_multi"
                    )
                else:
                    selected = [st.selectbox(
                        f"{category} 幣種",
                        symbols_in_category,
                        key=f"{key_prefix}_cat_single"
                    )]
            
            else:
                if multi:
                    text_input = st.text_area(
                        "輸入幣種 (逗號分隔)",
                        value=",".join(default_symbols) if default_symbols else "BTCUSDT,ETHUSDT",
                        key=f"{key_prefix}_manual",
                        height=100
                    )
                    selected = [s.strip().upper() for s in text_input.split(',') if s.strip()]
                else:
                    selected = [st.text_input(
                        "輸入幣種",
                        value="BTCUSDT",
                        key=f"{key_prefix}_manual_single"
                    ).strip().upper()]
        
        return selected
    
    else:
        if multi:
            text_input = st.text_area(
                "交易對 (逗號分隔)",
                value="BTCUSDT,ETHUSDT",
                key=f"{key_prefix}_binance"
            )
            return [s.strip().upper() for s in text_input.split(',') if s.strip()]
        else:
            return [st.text_input(
                "交易對",
                value="BTCUSDT",
                key=f"{key_prefix}_binance_single"
            ).strip().upper()]

# 渲染策略
if strategy_choice.startswith("A"):
    render_strategy_a_tab(loader, symbol_selector)
elif strategy_choice.startswith("B"):
    render_strategy_b_tab(loader, symbol_selector)
else:
    render_strategy_c_tab(loader, symbol_selector)

st.sidebar.markdown("---")
st.sidebar.info("""
### 預期表現

**策略A (SMC)**:
- 勝率: 45-55%
- 盈虧比: 1.5-2.5
- 適合: 趨勢市場

**策略B (SSL+AI)**:
- 勝率: 50-60%
- 盈虧比: 1.5-2.0
- 適合: 所有市況

**策略C (斐波那契)**:
- 勝率: 55-65%
- 盈虧比: 2.0-3.0
- 適合: 趨勢市場
""")
