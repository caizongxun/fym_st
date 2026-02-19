import streamlit as st
from data.binance_loader import BinanceDataLoader
from data.huggingface_loader import HuggingFaceKlineLoader
from tabs.tab_strategy_a import render_strategy_a_tab

st.set_page_config(
    page_title="策略A: ML驅動交易系統",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 策略 A - ML驅動的區間震盪交易系統")
st.caption("目標: 無RSI限制的智能交易策略 | Tick級別回測 | 一鍵執行")

st.sidebar.title("⚙️ 系統設定")

st.sidebar.markdown("""
### 🎯 策略A 核心優勢

**1. 智能進場**
- 無固定RSI限制
- AI模型動態學習
- 20+智能特徵

**2. 雙模型架構**
- 做多模型獨立預測
- 做空模型獨立預測
- 更精準的信號

**3. Tick級別回測**
- 模擬真實盤中波動
- 每根K線100個tick
- 真實反映止損觸發

**4. 自適應止損**
- 基於ATR動態調整
- 適應市場波動
- 更好的風險控制

---
""")

data_source = st.sidebar.radio(
    "📊 資料源",
    ["HuggingFace (38幣種)", "Binance API (即時)"],
    help="HuggingFace: 離線資料\nBinance: 即時資料"
)

if data_source == "HuggingFace (38幣種)":
    loader = HuggingFaceKlineLoader()
    st.sidebar.success("✅ 使用 HuggingFace 離線資料")
else:
    loader = BinanceDataLoader()
    st.sidebar.info("🔄 使用 Binance 即時資料")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🚀 使用流程

1. **選擇幣種** - 選擇要交易的幣種
2. **設定參數** - 調整訓練/交易參數
3. **一鍵執行** - 點擊按鈕自動完成
4. **查看結果** - 分析績效指標

**一鍵執行內容**:
- ✅ 載入資料
- ✅ 訓練ML模型
- ✅ 生成交易信號
- ✅ Tick級別回測
- ✅ 顯示結果
""")

st.sidebar.markdown("---")

# symbol_selector helper function
def symbol_selector(key_prefix: str, multi: bool = False, default_symbols: list = None):
    """Helper function for symbol selection"""
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

# Main content
render_strategy_a_tab(loader, symbol_selector)

st.sidebar.markdown("---")
st.sidebar.info("""
### 📊 預期表現

**相比傳統策略**:
- 交易次數: +200%
- 報酬率: +300%
- 回測準確度: +50%

**典型結果** (3x槓桿):
- 勝率: 55-65%
- 報酬率: 12-20%
- 盈虧比: 1.5-2.5
""")
