import streamlit as st
from data.binance_loader import BinanceDataLoader
from data.huggingface_loader import HuggingFaceKlineLoader

from tabs import (
    render_bb_visualization_tab,
    render_reversal_training_tab,
    render_trend_filter_tab,
    render_backtest_tab,
    render_live_monitor_tab,
    render_range_bound_backtest_tab
)

st.set_page_config(
    page_title="BB 反轉精準捕捉系統",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("BB 反轉精準捕捉系統")
st.caption("目標: 精準預測 BB 通道反轉點, 過濾強趨勢誤判")

st.sidebar.title("系統設定")

st.sidebar.markdown("""
### 雙模型架構

**模型 1: BB 反轉預測**
- 上軌模型: 預測上軌反轉機率
- 下軌模型: 預測下軌反轉機率

**模型 2: 趨勢過濾**
- 判斷當前趨勢強度
- 強趨勢時禁止交易

**決策邏輯**
```
做空: 上軌反轉機率 > 70%
     + 趨勢強度 < 30%
     
做多: 下軌反轉機率 > 70%
     + 趨勢強度 < 30%
```
---
""")

data_source = st.sidebar.radio(
    "資料源",
    ["HuggingFace (38幣)", "Binance API (即時)"],
    help="HuggingFace: 離線資料\nBinance: 即時資料"
)

if data_source == "HuggingFace (38幣)":
    loader = HuggingFaceKlineLoader()
    st.sidebar.success("使用 HuggingFace 離線資料")
else:
    loader = BinanceDataLoader()
    st.sidebar.info("使用 Binance 即時資料")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 使用流程

1. **BB 視覺化**: 觀察反轉模式
2. **反轉訓練**: 訓練上/下軌模型
3. **趨勢過濾**: 訓練過濾器
4. **歷史回測**: 測試策略
5. **實時監控**: 自動交易
6. **策略C**: 區間震盪策略
""")

# symbol_selector helper function
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

tabs = st.tabs([
    "1. BB 視覺化",
    "2. 反轉訓練",
    "3. 趨勢過濾",
    "4. 歷史回測",
    "5. 實時監控",
    "6. 策略C: 區間震盪"
])

with tabs[0]:
    render_bb_visualization_tab(loader)

with tabs[1]:
    render_reversal_training_tab(loader)

with tabs[2]:
    render_trend_filter_tab(loader)

with tabs[3]:
    render_backtest_tab(loader)

with tabs[4]:
    render_live_monitor_tab(loader)

with tabs[5]:
    render_range_bound_backtest_tab(loader, symbol_selector)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 策略參數
- BB 週期: 20
- BB 標準差: 2.0
- 反轉門檻: 70%
- 趨勢限制: 30%
- 時間框架: 15分鐘
""")