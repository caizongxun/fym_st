import streamlit as st
from data.binance_loader import BinanceDataLoader
from data.huggingface_loader import HuggingFaceKlineLoader

from tabs import (
    render_bb_visualization_tab,
    render_reversal_training_tab,
    render_trend_filter_tab,
    render_backtest_tab,
    render_live_monitor_tab,
    render_range_bound_backtest_tab,
    render_ml_strategy_d_tab
)

st.set_page_config(
    page_title="BB Reversal Trading System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("BB Reversal Trading System")
st.caption("Goal: Precisely predict BB band reversal points, filter strong trend false signals")

st.sidebar.title("System Settings")

st.sidebar.markdown("""
### Dual Model Architecture

**Model 1: BB Reversal Prediction**
- Upper band model: Predict upper band reversal probability
- Lower band model: Predict lower band reversal probability

**Model 2: Trend Filter**
- Determine current trend strength
- Prohibit trading during strong trends

**Decision Logic**
```
Short: Upper reversal prob > 70%
       + Trend strength < 30%
     
Long: Lower reversal prob > 70%
      + Trend strength < 30%
```
---
""")

data_source = st.sidebar.radio(
    "Data Source",
    ["HuggingFace (38 symbols)", "Binance API (Live)"],
    help="HuggingFace: Offline data\nBinance: Live data"
)

if data_source == "HuggingFace (38 symbols)":
    loader = HuggingFaceKlineLoader()
    st.sidebar.success("Using HuggingFace offline data")
else:
    loader = BinanceDataLoader()
    st.sidebar.info("Using Binance live data")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### Workflow

1. **BB Visualization**: Observe reversal patterns
2. **Reversal Training**: Train upper/lower band models
3. **Trend Filter**: Train filter model
4. **Backtest**: Test strategy
5. **Live Monitor**: Auto trading
6. **Strategy C**: Range-bound trading
7. **Strategy D**: ML-driven with tick-level backtest
""")

# symbol_selector helper function
def symbol_selector(key_prefix: str, multi: bool = False, default_symbols: list = None):
    if isinstance(loader, HuggingFaceKlineLoader):
        symbol_groups = HuggingFaceKlineLoader.get_symbol_groups()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selection_mode = st.radio(
                "Selection Mode",
                ["Top 10", "By Category", "Manual"],
                key=f"{key_prefix}_mode"
            )
        
        with col2:
            if selection_mode == "Top 10":
                top_symbols = HuggingFaceKlineLoader.get_top_symbols(10)
                if multi:
                    selected = st.multiselect(
                        "Select Symbols",
                        top_symbols,
                        default=default_symbols or top_symbols[:2],
                        key=f"{key_prefix}_top"
                    )
                else:
                    selected = [st.selectbox(
                        "Select Symbol",
                        top_symbols,
                        key=f"{key_prefix}_top_single"
                    )]
            
            elif selection_mode == "By Category":
                category = st.selectbox(
                    "Select Category",
                    list(symbol_groups.keys()),
                    key=f"{key_prefix}_category"
                )
                symbols_in_category = symbol_groups[category]
                
                if multi:
                    selected = st.multiselect(
                        f"{category} Symbols",
                        symbols_in_category,
                        default=default_symbols or symbols_in_category[:2],
                        key=f"{key_prefix}_cat_multi"
                    )
                else:
                    selected = [st.selectbox(
                        f"{category} Symbols",
                        symbols_in_category,
                        key=f"{key_prefix}_cat_single"
                    )]
            
            else:
                if multi:
                    text_input = st.text_area(
                        "Enter symbols (comma separated)",
                        value=",".join(default_symbols) if default_symbols else "BTCUSDT,ETHUSDT",
                        key=f"{key_prefix}_manual",
                        height=100
                    )
                    selected = [s.strip().upper() for s in text_input.split(',') if s.strip()]
                else:
                    selected = [st.text_input(
                        "Enter symbol",
                        value="BTCUSDT",
                        key=f"{key_prefix}_manual_single"
                    ).strip().upper()]
        
        return selected
    
    else:
        if multi:
            text_input = st.text_area(
                "Trading pairs (comma separated)",
                value="BTCUSDT,ETHUSDT",
                key=f"{key_prefix}_binance"
            )
            return [s.strip().upper() for s in text_input.split(',') if s.strip()]
        else:
            return [st.text_input(
                "Trading pair",
                value="BTCUSDT",
                key=f"{key_prefix}_binance_single"
            ).strip().upper()]

tabs = st.tabs([
    "1. BB Visualization",
    "2. Reversal Training",
    "3. Trend Filter",
    "4. Backtest",
    "5. Live Monitor",
    "6. Strategy C",
    "7. Strategy D (ML)"
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

with tabs[6]:
    render_ml_strategy_d_tab(loader, symbol_selector)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### Strategy Parameters
- BB Period: 20
- BB Std Dev: 2.0
- Reversal Threshold: 70%
- Trend Limit: 30%
- Timeframe: 15min
""")
