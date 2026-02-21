import streamlit as st
import pandas as pd
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

def render():
    # 標題區
    st.markdown("""
    <div style='text-align: center; padding: 40px 0;'>
        <h1 style='font-size: 3.5em; margin: 0;'>🎯</h1>
        <h1 style='color: #1f77b4; margin: 10px 0;'>BB + NW 波段反轉交易系統</h1>
        <p style='font-size: 1.2em; color: #7f7f7f;'>Bollinger Bands + Nadaraya-Watson Swing Reversal Trading System</p>
        <p style='color: #4CAF50; font-weight: bold;'>v2.0 - Institutional Grade</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ===== 系統概述 =====
    st.markdown("## 🌟 系統核心特色")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 觸發層
        **雙通道辨識**
        
        - ✅ Bollinger Bands (BB)
        - ✅ Nadaraya-Watson (NW)
        - 🔒 無未來函數 (No Repaint)
        
        **只在觸碸軌道時啟動**
        節省 85-98% 運算資源
        """)
    
    with col2:
        st.markdown("""
        ### ⚙️ 特徵層
        **機構級特徵**
        
        - 🌊 ADX 趨勢強度指標
        - 📈 CVD 背離 (流動性獵取)
        - 💨 VWWA 影線吸收率
        - 🌌 HTF 趨勢過濾 (1h)
        
        **兩大防禁機制**
        """)
    
    with col3:
        st.markdown("""
        ### 🤖 AI 層
        **Meta-Labeling**
        
        - 🏆 LightGBM / XGBoost
        - 🔁 5-Fold CV
        - ⏱️ Early Stopping
        
        **判斷「真反彈」VS「假反彈」**
        """)
    
    st.markdown("---")
    
    # ===== 核心優勢 =====
    st.markdown("## 🛡️ 核心防禁機制")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🚨 防止單邊趨勢輾壓
        
        **問題**: 價格觸碸 BB 下軌，但處於主跌浪，繼續下跌
        
        **解決方案**:
        1. **ADX 過濾**: ADX > 25 且持續上升 → 走勢中，不做反轉
        2. **HTF EMA 過濾**: 1h 級別價格遠離 EMA_50 → 強趨勢，降低反彈機率
        3. **趨勢風險評分**: `trend_crush_risk` 特徵自動計算
        
        📉 **效果**: 模型會在強趨勢中輸出極低機率 (< 0.20)
        """)
    
    with col2:
        st.markdown("""
        ### 🌊 辨識獵取流動性
        
        **問題**: 機構用長下影線刺穿下軌，掃掉散戶止損後拉盤
        
        **解決方案**:
        1. **CVD 背離**: 價格新低，但 CVD 未跟隨 → 機構接盤
        2. **VWWA 吸收**: 下影線長 × 爆量 → 流動性被抽乾
        3. **背離評分**: `sweep_divergence_buy` 特徵自動抓取
        
        📈 **效果**: 模型會在獲取流動性時輸出極高機率 (> 0.80)
        """)
    
    st.markdown("---")
    
    # ===== 使用流程 =====
    st.markdown("## 📌 快速開始流程")
    
    steps = [
        {
            'icon': '🧪',
            'title': '步驟 1: 模型訓練',
            'desc': '前往「模型訓練」頁面，選擇 BTCUSDT，啟用 BB+NW+ADX+CVD 特徵',
            'time': '~10-15 分鐘'
        },
        {
            'icon': '📊',
            'title': '步驟 2: 回測驗證',
            'desc': '前往「回測分析」，使用 2024 OOS 數據驗證，目標勝率 55-65%',
            'time': '~5 分鐘'
        },
        {
            'icon': '⚙️',
            'title': '步驟 3: 參數優化',
            'desc': '調整機率門檻、TP/SL 比例、持倉時間，追求最佳帏率比',
            'time': '~20 分鐘'
        },
        {
            'icon': '📡',
            'title': '步驟 4: 實時預測',
            'desc': '前往「即時預測」，連接 Binance API，監控 15m K線觸碸事件',
            'time': '實時'
        }
    ]
    
    for step in steps:
        with st.container():
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown(f"<h1 style='text-align: center; font-size: 3em;'>{step['icon']}</h1>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"**{step['title']}** `{step['time']}`")
                st.markdown(step['desc'])
            st.markdown("---")
    
    # ===== 系統狀態 =====
    st.markdown("## 📊 系統狀態")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # 檢查模型數量
        models_dir = "models"
        model_count = 0
        if os.path.exists(models_dir):
            model_count = len([f for f in os.listdir(models_dir) if f.endswith('.pkl')])
        
        st.metric(
            "🤖 已訓練模型",
            model_count,
            "Ready" if model_count > 0 else "None"
        )
    
    with col2:
        # 檢查數據庫
        data_dir = "data"
        data_available = os.path.exists(data_dir)
        
        st.metric(
            "💾 數據庫狀態",
            "可用" if data_available else "空",
            "HuggingFace" if data_available else "None"
        )
    
    with col3:
        st.metric(
            "📡 API 連接",
            "待測試",
            "Binance"
        )
    
    with col4:
        st.metric(
            "⌛ 系統運行時間",
            datetime.now().strftime("%H:%M"),
            datetime.now().strftime("%Y-%m-%d")
        )
    
    st.markdown("---")
    
    # ===== 技術規格 =====
    st.markdown("## 🔧 技術規格")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 特徵工程
        
        | 模組 | 特徵數 | 說明 |
        |------|--------|------|
        | BB 通道 | 5 | 中軌、上下軌、寬度、位置 |
        | NW 包絡線 | 4 | 中軌、上下軌、寬度 |
        | ADX 趨勢 | 3 | ADX、+DI、-DI |
        | CVD 流動性 | 6 | CVD_10、CVD_20、標準化、背離 |
        | VWWA | 2 | 上下影線吸收率 |
        | 反轉共振 | 8 | 刺穿深度、趨勢風險、擠壓 |
        | MTF (1h) | 50+ | 1h 級別所有特徵 |
        
        **總計**: ~80-100 個特徵
        """)
    
    with col2:
        st.markdown("""
        ### 🤖 模型參數
        
        | 參數 | 建議值 | 說明 |
        |------|----------|------|
        | 模型類型 | LightGBM | 速度快，效果佳 |
        | CV Folds | 5 | 5 折交叉驗證 |
        | Early Stop | 50 | 防止過括合 |
        | TP 倍數 | 2.5-3.5 | 波段交易用更大 TP |
        | SL 倍數 | 0.75-1.25 | 緊止損 |
        | 機率門檻 | 0.60 | > 60% 才進場 |
        | 持倉時間 | 10-20h | 15m × 40-80 |
        
        **目標**: 勝率55-65%，帏率比 2.5:1+
        """)
    
    st.markdown("---")
    
    # ===== 重要說明 =====
    st.markdown("## ⚠️ 重要說明")
    
    st.warning("""
    **本系統為教育與研究用途**
    
    1. **資金風險**: 加密貨幣交易具有極高風險，可能導致全部資金損失
    2. **無擔保**: 系統不擔保任何盈利，歷史績效不代表未來表現
    3. **自行責任**: 所有交易決策由使用者自行負責
    4. **建議**: 先在模擬盤充分測試，再考慮實盤
    """)
    
    st.markdown("---")
    
    # ===== 資源連結 =====
    st.markdown("## 🔗 資源連結")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📚 學習資源**
        - [Bollinger Bands](https://www.investopedia.com/terms/b/bollingerbands.asp)
        - [Nadaraya-Watson](https://en.wikipedia.org/wiki/Kernel_regression)
        - [ADX Indicator](https://www.investopedia.com/terms/a/adx.asp)
        """)
    
    with col2:
        st.markdown("""
        **🛠️ 技術文檔**
        - [LightGBM](https://lightgbm.readthedocs.io/)
        - [Meta-Labeling](https://www.quantstart.com/articles/meta-labeling/)
        - [Triple Barrier](https://mlfinlab.readthedocs.io/en/latest/labeling/tb_meta_labeling.html)
        """)
    
    with col3:
        st.markdown("""
        **📊 市場數據**
        - [Binance API](https://binance-docs.github.io/apidocs/)
        - [HuggingFace Datasets](https://huggingface.co/datasets)
        - [CryptoQuant](https://cryptoquant.com/)
        """)
    
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #7f7f7f;'>BB+NW Swing Reversal System v2.0 | Built with ❤️ for Traders</p>", unsafe_allow_html=True)