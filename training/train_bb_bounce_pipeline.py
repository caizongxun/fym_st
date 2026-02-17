#!/usr/bin/env python3
"""
BB反彈預測模型 - 完整訓練Pipeline

執行流程:
1. 載入歷史數據
2. 提取BB特徵 + ADX趨勢
3. 訓練上軌/下軌反彈模型
4. 保存模型
5. 評估效果
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.bb_bounce_features import BBBounceFeatureExtractor
from models.train_bb_bounce_model import BBBounceModelTrainer
from utils.data_fetcher import BinanceDataFetcher

def main():
    print("="*80)
    print("BB反彈預測模型 - 訓練Pipeline")
    print("="*80)
    print(f"\n開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # ===== 步驟1: 載入數據 =====
    print("步驟1: 載入歷史數據")
    print("-" * 80)
    
    symbol = 'BTCUSDT'
    interval = '15m'
    days_back = 60  # 訓練用60天數據
    
    print(f"  交易對: {symbol}")
    print(f"  週期: {interval}")
    print(f"  時間範圍: 最近{days_back}天")
    
    try:
        fetcher = BinanceDataFetcher()
        df = fetcher.fetch_historical_data(symbol, interval, days_back)
        print(f"\n  成功載入 {len(df)} 根K線")
        print(f"  時間範圍: {df.index[0]} 至 {df.index[-1]}")
    except Exception as e:
        print(f"\n錯誤: 無法載入數據 - {e}")
        print("\n請確認:")
        print("  1. 網路連接正常")
        print("  2. Binance API可訪問")
        print("  3. 或使用本地CSV數據:")
        print("     df = pd.read_csv('your_data.csv', parse_dates=['timestamp'])")
        return
    
    # ===== 步驟2: 提取特徵 =====
    print("\n" + "="*80)
    print("步驟2: 提取BB反彈特徵")
    print("-" * 80)
    
    extractor = BBBounceFeatureExtractor(
        bb_period=20,
        bb_std=2.0,
        adx_period=14,
        touch_threshold=0.3  # 距離軌道0.3σ內算觸碰
    )
    
    print("  計算布林通道...")
    print("  計算ADX趨勢指標...")
    print("  檢測觸碰點...")
    print("  提取30根K線特徵...")
    print("  生成訓練標籤...")
    
    df_processed = extractor.process(df, create_labels=True)
    
    # 檢查觸碰統計
    upper_touches = (df_processed['touch_upper'] == 1).sum()
    lower_touches = (df_processed['touch_lower'] == 1).sum()
    
    print(f"\n  特徵提取完成!")
    print(f"  觸碰上軌次數: {upper_touches} ({upper_touches/len(df)*100:.2f}%)")
    print(f"  觸碰下軌次數: {lower_touches} ({lower_touches/len(df)*100:.2f}%)")
    
    # 趨勢狀態分布
    print(f"\n  趨勢狀態分布:")
    trend_dist = df_processed['trend_state'].value_counts()
    for trend, count in trend_dist.items():
        print(f"    {trend}: {count} ({count/len(df)*100:.1f}%)")
    
    if upper_touches < 50 or lower_touches < 50:
        print("\n警告: 觸碰樣本數過少,建議:")
        print("  1. 增加訓練天數 (days_back > 60)")
        print("  2. 放寬觸碰閾值 (touch_threshold = 0.5)")
        print("  3. 使用1小時週期獲得更多樣本")
    
    # ===== 步驟3: 訓練模型 =====
    print("\n" + "="*80)
    print("步驟3: 訓練BB反彈預測模型")
    print("-" * 80)
    
    trainer = BBBounceModelTrainer(model_dir='models/saved')
    
    try:
        trainer.train_both_models(df_processed)
    except Exception as e:
        print(f"\n訓練失敗: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ===== 步驟4: 保存模型 =====
    print("\n" + "="*80)
    print("步驟4: 保存模型")
    print("-" * 80)
    
    trainer.save_models()
    
    # ===== 步驟5: 評估和建議 =====
    print("\n" + "="*80)
    print("步驟5: 訓練總結與建議")
    print("="*80)
    
    print("\n模型文件位置:")
    print("  - models/saved/bb_upper_bounce_model.pkl (上軌反彈,做空信號)")
    print("  - models/saved/bb_lower_bounce_model.pkl (下軌反彈,做多信號)")
    
    print("\n下一步操作:")
    print("  1. 回測驗證: python backtesting/run_bb_backtest.py")
    print("  2. 實時預測: 在app.py中整合signal_generator_bb.py")
    print("  3. 優化模型: 調整特徵或訓練參數")
    
    print("\n模型使用範例:")
    print("""
    from utils.signal_generator_bb import BBBounceSignalGenerator
    
    # 載入模型並生成信號
    signal_gen = BBBounceSignalGenerator()
    df_signals = signal_gen.generate_signals(df)
    
    # 查看信號
    signals = df_signals[df_signals['signal'] != 0]
    print(signals[['signal_name', 'bb_upper_bounce_prob', 
                   'bb_lower_bounce_prob', 'trend_state', 'rsi']])
    """)
    
    print(f"\n完成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == '__main__':
    main()