#!/usr/bin/env python3
"""
下一根K棒高低點預測 - 快速測試
"""

import sys
sys.path.append('.')

from data.huggingface_loader import HuggingFaceKlineLoader
from models.train_nextbar_model import NextBarModelTrainer
import pandas as pd

def main():
    print("====== 下一根K棒高低點預測 ======\n")
    
    # 載入數據
    symbol = 'BTCUSDT'
    print(f"載入 {symbol} 15m K線...")
    loader = HuggingFaceKlineLoader()
    df = loader.load_klines(symbol, '15m')
    print(f"載入完成: {len(df)} 根K線\n")
    
    # 訓練模型
    trainer = NextBarModelTrainer(model_type='xgboost')
    
    print("開始訓練...\n")
    metrics = trainer.train_with_oos(
        df,
        max_range_pct=0.015,  # 過濾 > 1.5% 的異常波動
        oos_days=30,
        test_size=0.2
    )
    
    # 分析結果
    print("\n====== 結果分析 ======\n")
    
    high_mae_pct = metrics['high_oos_mae'] * 100
    low_mae_pct = metrics['low_oos_mae'] * 100
    range_mae_pct = metrics['range_oos_mae'] * 100
    
    print(f"OOS測試結果:")
    print(f"  HIGH 預測誤差: {high_mae_pct:.3f}%")
    print(f"  LOW 預測誤差: {low_mae_pct:.3f}%")
    print(f"  區間預測誤差: {range_mae_pct:.3f}%")
    
    print(f"\n評估:")
    if high_mae_pct < 0.20 and low_mae_pct < 0.20:
        print("  優秀! 預測誤差 < 0.20%, 可以用於實盤交易")
    elif high_mae_pct < 0.30 and low_mae_pct < 0.30:
        print("  合格! 預測誤差 < 0.30%, 建議先回測驗證")
    else:
        print("  需要優化! 預測誤差偏高")
        print("  建議:")
        print("    1. 增加訓練數據")
        print("    2. 調整 max_range_pct 過濾參數")
        print("    3. 嘗試 lightgbm 模型")
    
    # 顯示重要特徵
    print("\n====== TOP 10 重要特徵 (HIGH) ======")
    print(metrics['feature_importance_high'].head(10).to_string(index=False))
    
    print("\n====== TOP 10 重要特徵 (LOW) ======")
    print(metrics['feature_importance_low'].head(10).to_string(index=False))
    
    # 保存模型
    print("\n保存模型...")
    trainer.save_model(symbol, prefix='v1')
    
    print("\n完成!")
    print("\n下一步: 使用此模型進行回測")
    print("  命令: python test_nextbar_backtest.py")

if __name__ == '__main__':
    main()