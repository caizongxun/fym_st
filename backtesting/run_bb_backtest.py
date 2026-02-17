#!/usr/bin/env python3
"""
BB反彈策略回測

使用訓練好的BB模型進行歷史回測
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.bb_bounce_features import BBBounceFeatureExtractor
from utils.signal_generator_bb import BBBounceSignalGenerator
from utils.data_fetcher import BinanceDataFetcher
from backtesting.engine import BacktestEngine

def main():
    print("="*80)
    print("BB反彈策略回測")
    print("="*80)
    print(f"\n開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # ===== 參數設定 =====
    symbol = 'BTCUSDT'
    interval = '15m'
    days_back = 30  # 回測30天(與訓練時期不重疊)
    
    initial_capital = 100.0  # USDT
    position_size_pct = 1.0  # 100%倉位
    
    # 止盈止損設定
    tp_pct = 0.6  # 0.6% 止盈
    sl_pct = 0.6  # 0.6% 止損
    
    print(f"回測參數:")
    print(f"  交易對: {symbol}")
    print(f"  週期: {interval}")
    print(f"  時間: 最近{days_back}天")
    print(f"  初始資金: ${initial_capital}")
    print(f"  止盈/止損: {tp_pct}% / {sl_pct}%")
    
    # ===== 載入數據 =====
    print("\n" + "-"*80)
    print("步驟1: 載入回測數據")
    print("-"*80)
    
    try:
        fetcher = BinanceDataFetcher()
        df = fetcher.fetch_historical_data(symbol, interval, days_back)
        print(f"  成功載入 {len(df)} 根K線")
        print(f"  時間範圍: {df.index[0]} 至 {df.index[-1]}")
    except Exception as e:
        print(f"\n錯誤: {e}")
        return
    
    # ===== 生成信號 =====
    print("\n" + "-"*80)
    print("步驟2: 生成BB反彈交易信號")
    print("-"*80)
    
    try:
        signal_gen = BBBounceSignalGenerator(
            bb_model_dir='models/saved',
            bb_bounce_threshold=0.60,
            adx_strong_trend_threshold=30
        )
        
        df_signals = signal_gen.generate_signals(df)
        signal_gen.print_signal_summary(df_signals)
        
    except FileNotFoundError as e:
        print(f"\n錯誤: 模型文件不存在")
        print(f"  {e}")
        print("\n請先執行訓練:")
        print("  python training/train_bb_bounce_pipeline.py")
        return
    except Exception as e:
        print(f"\n錯誤: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 檢查是否有信號
    total_signals = (df_signals['signal'] != 0).sum()
    if total_signals == 0:
        print("\n警告: 沒有生成任何交易信號")
        print("\n可能原因:")
        print("  1. 模型預測反彈機率都 < 60%")
        print("  2. ADX過濾過於嚴格 (大部分為強趨勢)")
        print("  3. RSI沒有達到超買/超賣")
        print("  4. 時間區間太短")
        print("\n建議:")
        print("  - 降低bb_bounce_threshold至60% (如50%)")
        print("  - 增加回測天數 (days_back > 30)")
        print("  - 查看df_signals的bb_*_bounce_prob分布")
        return
    
    # ===== 執行回測 =====
    print("\n" + "-"*80)
    print("步驟3: 執行回測")
    print("-"*80)
    
    engine = BacktestEngine(
        initial_capital=initial_capital,
        fee_rate=0.001,  # 0.1%
        slippage=0.0005  # 0.05%
    )
    
    results = engine.run(
        df_signals,
        position_size_pct=position_size_pct,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        use_trailing_stop=False
    )
    
    # ===== 顯示結果 =====
    print("\n" + "="*80)
    print("回測結果")
    print("="*80)
    
    metrics = results['metrics']
    trades = results['trades']
    
    print(f"\n績效指標:")
    print(f"  總交易次數: {metrics['total_trades']}")
    print(f"  勝率: {metrics['win_rate']:.2f}%")
    print(f"  最終權益: ${metrics['final_equity']:.2f}")
    print(f"  總回報: {metrics['total_return']:.2f}%")
    print(f"  獲利因子: {metrics['profit_factor']:.2f}")
    print(f"  夏普比率: {metrics['sharpe_ratio']:.2f}")
    print(f"  最大回撤: {metrics['max_drawdown']:.2f}%")
    print(f"  平均持倉時長: {metrics['avg_duration']:.0f}分鐘")
    
    # 離場原因分布
    if len(trades) > 0:
        print(f"\n離場原因分布:")
        exit_reasons = trades['exit_reason'].value_counts()
        for reason, count in exit_reasons.items():
            print(f"  {reason}: {count} ({count/len(trades)*100:.1f}%)")
        
        # 各離場原因績效
        print(f"\n各離場原因績效:")
        for reason in exit_reasons.index:
            subset = trades[trades['exit_reason'] == reason]
            win_rate = (subset['pnl'] > 0).sum() / len(subset) * 100
            avg_pnl = subset['pnl'].mean()
            total_pnl = subset['pnl'].sum()
            print(f"  {reason}: 勝率{win_rate:.1f}% | 平均{avg_pnl:.2f}U | 總計{total_pnl:.2f}U")
    
    # ===== 保存結果 =====
    print("\n" + "-"*80)
    print("保存回測結果")
    print("-"*80)
    
    timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    output_file = f'backtesting/results/bb_backtest_{symbol}_{timestamp}.csv'
    
    os.makedirs('backtesting/results', exist_ok=True)
    trades.to_csv(output_file, index=False)
    
    print(f"  交易記錄已保存: {output_file}")
    
    # ===== 總結 =====
    print("\n" + "="*80)
    print("總結與建議")
    print("="*80)
    
    if metrics['profit_factor'] < 1.0:
        print("\n⚠️  獲利因子 < 1.0, 策略需要優化")
        print("\n優化建議:")
        print("  1. 提高bb_bounce_threshold (如提高到70%)")
        print("  2. 加入反轉模型雙重確認")
        print("  3. 調整止盈止損比例 (1:2)")
        print("  4. 只在特定趨勢狀態交易 (ranging/weak_trend)")
        print("  5. 增加ADX過濾強度")
    elif metrics['profit_factor'] > 1.5:
        print("\n✅ 獲利因子 > 1.5, 策略表現良好!")
        print("\n下一步:")
        print("  1. 在更長時間區間測試 (90天+)")
        print("  2. 測試其他幣種 (ETHUSDT, BNBUSDT)")
        print("  3. 測試不同週期 (1h, 4h)")
        print("  4. 準備實盤 (Paper Trading)")
    else:
        print("\n🟡 獲利因子 1.0-1.5, 策略有潛力")
        print("\n優化建議:")
        print("  1. 微調闾值參數")
        print("  2. 加入更多過濾條件")
        print("  3. 優化出場策略 (移動止盈)")
    
    print(f"\n完成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == '__main__':
    main()