#!/usr/bin/env python3
"""
流動性掃蕩系統測試
"""

import sys
import os
sys.path.insert(0, 'trading_system')

from core import CryptoDataLoader, LiquiditySweepDetector
import pandas as pd

def test_liquidity_sweep():
    print("="*80)
    print("流動性掃蕩系統測試")
    print("="*80)
    
    # 1. 載入數據 (包含 OI 和資金費率)
    print("\n[1] 載入 BTCUSDT 1h 數據 (30天, 包含 OI & Funding Rate)...")
    loader = CryptoDataLoader()
    
    df = loader.fetch_latest_klines(
        symbol='BTCUSDT',
        timeframe='1h',
        days=30,
        include_oi=True,
        include_funding=True
    )
    
    print(f"   載入 {len(df)} 筆 K 線")
    print(f"   時間範圍: {df['open_time'].min()} ~ {df['open_time'].max()}")
    
    # 檢查欄位
    print(f"\n   欄位: {list(df.columns)}")
    
    if 'open_interest' in df.columns:
        oi_valid = df['open_interest'].notna().sum()
        print(f"   OI 有效數據: {oi_valid}/{len(df)} ({100*oi_valid/len(df):.1f}%)")
    
    if 'funding_rate' in df.columns:
        fr_valid = df['funding_rate'].notna().sum()
        print(f"   Funding Rate 有效數據: {fr_valid}/{len(df)} ({100*fr_valid/len(df):.1f}%)")
    
    # 2. 偵測流動性掃蕩
    print("\n[2] 偵測做多信號 (掃蕩低點)...")
    detector = LiquiditySweepDetector(
        lookback_period=50,
        wick_multiplier=2.0,
        oi_std_threshold=2.0,
        cvd_divergence_lookback=10
    )
    
    df_sweep = detector.detect_liquidity_sweep(df, direction='lower')
    
    # 3. 計算特徵
    print("\n[3] 計算流動性掃蕩特徵...")
    df_sweep = detector.calculate_sweep_features(df_sweep)
    
    # 4. 分析結果
    print("\n[4] 結果分析")
    print("="*80)
    
    # 分解每個步驟的築選
    has_wick = df_sweep['sweep_lower_wick'].sum()
    has_breach = df_sweep['sweep_lower_breach'].sum()
    has_oi_flush = df_sweep['sweep_lower_oi_flush'].sum()
    has_cvd_div = df_sweep['sweep_lower_cvd_div'].sum()
    final_signals = df_sweep['sweep_lower_signal'].sum()
    
    print(f"\n築選漏斗:")
    print(f"   長下影線:       {has_wick}/{len(df)} ({100*has_wick/len(df):.1f}%)")
    print(f"   突破支撑位:     {has_breach}/{len(df)} ({100*has_breach/len(df):.1f}%)")
    print(f"   OI 銳減:         {has_oi_flush}/{len(df)} ({100*has_oi_flush/len(df):.1f}%)")
    print(f"   CVD 背離:        {has_cvd_div}/{len(df)} ({100*has_cvd_div/len(df):.1f}%)")
    print(f"   最終信號:       {final_signals}/{len(df)} ({100*final_signals/len(df):.1f}%)")
    
    # 顯示信號詳情
    if final_signals > 0:
        print(f"\n\n流動性掃蕩信號詳情:")
        print("="*80)
        
        signals = df_sweep[df_sweep['sweep_lower_signal']].copy()
        
        display_cols = [
            'open_time', 'close', 'low', 'lower_wick_ratio',
            'oi_change_pct', 'cvd_slope', 'dist_to_support_pct'
        ]
        
        available_cols = [col for col in display_cols if col in signals.columns]
        
        for idx, row in signals[available_cols].iterrows():
            print(f"\n時間: {row['open_time']}")
            print(f"  價格: ${row['close']:.2f} (Low: ${row['low']:.2f})")
            print(f"  下影線比: {row['lower_wick_ratio']:.2f}x")
            if 'oi_change_pct' in row:
                print(f"  OI 變化: {row['oi_change_pct']*100:.2f}%")
            if 'cvd_slope' in row:
                print(f"  CVD 斜率: {row['cvd_slope']:.0f}")
            if 'dist_to_support_pct' in row:
                print(f"  距支撑: {row['dist_to_support_pct']:.2f}%")
    else:
        print("\n未偵測到流動性掃蕩信號")
        print("\n建議:")
        print("  1. 增加回測天數 (30 → 90)")
        print("  2. 調低 wick_multiplier (2.0 → 1.5)")
        print("  3. 調低 oi_std_threshold (2.0 → 1.5)")
    
    # 5. 儲存結果
    output_file = 'liquidity_sweep_test_results.csv'
    df_sweep.to_csv(output_file, index=False)
    print(f"\n\n結果已儲存至: {output_file}")
    print("="*80)

if __name__ == "__main__":
    test_liquidity_sweep()