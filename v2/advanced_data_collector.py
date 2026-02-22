import pandas as pd
import numpy as np
import requests
import time
import io
import zipfile
from datetime import datetime
from typing import Dict, Optional
import os


# ============================================================
# 收集策略
# ------------------------------------------------------------
# funding_rate (fapi/v1/fundingRate) — 完整歷史 2019～今
#   每 8 小時一筆，BTC 約 7,000+ 筆
#   是唯一具備完整歷史的進階特徵
#
# 不收集 (Binance API 僅保留 30 天，訓練時會引發時間偏差):
#   openInterestHist / topLongShortAccountRatio / takerlongshortRatio
# ============================================================


class BinanceAdvancedDataCollector:

    def __init__(self):
        self.futures_base_url = 'https://fapi.binance.com'
        self.rate_limit_delay = 0.3

    def get_earliest_available_time(self, symbol: str) -> int:
        """從對照表取得幣種期貨上線日 (ms timestamp)"""
        earliest_dates = {
            'BTCUSDT':   '2019-09-08',
            'ETHUSDT':   '2020-02-12',
            'BNBUSDT':   '2020-04-09',
            'XRPUSDT':   '2020-11-13',
            'DOTUSDT':   '2021-01-14',
            'LINKUSDT':  '2021-01-14',
            'ADAUSDT':   '2021-03-10',
            'MATICUSDT': '2021-05-09',
            'SOLUSDT':   '2021-08-11',
            'AVAXUSDT':  '2021-11-24',
        }
        date_str = earliest_dates.get(symbol, '2021-01-01')
        return int(pd.to_datetime(date_str).timestamp() * 1000)

    # ----------------------------------------------------------
    # 資金費率 — 完整歷史，從最早時間往後爬
    # ----------------------------------------------------------
    def get_funding_rate(self, symbol: str, start_time: int, limit: int = 1000) -> pd.DataFrame:
        """
        爬取完整歷史資金費率
        - 用 startTime + 循環往後爬 (正確方向)
        - 直到 len(data) < limit 則停止
        """
        url = f"{self.futures_base_url}/fapi/v1/fundingRate"
        all_funding = []
        current_start = start_time
        round_count = 0

        start_label = pd.to_datetime(start_time, unit='ms').strftime('%Y-%m-%d')
        print(f"  爬取資金費率 (從 {start_label})...")

        try:
            while True:
                params = {
                    'symbol': symbol,
                    'startTime': current_start,
                    'limit': limit
                }
                resp = requests.get(url, params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()

                if not data:
                    break

                all_funding.extend(data)
                round_count += 1
                if round_count % 5 == 0:
                    print(f"    已爬取 {len(all_funding):,} 筆...")

                if len(data) < limit:
                    break

                current_start = data[-1]['fundingTime'] + 1
                time.sleep(self.rate_limit_delay)

            if not all_funding:
                print(f"  ⚠️ 無資金費率數據")
                return pd.DataFrame()

            df = pd.DataFrame(all_funding)
            df['timestamp'] = pd.to_datetime(df['fundingTime'], unit='ms')
            df['fundingRate'] = df['fundingRate'].astype(float)
            df = df[['timestamp', 'fundingRate']].sort_values('timestamp').reset_index(drop=True)

            # 衍生特徵
            df['funding_rate_ma8']     = df['fundingRate'].rolling(8).mean()
            df['funding_rate_ma24']    = df['fundingRate'].rolling(24).mean()
            df['funding_rate_std']     = df['fundingRate'].rolling(24).std()
            df['funding_rate_extreme'] = (abs(df['fundingRate']) > df['funding_rate_std'] * 2).astype(int)
            df['fr_roc_8h']            = df['fundingRate'].diff(1)
            df['fr_roc_24h']           = df['fundingRate'].diff(3)
            df['fr_cumsum_7d']         = df['fundingRate'].rolling(21).sum()
            df['fr_positive']          = (df['fundingRate'] > 0).astype(int)
            df['fr_extreme_pos']       = (df['fundingRate'] >  0.001).astype(int)
            df['fr_extreme_neg']       = (df['fundingRate'] < -0.001).astype(int)
            df.fillna(0, inplace=True)

            t_min = df['timestamp'].min().strftime('%Y-%m-%d')
            t_max = df['timestamp'].max().strftime('%Y-%m-%d')
            print(f"  ✅ 共 {len(df):,} 筆 (從 {t_min} 至 {t_max})")
            return df

        except Exception as e:
            print(f"  ⚠️ 資金費率無法獲取: {e}")
            return pd.DataFrame()


class BatchAdvancedDataCollector:
    """
    批量收集所有幣種的資金費率歷史數據
    將每個幣種儲存為 {SYMBOL}_funding_rate.parquet
    """
    def __init__(self):
        self.collector = BinanceAdvancedDataCollector()
        self.hf_symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
            'ADAUSDT', 'AVAXUSDT', 'DOTUSDT', 'MATICUSDT', 'LINKUSDT',
            'UNIUSDT', 'LTCUSDT', 'ETCUSDT', 'XLMUSDT', 'ATOMUSDT',
            'FILUSDT', 'NEARUSDT', 'ALGOUSDT', 'VETUSDT', 'ICPUSDT',
            'APTUSDT', 'ARBUSDT', 'OPUSDT',  'INJUSDT', 'SUIUSDT',
            'PEPEUSDT', 'WIFUSDT', 'SHIBUSDT', 'DOGEUSDT', 'TRXUSDT',
            'TONUSDT', 'HBARUSDT', 'RENDERUSDT', 'FTMUSDT', 'AAVEUSDT',
            'RUNEUSDT', 'IMXUSDT', 'LDOUSDT'
        ]

    def collect_all_symbols(
        self,
        output_dir: str = 'v2/advanced_data'
    ) -> pd.DataFrame:
        """
        批量爬取所有幣種的完整歷史資金費率
        儲存: {output_dir}/{SYMBOL}_funding_rate.parquet
        """
        os.makedirs(output_dir, exist_ok=True)
        summary = []

        print(f"\n{'='*60}")
        print(f"批量資金費率收集")
        print(f"幣種數量: {len(self.hf_symbols)}")
        print(f"輸出目錄: {output_dir}")
        print(f"{'='*60}\n")

        for idx, symbol in enumerate(self.hf_symbols, 1):
            print(f"\n[{idx}/{len(self.hf_symbols)}] {symbol}")
            try:
                start_time = self.collector.get_earliest_available_time(symbol)
                df = self.collector.get_funding_rate(symbol, start_time)

                if not df.empty:
                    filepath = os.path.join(output_dir, f"{symbol}_funding_rate.parquet")
                    df.to_parquet(filepath, index=False)
                    print(f"  💾 儲存: {filepath} ({len(df):,} 筆)")
                    summary.append({
                        'symbol': symbol,
                        'records': len(df),
                        'from': df['timestamp'].min().strftime('%Y-%m-%d'),
                        'to': df['timestamp'].max().strftime('%Y-%m-%d'),
                        'status': 'success'
                    })
                else:
                    summary.append({'symbol': symbol, 'records': 0,
                                    'from': '-', 'to': '-', 'status': 'empty'})

                time.sleep(1)

            except Exception as e:
                print(f"  ❌ 錯誤: {e}")
                summary.append({'symbol': symbol, 'records': 0,
                                'from': '-', 'to': '-', 'status': f'failed: {str(e)[:30]}'})

        summary_df = pd.DataFrame(summary)
        summary_path = os.path.join(output_dir, 'collection_summary.csv')
        summary_df.to_csv(summary_path, index=False)

        print(f"\n{'='*60}")
        print(f"收集完成")
        success = (summary_df['status'] == 'success').sum()
        total_records = summary_df['records'].sum()
        print(f"成功: {success}/{len(self.hf_symbols)} 個幣種")
        print(f"總筆數: {total_records:,} 筆")
        print(f"{'='*60}\n")

        return summary_df


if __name__ == '__main__':
    BatchAdvancedDataCollector().collect_all_symbols(
        output_dir='v2/advanced_data'
    )
