import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import os


# ============================================================
# Binance API 歷史數據保留期限制 (無法繞過)
# ------------------------------------------------------------
# funding_rate (fapi/v1/fundingRate):   完整歷史 (2019 至今)
# open_interest (openInterestHist):     僅最近 30 天
# long_short_ratio (topLongShort...):   僅最近 30 天
# taker_buy_sell (takerlongshortRatio): 僅最近 30 天
# ============================================================


class BinanceAdvancedDataCollector:

    def __init__(self):
        self.spot_base_url = 'https://api.binance.com'
        self.futures_base_url = 'https://fapi.binance.com'
        self.rate_limit_delay = 0.3

    def get_earliest_available_time(self, symbol: str) -> int:
        """從對照表取得幣種期貨上線日"""
        earliest_dates = {
            'BTCUSDT': '2019-09-08',
            'ETHUSDT': '2020-02-12',
            'BNBUSDT': '2020-04-09',
            'ADAUSDT': '2021-03-10',
            'SOLUSDT': '2021-08-11',
            'XRPUSDT': '2020-11-13',
            'DOTUSDT': '2021-01-14',
            'AVAXUSDT': '2021-11-24',
            'MATICUSDT': '2021-05-09',
            'LINKUSDT': '2021-01-14',
        }
        date_str = earliest_dates.get(symbol, '2021-01-01')
        return int(pd.to_datetime(date_str).timestamp() * 1000)

    # ----------------------------------------------------------
    # 1. 資金費率 (完整歷史) — 使用 startTime 往後爬
    # ----------------------------------------------------------
    def get_funding_rate(self, symbol: str, start_time: int, limit: int = 1000) -> pd.DataFrame:
        url = f"{self.futures_base_url}/fapi/v1/fundingRate"
        all_funding = []
        current_start = start_time
        round_count = 0

        print(f"  爬取資金費率完整歷史 (從 {pd.to_datetime(start_time, unit='ms').strftime('%Y-%m-%d')})...")

        try:
            while True:
                params = {'symbol': symbol, 'startTime': current_start, 'limit': limit}
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
                return pd.DataFrame()

            df = pd.DataFrame(all_funding)
            df['timestamp'] = pd.to_datetime(df['fundingTime'], unit='ms')
            df['fundingRate'] = df['fundingRate'].astype(float)
            df = df[['timestamp', 'fundingRate']].sort_values('timestamp').reset_index(drop=True)

            df['funding_rate_ma8'] = df['fundingRate'].rolling(8).mean()
            df['funding_rate_ma24'] = df['fundingRate'].rolling(24).mean()
            df['funding_rate_std'] = df['fundingRate'].rolling(24).std()
            df['funding_rate_extreme'] = (abs(df['fundingRate']) > df['funding_rate_std'] * 2).astype(int)
            df.fillna(0, inplace=True)

            print(f"    ✅ 共 {len(df):,} 筆 (從 {df['timestamp'].min().strftime('%Y-%m-%d')} 至 {df['timestamp'].max().strftime('%Y-%m-%d')})")
            return df

        except Exception as e:
            print(f"  ⚠️ 資金費率無法獲取: {e}")
            return pd.DataFrame()

    # ----------------------------------------------------------
    # 2. 未平倉量 (僅最近 30 天) — 不傳 startTime
    # ----------------------------------------------------------
    def get_open_interest(self, symbol: str, interval: str = '15m', limit: int = 500) -> pd.DataFrame:
        url = f"{self.futures_base_url}/futures/data/openInterestHist"
        all_oi = []
        round_count = 0

        print(f"  爬取未平倉量 (注意: Binance API 僅保留最近 30 天)...")

        # 第一次請求不傳 startTime — 取最新個 limit 筆
        params = {'symbol': symbol, 'period': interval, 'limit': limit}

        try:
            while True:
                resp = requests.get(url, params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()

                if not data:
                    break

                # 避免重複 — 只加入新的
                if all_oi and data[-1]['timestamp'] <= all_oi[-1]['timestamp']:
                    break

                all_oi.extend(data)
                round_count += 1
                if round_count % 10 == 0:
                    print(f"    已爬取 {len(all_oi):,} 筆...")

                if len(data) < limit:
                    break

                # 下一批從最早筆之前開始
                earliest = data[0]['timestamp']
                params['endTime'] = earliest - 1
                params.pop('startTime', None)

                time.sleep(self.rate_limit_delay)

            if not all_oi:
                return pd.DataFrame()

            df = pd.DataFrame(all_oi)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['sumOpenInterest'] = df['sumOpenInterest'].astype(float)
            df['sumOpenInterestValue'] = df['sumOpenInterestValue'].astype(float)
            df = df.sort_values('timestamp').reset_index(drop=True)

            df['oi_change'] = df['sumOpenInterest'].diff()
            df['oi_change_rate'] = df['oi_change'] / df['sumOpenInterest'].shift(1)
            df['oi_ma7'] = df['sumOpenInterest'].rolling(7).mean()
            df['oi_ma30'] = df['sumOpenInterest'].rolling(30).mean()
            df.fillna(0, inplace=True)

            print(f"    ✅ 共 {len(df):,} 筆 (從 {df['timestamp'].min().strftime('%Y-%m-%d')} 至 {df['timestamp'].max().strftime('%Y-%m-%d')})")
            print(f"    ℹ️ Binance API 確認僅保留 30 天, 這是最大可用量")
            return df

        except Exception as e:
            print(f"  ⚠️ 未平倉量無法獲取: {e}")
            return pd.DataFrame()

    # ----------------------------------------------------------
    # 3. 多空比 (僅最近 30 天) — 不傳 startTime
    # ----------------------------------------------------------
    def get_long_short_ratio(self, symbol: str, interval: str = '15m', limit: int = 500) -> pd.DataFrame:
        url = f"{self.futures_base_url}/futures/data/topLongShortAccountRatio"
        all_ratio = []
        round_count = 0

        print(f"  爬取多空比 (注意: Binance API 僅保留最近 30 天)...")

        params = {'symbol': symbol, 'period': interval, 'limit': limit}

        try:
            while True:
                resp = requests.get(url, params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()

                if not data:
                    break

                if all_ratio and data[-1]['timestamp'] <= all_ratio[-1]['timestamp']:
                    break

                all_ratio.extend(data)
                round_count += 1
                if round_count % 10 == 0:
                    print(f"    已爬取 {len(all_ratio):,} 筆...")

                if len(data) < limit:
                    break

                earliest = data[0]['timestamp']
                params['endTime'] = earliest - 1
                params.pop('startTime', None)

                time.sleep(self.rate_limit_delay)

            if not all_ratio:
                return pd.DataFrame()

            df = pd.DataFrame(all_ratio)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['longShortRatio'] = df['longShortRatio'].astype(float)
            df['longAccount'] = df['longAccount'].astype(float)
            df['shortAccount'] = df['shortAccount'].astype(float)
            df = df.sort_values('timestamp').reset_index(drop=True)

            df['ls_ratio_ma7'] = df['longShortRatio'].rolling(7).mean()
            df['ls_ratio_extreme'] = ((df['longShortRatio'] > 2) | (df['longShortRatio'] < 0.5)).astype(int)
            df.fillna(0, inplace=True)

            print(f"    ✅ 共 {len(df):,} 筆 (從 {df['timestamp'].min().strftime('%Y-%m-%d')} 至 {df['timestamp'].max().strftime('%Y-%m-%d')})")
            print(f"    ℹ️ Binance API 確認僅保留 30 天, 這是最大可用量")
            return df

        except Exception as e:
            print(f"  ⚠️ 多空比無法獲取: {e}")
            return pd.DataFrame()

    # ----------------------------------------------------------
    # 4. 主動買賣比 (僅最近 30 天) — 不傳 startTime
    # ----------------------------------------------------------
    def get_taker_buy_sell(self, symbol: str, interval: str = '15m', limit: int = 500) -> pd.DataFrame:
        url = f"{self.futures_base_url}/futures/data/takerlongshortRatio"
        all_taker = []
        round_count = 0

        print(f"  爬取主動買賣比 (注意: Binance API 僅保留最近 30 天)...")

        params = {'symbol': symbol, 'period': interval, 'limit': limit}

        try:
            while True:
                resp = requests.get(url, params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()

                if not data:
                    break

                if all_taker and data[-1]['timestamp'] <= all_taker[-1]['timestamp']:
                    break

                all_taker.extend(data)
                round_count += 1
                if round_count % 10 == 0:
                    print(f"    已爬取 {len(all_taker):,} 筆...")

                if len(data) < limit:
                    break

                earliest = data[0]['timestamp']
                params['endTime'] = earliest - 1
                params.pop('startTime', None)

                time.sleep(self.rate_limit_delay)

            if not all_taker:
                return pd.DataFrame()

            df = pd.DataFrame(all_taker)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['buySellRatio'] = df['buySellRatio'].astype(float)
            df['buyVol'] = df['buyVol'].astype(float)
            df['sellVol'] = df['sellVol'].astype(float)
            df = df.sort_values('timestamp').reset_index(drop=True)

            df['taker_buy_sell_delta'] = df['buyVol'] - df['sellVol']
            df['taker_imbalance'] = df['taker_buy_sell_delta'] / (df['buyVol'] + df['sellVol'])
            df.fillna(0, inplace=True)

            print(f"    ✅ 共 {len(df):,} 筆 (從 {df['timestamp'].min().strftime('%Y-%m-%d')} 至 {df['timestamp'].max().strftime('%Y-%m-%d')})")
            print(f"    ℹ️ Binance API 確認僅保留 30 天, 這是最大可用量")
            return df

        except Exception as e:
            print(f"  ⚠️ 主動買賣比無法獲取: {e}")
            return pd.DataFrame()

    # ----------------------------------------------------------
    # 5. 訂單流 CVD — 從 Taker 計算
    # ----------------------------------------------------------
    def calculate_order_flow_from_taker_buysell(self, taker_df: pd.DataFrame) -> pd.DataFrame:
        if taker_df.empty:
            return pd.DataFrame()

        df = taker_df.copy()
        total_volume = df['buyVol'] + df['sellVol']
        df['delta_volume'] = df['buyVol'] - df['sellVol']
        df['buy_pressure'] = df['buyVol'] / total_volume
        df['sell_pressure'] = df['sellVol'] / total_volume
        df['cvd'] = df['delta_volume'].cumsum()
        df['taker_imbalance'] = df['delta_volume'] / total_volume
        df['cvd_change'] = df['cvd'].diff()
        df['cvd_change_rate'] = df['cvd'].pct_change()
        df['cvd_ma7'] = df['cvd'].rolling(7).mean()
        df['cvd_ma24'] = df['cvd'].rolling(24).mean()
        df['cvd_momentum'] = df['cvd'].diff(7)
        df.fillna(0, inplace=True)
        df.replace([np.inf, -np.inf], 0, inplace=True)
        return df

    # ----------------------------------------------------------
    # 主入口
    # ----------------------------------------------------------
    def collect_all_advanced_features(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: str = '15m'
    ) -> Dict[str, pd.DataFrame]:

        start_time = self.get_earliest_available_time(symbol)
        if start_date:
            start_time = int(pd.to_datetime(start_date).timestamp() * 1000)
        start_label = pd.to_datetime(start_time, unit='ms').strftime('%Y-%m-%d')
        end_label = end_date or datetime.now().strftime('%Y-%m-%d')

        print(f"\n{'='*60}")
        print(f"收集 {symbol} 進階特徵")
        print(f"資金費率時間範圍: {start_label} 至 {end_label} (完整歷史)")
        print(f"OI/多空比/Taker 時間範圍: 最近 30 天 (Binance API 硬性限制)")
        print(f"{'='*60}")

        results = {}

        print("\n[1/5] 資金費率 (Funding Rate) — 完整歷史...")
        results['funding_rate'] = self.get_funding_rate(symbol, start_time)

        print("\n[2/5] 未平倉量 (Open Interest) — 最近 30 天...")
        results['open_interest'] = self.get_open_interest(symbol, timeframe)

        print("\n[3/5] 多空比 (Long/Short Ratio) — 最近 30 天...")
        results['long_short_ratio'] = self.get_long_short_ratio(symbol, timeframe)

        print("\n[4/5] 主動買賣比 (Taker Buy/Sell) — 最近 30 天...")
        taker_df = self.get_taker_buy_sell(symbol, timeframe)
        results['taker_buy_sell'] = taker_df

        print("\n[5/5] 訂單流 CVD (從 Taker 計算)...")
        if not taker_df.empty:
            results['order_flow'] = self.calculate_order_flow_from_taker_buysell(taker_df)
            print(f"  ✅ 生成 {len(results['order_flow']):,} 筆 CVD 特徵")
        else:
            results['order_flow'] = pd.DataFrame()
            print(f"  ⚠️ 無 Taker 數據,無法計算 CVD")

        total = sum(len(v) for v in results.values() if not v.empty)
        print(f"\n{'='*60}")
        print(f"✅ {symbol} 收集完成, 共 {total:,} 筆")
        print(f"{'='*60}\n")

        return results

    def save_advanced_features(
        self,
        symbol: str,
        features_dict: Dict[str, pd.DataFrame],
        output_dir: str = 'v2/advanced_data'
    ):
        os.makedirs(output_dir, exist_ok=True)
        for feature_type, df in features_dict.items():
            if df.empty:
                continue
            filepath = os.path.join(output_dir, f"{symbol}_{feature_type}.parquet")
            df.to_parquet(filepath, index=False)
            print(f"  💾 儲存: {filepath} ({len(df):,} 筆)")


class BatchAdvancedDataCollector:
    def __init__(self):
        self.collector = BinanceAdvancedDataCollector()
        self.hf_symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
            'ADAUSDT', 'AVAXUSDT', 'DOTUSDT', 'MATICUSDT', 'LINKUSDT',
            'UNIUSDT', 'LTCUSDT', 'ETCUSDT', 'XLMUSDT', 'ATOMUSDT',
            'FILUSDT', 'NEARUSDT', 'ALGOUSDT', 'VETUSDT', 'ICPUSDT',
            'APTUSDT', 'ARBUSDT', 'OPUSDT', 'INJUSDT', 'SUIUSDT',
            'PEPEUSDT', 'WIFUSDT', 'SHIBUSDT', 'DOGEUSDT', 'TRXUSDT',
            'TONUSDT', 'HBARUSDT', 'RENDERUSDT', 'FTMUSDT', 'AAVEUSDT',
            'RUNEUSDT', 'IMXUSDT', 'LDOUSDT'
        ]

    def collect_all_symbols(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: str = '15m',
        output_dir: str = 'v2/advanced_data'
    ):
        os.makedirs(output_dir, exist_ok=True)
        summary = []

        for idx, symbol in enumerate(self.hf_symbols, 1):
            print(f"\n[{idx}/{len(self.hf_symbols)}] {symbol}")
            try:
                features_dict = self.collector.collect_all_advanced_features(
                    symbol=symbol, start_date=start_date,
                    end_date=end_date, timeframe=timeframe
                )
                self.collector.save_advanced_features(
                    symbol=symbol, features_dict=features_dict, output_dir=output_dir
                )
                summary.append({
                    'symbol': symbol,
                    'funding_rate': len(features_dict.get('funding_rate', pd.DataFrame())),
                    'open_interest': len(features_dict.get('open_interest', pd.DataFrame())),
                    'long_short_ratio': len(features_dict.get('long_short_ratio', pd.DataFrame())),
                    'taker_buy_sell': len(features_dict.get('taker_buy_sell', pd.DataFrame())),
                    'order_flow_cvd': len(features_dict.get('order_flow', pd.DataFrame())),
                    'status': 'success'
                })
                time.sleep(2)
            except Exception as e:
                print(f"  ❌ 錯誤: {e}")
                summary.append({
                    'symbol': symbol, 'funding_rate': 0, 'open_interest': 0,
                    'long_short_ratio': 0, 'taker_buy_sell': 0, 'order_flow_cvd': 0,
                    'status': f'failed: {str(e)[:40]}'
                })

        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(os.path.join(output_dir, 'collection_summary.csv'), index=False)
        print(summary_df.to_string())
        return summary_df


if __name__ == '__main__':
    BatchAdvancedDataCollector().collect_all_symbols(
        start_date=None, end_date=None,
        timeframe='15m', output_dir='v2/advanced_data'
    )
