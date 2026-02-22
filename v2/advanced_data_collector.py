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
# Binance API 歷史數據保留期限制 (無法繞過)
# ------------------------------------------------------------
# funding_rate (fapi/v1/fundingRate):   完整歷史 (2019 至今)
# open_interest (openInterestHist):     僅最近 30 天
# long_short_ratio (topLongShort...):   僅最近 30 天
# taker_buy_sell (takerlongshortRatio): 僅最近 30 天
#
# 完整歷史 OI 下載來源:
# https://data.binance.vision/data/futures/um/monthly/openInterestFapi/
# ============================================================


class BinanceAdvancedDataCollector:

    def __init__(self):
        self.spot_base_url = 'https://api.binance.com'
        self.futures_base_url = 'https://fapi.binance.com'
        self.public_data_url = 'https://data.binance.vision'
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
        print(f"  爬取未平倉量 (注意: Binance API 僅保留最近 30 天)...")
        params = {'symbol': symbol, 'period': interval, 'limit': limit}
        try:
            while True:
                resp = requests.get(url, params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                if not data:
                    break
                if all_oi and data[-1]['timestamp'] <= all_oi[-1]['timestamp']:
                    break
                all_oi.extend(data)
                if len(data) < limit:
                    break
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
            print(f"    ℹ️ Binance API 硬性 30 天限制，這是最大可用量")
            return df
        except Exception as e:
            print(f"  ⚠️ 未平倉量無法獲取: {e}")
            return pd.DataFrame()

    # ----------------------------------------------------------
    # 3. 多空比 (僅最近 30 天)
    # ----------------------------------------------------------
    def get_long_short_ratio(self, symbol: str, interval: str = '15m', limit: int = 500) -> pd.DataFrame:
        url = f"{self.futures_base_url}/futures/data/topLongShortAccountRatio"
        all_ratio = []
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
            print(f"    ℹ️ Binance API 硬性 30 天限制，這是最大可用量")
            return df
        except Exception as e:
            print(f"  ⚠️ 多空比無法獲取: {e}")
            return pd.DataFrame()

    # ----------------------------------------------------------
    # 4. 主動買賣比 (僅最近 30 天)
    # ----------------------------------------------------------
    def get_taker_buy_sell(self, symbol: str, interval: str = '15m', limit: int = 500) -> pd.DataFrame:
        url = f"{self.futures_base_url}/futures/data/takerlongshortRatio"
        all_taker = []
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
            print(f"    ℹ️ Binance API 硬性 30 天限制，這是最大可用量")
            return df
        except Exception as e:
            print(f"  ⚠️ 主動買賣比無法獲取: {e}")
            return pd.DataFrame()

    # ----------------------------------------------------------
    # 5. Binance Public Data — 完整歷史 OI 下載
    # ----------------------------------------------------------
    def download_oi_from_public_data(
        self,
        symbol: str,
        output_dir: str = 'v2/advanced_data'
    ) -> pd.DataFrame:
        """
        從 data.binance.vision 下載完整歷史 OI (CSV zip 按月)
        URL: {base}/{symbol}/{symbol}-openInterest-{YYYY}-{MM}.zip

        資料格式: create_time, symbol, sum_open_interest, sum_open_interest_value
        """
        base_url = f"{self.public_data_url}/data/futures/um/monthly/openInterestFapi"
        start_time = self.get_earliest_available_time(symbol)
        start_date = pd.to_datetime(start_time, unit='ms').replace(day=1)
        end_date = datetime.now()

        print(f"\n  從 Binance Public Data 下載完整歷史 OI ({symbol})...")
        print(f"  時間範圍: {start_date.strftime('%Y-%m')} 至 {end_date.strftime('%Y-%m')}")

        all_dfs = []
        current = start_date

        while current <= end_date:
            year = current.year
            month = current.month
            url = f"{base_url}/{symbol}/{symbol}-openInterest-{year}-{month:02d}.zip"

            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 404:
                    print(f"    {year}-{month:02d}: 無數據 (404)")
                    current = current + pd.DateOffset(months=1)
                    continue
                resp.raise_for_status()

                with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                    csv_name = z.namelist()[0]
                    with z.open(csv_name) as f:
                        try:
                            df_month = pd.read_csv(f)
                            # 嘗試標準欄位名
                            if 'create_time' not in df_month.columns:
                                f.seek(0)
                                df_month = pd.read_csv(
                                    f, header=None,
                                    names=['create_time', 'symbol', 'sum_open_interest', 'sum_open_interest_value']
                                )
                        except Exception:
                            df_month = pd.read_csv(
                                io.BytesIO(z.read(csv_name)), header=None,
                                names=['create_time', 'symbol', 'sum_open_interest', 'sum_open_interest_value']
                            )

                all_dfs.append(df_month)
                print(f"    ✅ {year}-{month:02d}: {len(df_month):,} 筆")

            except Exception as e:
                print(f"    ⚠️ {year}-{month:02d}: {str(e)[:50]}")

            current = current + pd.DateOffset(months=1)
            time.sleep(0.2)

        if not all_dfs:
            print(f"  ⚠️ 無法下載任何 OI 數據，請檢查網絡或 URL 形式")
            return pd.DataFrame()

        df = pd.concat(all_dfs, ignore_index=True)

        # 標準化欄位名
        col_map = {
            'create_time': 'raw_time',
            'sum_open_interest': 'sumOpenInterest',
            'sum_open_interest_value': 'sumOpenInterestValue'
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        # 時間解析 (ms 或 datetime 字串)
        if 'raw_time' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['raw_time'], unit='ms')
            except Exception:
                df['timestamp'] = pd.to_datetime(df['raw_time'])

        if 'timestamp' not in df.columns:
            print("  ⚠️ 無法解析時間欄位")
            return pd.DataFrame()

        df['sumOpenInterest'] = pd.to_numeric(df['sumOpenInterest'], errors='coerce')
        df['sumOpenInterestValue'] = pd.to_numeric(df.get('sumOpenInterestValue', 0), errors='coerce')
        df = df[['timestamp', 'sumOpenInterest', 'sumOpenInterestValue']]
        df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)

        # 衍生特徵
        df['oi_change'] = df['sumOpenInterest'].diff()
        df['oi_change_rate'] = df['oi_change'] / df['sumOpenInterest'].shift(1)
        df['oi_ma7'] = df['sumOpenInterest'].rolling(7).mean()
        df['oi_ma30'] = df['sumOpenInterest'].rolling(30).mean()
        df.fillna(0, inplace=True)

        # 儲存
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{symbol}_open_interest_full.parquet")
        df.to_parquet(filepath, index=False)
        print(f"\n  ✅ 完整歷史 OI 完成")
        print(f"  共 {len(df):,} 筆 (從 {df['timestamp'].min().strftime('%Y-%m-%d')} 至 {df['timestamp'].max().strftime('%Y-%m-%d')})")
        print(f"  💾 儲存: {filepath}")

        return df

    # ----------------------------------------------------------
    # 收集所有特徵
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
        print(f"資金費率: {start_label} 至 {end_label} (完整歷史)")
        print(f"OI/多空比/Taker: 最近 30 天 (Binance API 硬性限制)")
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

    def download_all_oi_history(self, output_dir: str = 'v2/advanced_data'):
        """從 Binance Public Data 下載所有幣種完整歷史 OI"""
        print(f"\n{'='*80}")
        print(f"下載完整歷史 OI (data.binance.vision)")
        print(f"幣種數量: {len(self.hf_symbols)}")
        print(f"{'='*80}")

        results = []
        for idx, symbol in enumerate(self.hf_symbols, 1):
            print(f"\n[{idx}/{len(self.hf_symbols)}] {symbol}")
            df = self.collector.download_oi_from_public_data(symbol, output_dir)
            results.append({'symbol': symbol, 'records': len(df), 'status': 'ok' if not df.empty else 'empty'})
            time.sleep(1)

        summary = pd.DataFrame(results)
        print("\n摘要:")
        print(summary.to_string())
        return summary

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
