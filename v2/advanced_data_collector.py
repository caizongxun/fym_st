import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import os
from tqdm import tqdm


class BinanceAdvancedDataCollector:
    """
    收集 Binance 進階數據:
    1. 訂單流 (Order Flow): CVD, 主動買賣壓力, 大單特徵
    2. 資金費率 (Funding Rate): 散戶多空情緒
    3. 未平倉量 (Open Interest): 流動性獲取證據
    4. 多空比 (Long/Short Ratio): 極端情緒指標
    5. 主動大单比 (Taker Buy/Sell): 爬爆倉代理指標
    
    關鍵優化:
    - 分塊爬取 (1天/塊) 避免 API 超時
    - 自動往前爬取所有可用歷史數據
    - 新增 CVD (累計成交量差) 特徵
    """
    
    def __init__(self):
        self.spot_base_url = 'https://api.binance.com'
        self.futures_base_url = 'https://fapi.binance.com'
        self.rate_limit_delay = 0.3
        
    def get_earliest_available_time(self, symbol: str, data_type: str = 'klines') -> Optional[int]:
        """自動偵測某幣種最早可用數據時間"""
        if data_type == 'klines':
            url = f"{self.spot_base_url}/api/v3/klines"
            params = {'symbol': symbol, 'interval': '1d', 'limit': 1}
        elif data_type == 'futures':
            url = f"{self.futures_base_url}/fapi/v1/klines"
            params = {'symbol': symbol, 'interval': '1d', 'limit': 1}
        else:
            return None
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data and len(data) > 0:
                return data[0][0]
        except:
            pass
        
        return None
    
    def get_aggregate_trades(self, symbol: str, start_time: int, end_time: int) -> pd.DataFrame:
        """
        改進版: 分塊收集聚合成交數據以避免 API 超時
        每次抓取 1 天的數據, 避免一次請求過多導致卡住
        """
        url = f"{self.spot_base_url}/api/v3/aggTrades"
        
        all_trades = []
        chunk_size = 24 * 60 * 60 * 1000
        current_start = start_time
        
        total_chunks = (end_time - start_time) // chunk_size + 1
        print(f"  預計分 {total_chunks} 塊爬取 (每塊 1天)")
        
        chunk_idx = 0
        while current_start < end_time:
            chunk_idx += 1
            chunk_end = min(current_start + chunk_size, end_time)
            
            chunk_start_date = pd.to_datetime(current_start, unit='ms').strftime('%Y-%m-%d')
            print(f"  [{chunk_idx}/{total_chunks}] {chunk_start_date} ...", end=' ')
            
            params = {
                'symbol': symbol,
                'startTime': current_start,
                'endTime': chunk_end,
                'limit': 1000
            }
            
            chunk_trades = []
            last_id = None
            
            try:
                while True:
                    if last_id:
                        params['fromId'] = last_id + 1
                    
                    response = requests.get(url, params=params, timeout=10)
                    response.raise_for_status()
                    trades = response.json()
                    
                    if not trades:
                        break
                    
                    chunk_trades.extend(trades)
                    
                    if len(trades) < 1000:
                        break
                    
                    last_id = trades[-1]['a']
                    
                    if trades[-1]['T'] >= chunk_end:
                        break
                    
                    time.sleep(self.rate_limit_delay)
                
                all_trades.extend(chunk_trades)
                print(f"✓ {len(chunk_trades):,} 筆")
                
            except Exception as e:
                print(f"⚠️ {str(e)[:30]}")
            
            current_start = chunk_end
            time.sleep(self.rate_limit_delay)
        
        if not all_trades:
            print(f"  ❌ 無聚合成交數據")
            return pd.DataFrame()
        
        print(f"  ✅ 總計 {len(all_trades):,} 筆成交")
        
        df = pd.DataFrame(all_trades)
        df['T'] = pd.to_datetime(df['T'], unit='ms')
        df['timestamp'] = df['T']
        df['price'] = df['p'].astype(float)
        df['qty'] = df['q'].astype(float)
        df['is_buyer_maker'] = df['m']
        
        df = df[['timestamp', 'price', 'qty', 'is_buyer_maker']]
        
        return df
    
    def calculate_order_flow_features(self, trades_df: pd.DataFrame, timeframe: str = '15T') -> pd.DataFrame:
        """訂單流特徵: CVD, 主動買賣壓, 大單比例"""
        if trades_df.empty:
            return pd.DataFrame()
        
        trades_df = trades_df.copy()
        
        trades_df['buy_volume'] = trades_df.apply(
            lambda x: 0 if x['is_buyer_maker'] else x['qty'], axis=1
        )
        trades_df['sell_volume'] = trades_df.apply(
            lambda x: x['qty'] if x['is_buyer_maker'] else 0, axis=1
        )
        
        trades_df.set_index('timestamp', inplace=True)
        
        agg_dict = {
            'price': ['first', 'last', 'mean'],
            'qty': ['sum', 'count', 'mean', 'std'],
            'buy_volume': 'sum',
            'sell_volume': 'sum'
        }
        
        df_agg = trades_df.resample(timeframe).agg(agg_dict)
        df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]
        
        df_agg['delta_volume'] = df_agg['buy_volume_sum'] - df_agg['sell_volume_sum']
        
        total_volume = df_agg['buy_volume_sum'] + df_agg['sell_volume_sum']
        df_agg['buy_pressure'] = df_agg['buy_volume_sum'] / total_volume
        df_agg['sell_pressure'] = df_agg['sell_volume_sum'] / total_volume
        
        df_agg['cvd'] = df_agg['delta_volume'].cumsum()
        
        df_agg['trade_intensity'] = df_agg['qty_count'] / 15
        df_agg['avg_trade_size'] = df_agg['qty_sum'] / df_agg['qty_count']
        df_agg['trade_size_volatility'] = df_agg['qty_std'] / df_agg['qty_mean']
        
        large_threshold = df_agg['qty_mean'] * 2
        trades_df['is_large_trade'] = trades_df['qty'] > large_threshold.reindex(trades_df.index, method='ffill')
        large_trades = trades_df[trades_df['is_large_trade']].resample(timeframe).size()
        df_agg['large_trade_count'] = large_trades
        df_agg['large_trade_ratio'] = df_agg['large_trade_count'] / df_agg['qty_count']
        
        df_agg.fillna(0, inplace=True)
        df_agg.replace([np.inf, -np.inf], 0, inplace=True)
        
        df_agg.reset_index(inplace=True)
        
        return df_agg
    
    def get_funding_rate(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        """資金費率: 往前爬取所有歷史數據"""
        url = f"{self.futures_base_url}/fapi/v1/fundingRate"
        
        all_funding = []
        params = {'symbol': symbol, 'limit': limit}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            funding_data = response.json()
            
            if not funding_data:
                return pd.DataFrame()
            
            all_funding.extend(funding_data)
            
            while len(funding_data) == limit:
                earliest_time = funding_data[0]['fundingTime']
                params['endTime'] = earliest_time - 1
                
                time.sleep(self.rate_limit_delay)
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                funding_data = response.json()
                
                if not funding_data:
                    break
                
                all_funding = funding_data + all_funding
            
            df = pd.DataFrame(all_funding)
            df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
            df['fundingRate'] = df['fundingRate'].astype(float)
            df['timestamp'] = df['fundingTime']
            
            df = df[['timestamp', 'fundingRate']]
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            df['funding_rate_ma8'] = df['fundingRate'].rolling(8).mean()
            df['funding_rate_ma24'] = df['fundingRate'].rolling(24).mean()
            df['funding_rate_std'] = df['fundingRate'].rolling(24).std()
            df['funding_rate_extreme'] = (abs(df['fundingRate']) > df['funding_rate_std'] * 2).astype(int)
            
            return df
            
        except Exception as e:
            print(f"  ⚠️ 資金費率無法獲取: {e}")
            return pd.DataFrame()
    
    def get_open_interest(self, symbol: str, interval: str = '15m', limit: int = 500) -> pd.DataFrame:
        """未平倉量: 爬取所有歷史數據"""
        url = f"{self.futures_base_url}/futures/data/openInterestHist"
        
        all_oi = []
        params = {'symbol': symbol, 'period': interval, 'limit': limit}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            oi_data = response.json()
            
            if not oi_data:
                return pd.DataFrame()
            
            all_oi.extend(oi_data)
            
            while len(oi_data) == limit:
                earliest_time = oi_data[0]['timestamp']
                params['endTime'] = earliest_time - 1
                
                time.sleep(self.rate_limit_delay)
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                oi_data = response.json()
                
                if not oi_data:
                    break
                
                all_oi = oi_data + all_oi
            
            df = pd.DataFrame(all_oi)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['sumOpenInterest'] = df['sumOpenInterest'].astype(float)
            df['sumOpenInterestValue'] = df['sumOpenInterestValue'].astype(float)
            
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            df['oi_change'] = df['sumOpenInterest'].diff()
            df['oi_change_rate'] = df['oi_change'] / df['sumOpenInterest'].shift(1)
            df['oi_ma7'] = df['sumOpenInterest'].rolling(7).mean()
            df['oi_ma30'] = df['sumOpenInterest'].rolling(30).mean()
            
            return df
            
        except Exception as e:
            print(f"  ⚠️ 未平倉量無法獲取: {e}")
            return pd.DataFrame()
    
    def get_long_short_ratio(self, symbol: str, interval: str = '15m', limit: int = 500) -> pd.DataFrame:
        """多空比: 爬取所有歷史數據"""
        url = f"{self.futures_base_url}/futures/data/topLongShortAccountRatio"
        
        all_ratio = []
        params = {'symbol': symbol, 'period': interval, 'limit': limit}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            ratio_data = response.json()
            
            if not ratio_data:
                return pd.DataFrame()
            
            all_ratio.extend(ratio_data)
            
            while len(ratio_data) == limit:
                earliest_time = ratio_data[0]['timestamp']
                params['endTime'] = earliest_time - 1
                
                time.sleep(self.rate_limit_delay)
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                ratio_data = response.json()
                
                if not ratio_data:
                    break
                
                all_ratio = ratio_data + all_ratio
            
            df = pd.DataFrame(all_ratio)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['longShortRatio'] = df['longShortRatio'].astype(float)
            df['longAccount'] = df['longAccount'].astype(float)
            df['shortAccount'] = df['shortAccount'].astype(float)
            
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            df['ls_ratio_ma7'] = df['longShortRatio'].rolling(7).mean()
            df['ls_ratio_extreme'] = ((df['longShortRatio'] > 2) | (df['longShortRatio'] < 0.5)).astype(int)
            
            return df
            
        except Exception as e:
            print(f"  ⚠️ 多空比無法獲取: {e}")
            return pd.DataFrame()
    
    def get_taker_buy_sell(self, symbol: str, interval: str = '15m', limit: int = 500) -> pd.DataFrame:
        """主動買賣比: 爬爆倉代理指標"""
        url = f"{self.futures_base_url}/futures/data/takerlongshortRatio"
        
        all_taker = []
        params = {'symbol': symbol, 'period': interval, 'limit': limit}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            taker_data = response.json()
            
            if not taker_data:
                return pd.DataFrame()
            
            all_taker.extend(taker_data)
            
            while len(taker_data) == limit:
                earliest_time = taker_data[0]['timestamp']
                params['endTime'] = earliest_time - 1
                
                time.sleep(self.rate_limit_delay)
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                taker_data = response.json()
                
                if not taker_data:
                    break
                
                all_taker = taker_data + all_taker
            
            df = pd.DataFrame(all_taker)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['buySellRatio'] = df['buySellRatio'].astype(float)
            df['buyVol'] = df['buyVol'].astype(float)
            df['sellVol'] = df['sellVol'].astype(float)
            
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            df['taker_buy_sell_delta'] = df['buyVol'] - df['sellVol']
            df['taker_imbalance'] = df['taker_buy_sell_delta'] / (df['buyVol'] + df['sellVol'])
            
            return df
            
        except Exception as e:
            print(f"  ⚠️ 主動買賣比無法獲取: {e}")
            return pd.DataFrame()
    
    def collect_all_advanced_features(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: str = '15m'
    ) -> Dict[str, pd.DataFrame]:
        """收集所有進階特徵, 如未指定日期則自動偵測並爬取所有可用數據"""
        
        if start_date is None or end_date is None:
            earliest_time = self.get_earliest_available_time(symbol, 'futures')
            if earliest_time:
                start_date = pd.to_datetime(earliest_time, unit='ms').strftime('%Y-%m-%d')
            else:
                start_date = '2019-01-01'
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        start_time = int(start_dt.timestamp() * 1000)
        end_time = int(end_dt.timestamp() * 1000)
        
        print(f"\n{'='*60}")
        print(f"收集 {symbol} 進階特徵")
        print(f"時間範圍: {start_date} 至 {end_date}")
        print(f"{'='*60}")
        
        results = {}
        
        print("\n[1/5] 訂單流特徵 (Order Flow - CVD)...")
        trades_df = self.get_aggregate_trades(symbol, start_time, end_time)
        if not trades_df.empty:
            order_flow_df = self.calculate_order_flow_features(trades_df, timeframe)
            results['order_flow'] = order_flow_df
            print(f"  ✅ 生成 {len(order_flow_df)} 筆訂單流特徵")
        else:
            results['order_flow'] = pd.DataFrame()
        
        print("\n[2/5] 資金費率 (Funding Rate)...")
        funding_df = self.get_funding_rate(symbol)
        if not funding_df.empty:
            results['funding_rate'] = funding_df
            print(f"  ✅ 獲取 {len(funding_df)} 筆資金費率")
        else:
            results['funding_rate'] = pd.DataFrame()
        
        print("\n[3/5] 未平倉量 (Open Interest)...")
        oi_df = self.get_open_interest(symbol, timeframe)
        if not oi_df.empty:
            results['open_interest'] = oi_df
            print(f"  ✅ 獲取 {len(oi_df)} 筆未平倉量")
        else:
            results['open_interest'] = pd.DataFrame()
        
        print("\n[4/5] 多空比 (Long/Short Ratio)...")
        ls_ratio_df = self.get_long_short_ratio(symbol, timeframe)
        if not ls_ratio_df.empty:
            results['long_short_ratio'] = ls_ratio_df
            print(f"  ✅ 獲取 {len(ls_ratio_df)} 筆多空比")
        else:
            results['long_short_ratio'] = pd.DataFrame()
        
        print("\n[5/5] 主動買賣比 (Taker Buy/Sell)...")
        taker_df = self.get_taker_buy_sell(symbol, timeframe)
        if not taker_df.empty:
            results['taker_buy_sell'] = taker_df
            print(f"  ✅ 獲取 {len(taker_df)} 筆主動買賣比")
        else:
            results['taker_buy_sell'] = pd.DataFrame()
        
        print(f"\n{'='*60}")
        total_records = sum([len(df) for df in results.values() if not df.empty])
        print(f"✅ {symbol} 收集完成, 共 {total_records:,} 筆數據")
        print(f"{'='*60}\n")
        
        return results
    
    def save_advanced_features(
        self,
        symbol: str,
        features_dict: Dict[str, pd.DataFrame],
        output_dir: str = 'v2/advanced_data'
    ):
        """Save all advanced features to parquet files"""
        os.makedirs(output_dir, exist_ok=True)
        
        for feature_type, df in features_dict.items():
            if df.empty:
                continue
            
            filename = f"{symbol}_{feature_type}.parquet"
            filepath = os.path.join(output_dir, filename)
            
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
        """批量收集所有幣種的進階數據"""
        print(f"\n{'='*80}")
        print(f"批量進階數據收集")
        print(f"{'='*80}")
        print(f"幣種數量: {len(self.hf_symbols)}")
        print(f"時間範圍: {start_date or '自動偵測'} 至 {end_date or '現在'}")
        print(f"時間框架: {timeframe}")
        print(f"輸出目錄: {output_dir}")
        print(f"{'='*80}\n")
        
        os.makedirs(output_dir, exist_ok=True)
        summary = []
        
        for idx, symbol in enumerate(self.hf_symbols, 1):
            print(f"\n[{idx}/{len(self.hf_symbols)}] {symbol}")
            
            try:
                features_dict = self.collector.collect_all_advanced_features(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe=timeframe
                )
                
                self.collector.save_advanced_features(
                    symbol=symbol,
                    features_dict=features_dict,
                    output_dir=output_dir
                )
                
                summary.append({
                    'symbol': symbol,
                    'order_flow': len(features_dict.get('order_flow', pd.DataFrame())),
                    'funding_rate': len(features_dict.get('funding_rate', pd.DataFrame())),
                    'open_interest': len(features_dict.get('open_interest', pd.DataFrame())),
                    'long_short_ratio': len(features_dict.get('long_short_ratio', pd.DataFrame())),
                    'taker_buy_sell': len(features_dict.get('taker_buy_sell', pd.DataFrame())),
                    'status': 'success'
                })
                
                time.sleep(2)
                
            except Exception as e:
                print(f"  ❌ 錯誤: {e}")
                summary.append({
                    'symbol': symbol,
                    'order_flow': 0,
                    'funding_rate': 0,
                    'open_interest': 0,
                    'long_short_ratio': 0,
                    'taker_buy_sell': 0,
                    'status': f'failed: {str(e)[:30]}'
                })
        
        summary_df = pd.DataFrame(summary)
        summary_path = os.path.join(output_dir, 'collection_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\n{'='*80}")
        print(f"批量收集完成")
        print(f"{'='*80}")
        print(f"\n摘要:")
        print(summary_df)
        print(f"\n摘要儲存至: {summary_path}")
        
        return summary_df


if __name__ == '__main__':
    collector = BatchAdvancedDataCollector()
    
    summary = collector.collect_all_symbols(
        start_date=None,
        end_date=None,
        timeframe='15m',
        output_dir='v2/advanced_data'
    )
