import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import os
from tqdm import tqdm


class BinanceAdvancedDataCollector:
    def __init__(self):
        self.spot_base_url = 'https://api.binance.com'
        self.futures_base_url = 'https://fapi.binance.com'
        self.rate_limit_delay = 0.2
        
    def get_aggregate_trades(self, symbol: str, start_time: int, end_time: int) -> pd.DataFrame:
        url = f"{self.spot_base_url}/api/v3/aggTrades"
        
        all_trades = []
        current_start = start_time
        
        while current_start < end_time:
            params = {
                'symbol': symbol,
                'startTime': current_start,
                'endTime': end_time,
                'limit': 1000
            }
            
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                trades = response.json()
                
                if not trades:
                    break
                
                all_trades.extend(trades)
                current_start = trades[-1]['T'] + 1
                
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                print(f"Error fetching trades for {symbol}: {e}")
                break
        
        if not all_trades:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_trades)
        df['T'] = pd.to_datetime(df['T'], unit='ms')
        df['timestamp'] = df['T']
        df['price'] = df['p'].astype(float)
        df['qty'] = df['q'].astype(float)
        df['is_buyer_maker'] = df['m']
        
        df = df[['timestamp', 'price', 'qty', 'is_buyer_maker']]
        
        return df
    
    def calculate_order_flow_features(self, trades_df: pd.DataFrame, timeframe: str = '15T') -> pd.DataFrame:
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
        df_agg['buy_pressure'] = df_agg['buy_volume_sum'] / (df_agg['buy_volume_sum'] + df_agg['sell_volume_sum'])
        df_agg['sell_pressure'] = df_agg['sell_volume_sum'] / (df_agg['buy_volume_sum'] + df_agg['sell_volume_sum'])
        
        df_agg['trade_intensity'] = df_agg['qty_count'] / 15
        
        df_agg['avg_trade_size'] = df_agg['qty_sum'] / df_agg['qty_count']
        df_agg['trade_size_volatility'] = df_agg['qty_std'] / df_agg['qty_mean']
        
        large_threshold = df_agg['qty_mean'] * 2
        trades_df['is_large_trade'] = trades_df['qty'] > large_threshold.reindex(trades_df.index, method='ffill')
        large_trades = trades_df[trades_df['is_large_trade']].resample(timeframe).size()
        df_agg['large_trade_count'] = large_trades
        df_agg['large_trade_ratio'] = df_agg['large_trade_count'] / df_agg['qty_count']
        
        df_agg.fillna(0, inplace=True)
        
        df_agg.reset_index(inplace=True)
        
        return df_agg
    
    def get_funding_rate(self, symbol: str, start_time: int, end_time: int) -> pd.DataFrame:
        url = f"{self.futures_base_url}/fapi/v1/fundingRate"
        
        params = {
            'symbol': symbol,
            'startTime': start_time,
            'endTime': end_time,
            'limit': 1000
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            funding_data = response.json()
            
            if not funding_data:
                return pd.DataFrame()
            
            df = pd.DataFrame(funding_data)
            df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
            df['fundingRate'] = df['fundingRate'].astype(float)
            df['timestamp'] = df['fundingTime']
            
            df = df[['timestamp', 'fundingRate']]
            
            df['funding_rate_ma8'] = df['fundingRate'].rolling(8).mean()
            df['funding_rate_ma24'] = df['fundingRate'].rolling(24).mean()
            df['funding_rate_std'] = df['fundingRate'].rolling(24).std()
            df['funding_rate_extreme'] = (abs(df['fundingRate']) > df['funding_rate_std'] * 2).astype(int)
            
            return df
            
        except Exception as e:
            print(f"Error fetching funding rate for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_open_interest(self, symbol: str, interval: str = '15m', start_time: int = None, end_time: int = None) -> pd.DataFrame:
        url = f"{self.futures_base_url}/futures/data/openInterestHist"
        
        params = {
            'symbol': symbol,
            'period': interval,
            'limit': 500
        }
        
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            oi_data = response.json()
            
            if not oi_data:
                return pd.DataFrame()
            
            df = pd.DataFrame(oi_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['sumOpenInterest'] = df['sumOpenInterest'].astype(float)
            df['sumOpenInterestValue'] = df['sumOpenInterestValue'].astype(float)
            
            df['oi_change'] = df['sumOpenInterest'].diff()
            df['oi_change_rate'] = df['oi_change'] / df['sumOpenInterest'].shift(1)
            
            df['oi_ma7'] = df['sumOpenInterest'].rolling(7).mean()
            df['oi_ma30'] = df['sumOpenInterest'].rolling(30).mean()
            
            return df
            
        except Exception as e:
            print(f"Error fetching open interest for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_long_short_ratio(self, symbol: str, interval: str = '15m', start_time: int = None, end_time: int = None) -> pd.DataFrame:
        url = f"{self.futures_base_url}/futures/data/topLongShortAccountRatio"
        
        params = {
            'symbol': symbol,
            'period': interval,
            'limit': 500
        }
        
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            ratio_data = response.json()
            
            if not ratio_data:
                return pd.DataFrame()
            
            df = pd.DataFrame(ratio_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['longShortRatio'] = df['longShortRatio'].astype(float)
            df['longAccount'] = df['longAccount'].astype(float)
            df['shortAccount'] = df['shortAccount'].astype(float)
            
            df['ls_ratio_ma7'] = df['longShortRatio'].rolling(7).mean()
            df['ls_ratio_extreme'] = ((df['longShortRatio'] > 2) | (df['longShortRatio'] < 0.5)).astype(int)
            
            return df
            
        except Exception as e:
            print(f"Error fetching long/short ratio for {symbol}: {e}")
            return pd.DataFrame()
    
    def collect_all_advanced_features(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str = '15m'
    ) -> Dict[str, pd.DataFrame]:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        start_time = int(start_dt.timestamp() * 1000)
        end_time = int(end_dt.timestamp() * 1000)
        
        print(f"\nCollecting advanced features for {symbol}...")
        print(f"Period: {start_date} to {end_date}")
        
        results = {}
        
        print("\n[1/5] Fetching aggregate trades...")
        trades_df = self.get_aggregate_trades(symbol, start_time, end_time)
        if not trades_df.empty:
            print(f"  Retrieved {len(trades_df)} trades")
            
            print("\n[2/5] Calculating order flow features...")
            order_flow_df = self.calculate_order_flow_features(trades_df, timeframe)
            print(f"  Generated {len(order_flow_df)} timeframe records")
            results['order_flow'] = order_flow_df
        else:
            print("  No trade data available")
            results['order_flow'] = pd.DataFrame()
        
        print("\n[3/5] Fetching funding rate...")
        funding_df = self.get_funding_rate(symbol, start_time, end_time)
        if not funding_df.empty:
            print(f"  Retrieved {len(funding_df)} funding rate records")
            results['funding_rate'] = funding_df
        else:
            print("  No funding rate data available")
            results['funding_rate'] = pd.DataFrame()
        
        print("\n[4/5] Fetching open interest...")
        oi_df = self.get_open_interest(symbol, timeframe, start_time, end_time)
        if not oi_df.empty:
            print(f"  Retrieved {len(oi_df)} open interest records")
            results['open_interest'] = oi_df
        else:
            print("  No open interest data available")
            results['open_interest'] = pd.DataFrame()
        
        print("\n[5/5] Fetching long/short ratio...")
        ls_ratio_df = self.get_long_short_ratio(symbol, timeframe, start_time, end_time)
        if not ls_ratio_df.empty:
            print(f"  Retrieved {len(ls_ratio_df)} long/short ratio records")
            results['long_short_ratio'] = ls_ratio_df
        else:
            print("  No long/short ratio data available")
            results['long_short_ratio'] = pd.DataFrame()
        
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
            
            filename = f"{symbol}_{feature_type}.parquet"
            filepath = os.path.join(output_dir, filename)
            
            df.to_parquet(filepath, index=False)
            print(f"Saved {feature_type}: {filepath} ({len(df)} records)")


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
        start_date: str = '2024-01-01',
        end_date: str = '2024-12-31',
        timeframe: str = '15m',
        output_dir: str = 'v2/advanced_data'
    ):
        print(f"\n{'='*80}")
        print(f"Batch Advanced Data Collection")
        print(f"{'='*80}")
        print(f"Symbols: {len(self.hf_symbols)}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Timeframe: {timeframe}")
        print(f"Output: {output_dir}")
        print(f"{'='*80}\n")
        
        os.makedirs(output_dir, exist_ok=True)
        
        summary = []
        
        for idx, symbol in enumerate(self.hf_symbols, 1):
            print(f"\n[{idx}/{len(self.hf_symbols)}] Processing {symbol}...")
            
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
                    'order_flow_records': len(features_dict.get('order_flow', pd.DataFrame())),
                    'funding_rate_records': len(features_dict.get('funding_rate', pd.DataFrame())),
                    'open_interest_records': len(features_dict.get('open_interest', pd.DataFrame())),
                    'long_short_ratio_records': len(features_dict.get('long_short_ratio', pd.DataFrame())),
                    'status': 'success'
                })
                
                time.sleep(1)
                
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                summary.append({
                    'symbol': symbol,
                    'order_flow_records': 0,
                    'funding_rate_records': 0,
                    'open_interest_records': 0,
                    'long_short_ratio_records': 0,
                    'status': f'failed: {str(e)}'
                })
        
        summary_df = pd.DataFrame(summary)
        summary_path = os.path.join(output_dir, 'collection_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\n{'='*80}")
        print(f"Batch Collection Complete")
        print(f"{'='*80}")
        print(f"\nSummary:")
        print(summary_df)
        print(f"\nSummary saved to: {summary_path}")
        
        return summary_df


if __name__ == '__main__':
    collector = BatchAdvancedDataCollector()
    
    summary = collector.collect_all_symbols(
        start_date='2024-01-01',
        end_date='2024-12-31',
        timeframe='15m',
        output_dir='v2/advanced_data'
    )
