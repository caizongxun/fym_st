from huggingface_hub import hf_hub_download
import pandas as pd
from typing import List, Optional
import logging
from binance.client import Client
from datetime import datetime, timedelta
import os
import requests

logger = logging.getLogger(__name__)

class CryptoDataLoader:
    def __init__(self):
        self.repo_id = "zongowo111/v2-crypto-ohlcv-data"
        self.symbols = [
            "AAVEUSDT", "ADAUSDT", "ALGOUSDT", "ARBUSDT", "ATOMUSDT",
            "AVAXUSDT", "BALUSDT", "BATUSDT", "BCHUSDT", "BNBUSDT",
            "BTCUSDT", "COMPUSDT", "CRVUSDT", "DOGEUSDT", "DOTUSDT",
            "ENJUSDT", "ENSUSDT", "ETCUSDT", "ETHUSDT", "FILUSDT",
            "GALAUSDT", "GRTUSDT", "IMXUSDT", "KAVAUSDT", "LINKUSDT",
            "LTCUSDT", "MANAUSDT", "MATICUSDT", "MKRUSDT", "NEARUSDT",
            "OPUSDT", "SANDUSDT", "SNXUSDT", "SOLUSDT", "SPELLUSDT",
            "UNIUSDT", "XRPUSDT", "ZRXUSDT"
        ]
        self.timeframes = ["15m", "1h", "1d"]
        self.binance_client = None
        self.binance_futures_base_url = "https://fapi.binance.com"
    
    def _get_binance_client(self):
        """Lazy initialization of Binance client"""
        if self.binance_client is None:
            api_key = os.getenv('BINANCE_API_KEY', '')
            api_secret = os.getenv('BINANCE_API_SECRET', '')
            self.binance_client = Client(api_key, api_secret)
        return self.binance_client
    
    def fetch_open_interest(self, symbol: str, timeframe: str, days: int = 90) -> pd.DataFrame:
        """
        從 Binance Futures API 獲取歷史 Open Interest 數據
        
        Args:
            symbol: 交易對 (e.g., BTCUSDT)
            timeframe: 15m, 1h, 1d
            days: 往前抽取多少天
        
        Returns:
            DataFrame with columns: ['timestamp', 'open_interest']
        """
        interval_map = {
            '15m': '15m',
            '1h': '1h',
            '1d': '1d'
        }
        
        period = interval_map.get(timeframe, '1h')
        
        # 計算開始時間戳
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        
        url = f"{self.binance_futures_base_url}/futures/data/openInterestHist"
        
        all_data = []
        limit = 500  # Binance API 每次最多返回 500 筆
        
        try:
            while start_time < end_time:
                params = {
                    'symbol': symbol,
                    'period': period,
                    'limit': limit,
                    'startTime': start_time,
                    'endTime': end_time
                }
                
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    break
                
                all_data.extend(data)
                
                # 更新開始時間為最後一筆數據的時間
                start_time = data[-1]['timestamp'] + 1
                
                if len(data) < limit:
                    break
            
            df = pd.DataFrame(all_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['open_interest'] = pd.to_numeric(df['sumOpenInterest'])
            df['open_interest_value'] = pd.to_numeric(df['sumOpenInterestValue'])
            
            df = df[['timestamp', 'open_interest', 'open_interest_value']]
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Fetched {len(df)} OI records for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch OI data: {str(e)}")
            # 返回空 DataFrame
            return pd.DataFrame(columns=['timestamp', 'open_interest', 'open_interest_value'])
    
    def fetch_funding_rate(self, symbol: str, days: int = 90) -> pd.DataFrame:
        """
        獲取資金費率歷史數據
        
        Args:
            symbol: 交易對
            days: 往前抽取多少天
        
        Returns:
            DataFrame with columns: ['timestamp', 'funding_rate']
        """
        url = f"{self.binance_futures_base_url}/fapi/v1/fundingRate"
        
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        end_time = int(datetime.now().timestamp() * 1000)
        
        all_data = []
        limit = 1000
        
        try:
            while start_time < end_time:
                params = {
                    'symbol': symbol,
                    'startTime': start_time,
                    'endTime': end_time,
                    'limit': limit
                }
                
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    break
                
                all_data.extend(data)
                start_time = data[-1]['fundingTime'] + 1
                
                if len(data) < limit:
                    break
            
            df = pd.DataFrame(all_data)
            df['timestamp'] = pd.to_datetime(df['fundingTime'], unit='ms')
            df['funding_rate'] = pd.to_numeric(df['fundingRate'])
            
            df = df[['timestamp', 'funding_rate']]
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Fetched {len(df)} funding rate records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch funding rate: {str(e)}")
            return pd.DataFrame(columns=['timestamp', 'funding_rate'])
    
    def fetch_latest_klines(self, symbol: str, timeframe: str, days: int = 90, 
                           include_oi: bool = False, include_funding: bool = False) -> pd.DataFrame:
        """
        從 Binance API 獲取最新 K 線數據,可選加入 OI 和資金費率
        
        Args:
            symbol: 交易對 (e.g., BTCUSDT)
            timeframe: 時間框架 (15m, 1h, 1d)
            days: 往前抽取多少天
            include_oi: 是否包含 OI 數據
            include_funding: 是否包含資金費率
        
        Returns:
            DataFrame with OHLCV + OI + Funding Rate
        """
        if symbol not in self.symbols:
            raise ValueError(f"Symbol {symbol} not supported. Available: {self.symbols}")
        if timeframe not in self.timeframes:
            raise ValueError(f"Timeframe {timeframe} not supported. Available: {self.timeframes}")
        
        logger.info(f"Fetching {symbol} {timeframe} (OI={include_oi}, Funding={include_funding})")
        
        # 計算開始時間
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        # 轉換時間框架格式
        interval_map = {
            '15m': Client.KLINE_INTERVAL_15MINUTE,
            '1h': Client.KLINE_INTERVAL_1HOUR,
            '1d': Client.KLINE_INTERVAL_1DAY
        }
        interval = interval_map.get(timeframe)
        
        try:
            client = self._get_binance_client()
            
            # 獲取 K 線數據
            klines = client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_time.strftime('%Y-%m-%d'),
                end_str=end_time.strftime('%Y-%m-%d')
            )
            
            # 轉換為 DataFrame
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades',
                'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
            ])
            
            # 轉換數據類型
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 
                          'quote_volume', 'taker_buy_volume', 'taker_buy_quote_volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['trades'] = pd.to_numeric(df['trades'], errors='coerce').astype(int)
            
            # 加入 OI 數據
            if include_oi:
                df_oi = self.fetch_open_interest(symbol, timeframe, days)
                if len(df_oi) > 0:
                    df = pd.merge_asof(
                        df.sort_values('open_time'),
                        df_oi.sort_values('timestamp'),
                        left_on='open_time',
                        right_on='timestamp',
                        direction='nearest',
                        tolerance=pd.Timedelta('5min')
                    )
                    df = df.drop(columns=['timestamp'], errors='ignore')
                    logger.info("Merged OI data successfully")
                else:
                    df['open_interest'] = 0
                    df['open_interest_value'] = 0
            
            # 加入資金費率
            if include_funding:
                df_funding = self.fetch_funding_rate(symbol, days)
                if len(df_funding) > 0:
                    df = pd.merge_asof(
                        df.sort_values('open_time'),
                        df_funding.sort_values('timestamp'),
                        left_on='open_time',
                        right_on='timestamp',
                        direction='backward',
                        tolerance=pd.Timedelta('8h')
                    )
                    df = df.drop(columns=['timestamp'], errors='ignore')
                    df['funding_rate'] = df['funding_rate'].fillna(method='ffill')
                    logger.info("Merged funding rate data successfully")
                else:
                    df['funding_rate'] = 0
            
            df = df.sort_values('open_time').reset_index(drop=True)
            
            logger.info(f"Fetched {len(df)} rows from Binance for {symbol} {timeframe}")
            logger.info(f"Time range: {df['open_time'].min()} to {df['open_time'].max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch from Binance: {str(e)}")
            logger.warning("Falling back to HuggingFace cached data...")
            return self.load_klines(symbol, timeframe)
    
    def load_klines(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """從 HuggingFace 載入緩存數據"""
        if symbol not in self.symbols:
            raise ValueError(f"Symbol {symbol} not supported. Available: {self.symbols}")
        if timeframe not in self.timeframes:
            raise ValueError(f"Timeframe {timeframe} not supported. Available: {self.timeframes}")
        
        base = symbol.replace("USDT", "")
        filename = f"{base}_{timeframe}.parquet"
        path_in_repo = f"klines/{symbol}/{filename}"
        
        try:
            local_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=path_in_repo,
                repo_type="dataset"
            )
            df = pd.read_parquet(local_path)
            df['open_time'] = pd.to_datetime(df['open_time'])
            df['close_time'] = pd.to_datetime(df['close_time'])
            df = df.sort_values('open_time').reset_index(drop=True)
            logger.info(f"Loaded {len(df)} rows for {symbol} {timeframe} from HuggingFace")
            return df
        except Exception as e:
            logger.error(f"Failed to load {symbol} {timeframe}: {str(e)}")
            raise
    
    def load_multiple(self, symbols: List[str], timeframe: str) -> dict:
        result = {}
        for symbol in symbols:
            try:
                result[symbol] = self.load_klines(symbol, timeframe)
            except Exception as e:
                logger.warning(f"Skipped {symbol}: {str(e)}")
        return result
    
    def get_available_symbols(self) -> List[str]:
        return self.symbols.copy()
    
    def get_available_timeframes(self) -> List[str]:
        return self.timeframes.copy()