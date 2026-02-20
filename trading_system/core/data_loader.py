from huggingface_hub import hf_hub_download
import pandas as pd
from typing import List, Optional
import logging
from binance.client import Client
from datetime import datetime, timedelta
import os

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
        
        # Binance API client (使用公開 API,不需要 key)
        self.binance_client = None
    
    def _get_binance_client(self):
        """Lazy initialization of Binance client"""
        if self.binance_client is None:
            api_key = os.getenv('BINANCE_API_KEY', '')
            api_secret = os.getenv('BINANCE_API_SECRET', '')
            self.binance_client = Client(api_key, api_secret)
        return self.binance_client
    
    def fetch_latest_klines(self, symbol: str, timeframe: str, days: int = 90) -> pd.DataFrame:
        """
        從 Binance API 直接獲取最新 N 天的 K 線數據
        
        Args:
            symbol: 交易對 (e.g., BTCUSDT)
            timeframe: 時間框架 (15m, 1h, 1d)
            days: 往前抽取多少天
        
        Returns:
            DataFrame with OHLCV data
        """
        if symbol not in self.symbols:
            raise ValueError(f"Symbol {symbol} not supported. Available: {self.symbols}")
        if timeframe not in self.timeframes:
            raise ValueError(f"Timeframe {timeframe} not supported. Available: {self.timeframes}")
        
        logger.info(f"Fetching latest {days} days of {symbol} {timeframe} from Binance...")
        
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