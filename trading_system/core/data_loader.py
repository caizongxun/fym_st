from huggingface_hub import hf_hub_download
import pandas as pd
from typing import List, Optional
import logging

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
    
    def load_klines(self, symbol: str, timeframe: str) -> pd.DataFrame:
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
            logger.info(f"Loaded {len(df)} rows for {symbol} {timeframe}")
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