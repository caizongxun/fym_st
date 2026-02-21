# System Configuration Constants

class SystemConfig:
    """System-wide configuration"""
    VERSION = "2.0.0"
    SYSTEM_NAME = "BB+NW Swing Reversal Trading System"
    DEFAULT_SYMBOL = "BTCUSDT"
    DEFAULT_TIMEFRAME_15M = "15m"
    DEFAULT_TIMEFRAME_1H = "1h"

class FeatureConfig:
    """Feature engineering configuration"""
    # Bollinger Bands
    BB_PERIOD = 20
    BB_STD = 2.0
    
    # Nadaraya-Watson
    NW_H = 8.0
    NW_MULT = 3.0
    NW_WINDOW = 50
    
    # ADX
    ADX_PERIOD = 14
    ADX_TREND_THRESHOLD = 25
    
    # ATR
    ATR_PERIOD = 14
    
    # EMA
    EMA_FAST = 9
    EMA_MID = 21
    EMA_SLOW = 50
    
    # Volume
    VOLUME_MA_PERIOD = 20
    VOLUME_SURGE_THRESHOLD = 1.5
    
    # RSI
    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30

class FilterConfig:
    """Event filter configuration"""
    # BBNW Bounce Filter
    MIN_PIERCE_PCT = 0.001  # 0.1%
    REQUIRE_VOLUME_SURGE = False
    MIN_VOLUME_RATIO = 1.2
    
    # Event Filter
    MIN_SAMPLES_PCT = 0.05  # 5%
    MAX_SAMPLES_PCT = 0.25  # 25%
    TARGET_SAMPLES_PCT = 0.10  # 10%

class ModelConfig:
    """Model training configuration"""
    # Cross-validation
    CV_FOLDS = 5
    PURGE_GAP = 24
    EMBARGO_PCT = 0.01
    
    # Training
    EARLY_STOPPING_ROUNDS = 50
    RANDOM_STATE = 42
    
    # Model types
    SUPPORTED_MODELS = ['lightgbm', 'xgboost', 'catboost']
    DEFAULT_MODEL = 'lightgbm'
    
    # Feature selection
    MIN_FEATURE_IMPORTANCE = 0.001
    MAX_FEATURES = 150

class LabelConfig:
    """Labeling configuration for Triple Barrier"""
    # Default TP/SL multipliers
    DEFAULT_TP_MULTIPLIER = 3.0
    DEFAULT_SL_MULTIPLIER = 1.0
    
    # Max holding period (in bars)
    DEFAULT_MAX_HOLD_BARS = 60  # 15m * 60 = 15 hours
    
    # Minimum acceptable label ratio
    MIN_POSITIVE_LABEL_RATIO = 0.20
    MAX_POSITIVE_LABEL_RATIO = 0.80

class BacktestConfig:
    """Backtesting configuration"""
    # Initial capital
    DEFAULT_INITIAL_CAPITAL = 10000.0
    
    # Position sizing
    DEFAULT_POSITION_SIZE_PCT = 10.0
    MIN_POSITION_SIZE_PCT = 5.0
    MAX_POSITION_SIZE_PCT = 50.0
    
    # Trading costs
    DEFAULT_SLIPPAGE_PCT = 0.05
    DEFAULT_COMMISSION_PCT = 0.04  # Binance Maker
    
    # Risk management
    DEFAULT_MAX_DRAWDOWN_THRESHOLD = 0.30  # 30%
    DEFAULT_RISK_PER_TRADE = 0.02  # 2%
    
    # Probability threshold
    DEFAULT_PROB_THRESHOLD = 0.60
    MIN_PROB_THRESHOLD = 0.50
    MAX_PROB_THRESHOLD = 0.85

class PerformanceThresholds:
    """Performance evaluation thresholds"""
    # Win rate
    TARGET_WIN_RATE = 0.60
    MIN_ACCEPTABLE_WIN_RATE = 0.50
    MAX_ACCEPTABLE_WIN_RATE = 0.70
    
    # Profit factor
    TARGET_PROFIT_FACTOR = 2.0
    MIN_ACCEPTABLE_PROFIT_FACTOR = 1.5
    
    # Sharpe ratio
    TARGET_SHARPE_RATIO = 2.0
    MIN_ACCEPTABLE_SHARPE_RATIO = 1.0
    
    # Max drawdown
    MAX_ACCEPTABLE_DRAWDOWN = 0.25  # 25%
    
    # ROI (annualized)
    TARGET_ANNUAL_ROI = 0.30  # 30%

class DataConfig:
    """Data loading and processing configuration"""
    # HuggingFace dataset
    HF_DATASET_REPO = "bitmind/crypto-ohlcv"
    HF_DATASET_SUBSET = "15m"
    
    # Date ranges
    TRAIN_START_DATE = "2021-01-01"
    TRAIN_END_DATE = "2023-12-31"
    OOS_START_DATE = "2024-01-01"
    OOS_END_DATE = "2024-12-31"
    
    # Data quality
    MAX_MISSING_PCT = 0.05  # 5%
    MIN_TRADING_DAYS = 90

class UIConfig:
    """UI configuration (no emoji version)"""
    # Page titles
    PAGE_TITLE_DASHBOARD = "Dashboard"
    PAGE_TITLE_TRAINING = "Model Training"
    PAGE_TITLE_BACKTEST = "Backtesting Analysis"
    PAGE_TITLE_CALIBRATION = "Probability Calibration"
    PAGE_TITLE_OPTIMIZATION = "Strategy Optimization"
    PAGE_TITLE_LIQUIDITY = "Liquidity Analysis"
    PAGE_TITLE_LIVE = "Live Prediction"
    
    # Status messages
    STATUS_SUCCESS = "Success"
    STATUS_ERROR = "Error"
    STATUS_WARNING = "Warning"
    STATUS_INFO = "Info"
    
    # Progress steps
    PROGRESS_LOAD_DATA = "Loading data"
    PROGRESS_BUILD_FEATURES = "Building features"
    PROGRESS_FILTER_EVENTS = "Filtering events"
    PROGRESS_LABEL_DATA = "Creating labels"
    PROGRESS_TRAIN_MODEL = "Training model"
    PROGRESS_EVALUATE = "Evaluating performance"

class PathConfig:
    """File and directory paths"""
    MODELS_DIR = "models"
    DATA_DIR = "data"
    LOGS_DIR = "logs"
    REPORTS_DIR = "reports"
    
    # File extensions
    MODEL_EXT = ".pkl"
    DATA_EXT = ".parquet"
    LOG_EXT = ".log"
    REPORT_EXT = ".html"