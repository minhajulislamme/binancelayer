import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API configuration
API_KEY = os.getenv('BINANCE_API_KEY', '')
API_SECRET = os.getenv('BINANCE_API_SECRET', '')

# Testnet configuration
API_TESTNET = os.getenv('BINANCE_API_TESTNET', 'False').lower() == 'true'

# API URLs - Automatically determined based on testnet setting
if API_TESTNET:
    # Testnet URLs
    API_URL = 'https://testnet.binancefuture.com'
    WS_BASE_URL = 'wss://stream.binancefuture.com'
else:
    # Production URLs
    API_URL = os.getenv('BINANCE_API_URL', 'https://fapi.binance.com')
    WS_BASE_URL = 'wss://fstream.binance.com'

# API request settings
RECV_WINDOW = int(os.getenv('BINANCE_RECV_WINDOW', '10000'))

# Trading parameters - LAYER only
TRADING_SYMBOL = os.getenv('TRADING_SYMBOL', 'LAYERUSDT')
TRADING_TYPE = 'FUTURES'  # Use futures trading
LEVERAGE = int(os.getenv('LEVERAGE', '10'))
MARGIN_TYPE = os.getenv('MARGIN_TYPE', 'ISOLATED')  # ISOLATED or CROSSED
STRATEGY = os.getenv('STRATEGY', 'LayerDynamicGrid')

# LAYER-specific strategy settings
LAYER_GRID_LEVELS = int(os.getenv('LAYER_GRID_LEVELS', '5'))
LAYER_GRID_SPACING_PCT = float(os.getenv('LAYER_GRID_SPACING_PCT', '1.2'))
LAYER_TREND_EMA_FAST = int(os.getenv('LAYER_TREND_EMA_FAST', '8'))
LAYER_TREND_EMA_SLOW = int(os.getenv('LAYER_TREND_EMA_SLOW', '21'))
LAYER_VOLATILITY_LOOKBACK = int(os.getenv('LAYER_VOLATILITY_LOOKBACK', '20'))
LAYER_VOLUME_MA_PERIOD = int(os.getenv('LAYER_VOLUME_MA_PERIOD', '20'))
# LAYER-specific advanced parameters
LAYER_VOLATILITY_MULTIPLIER = float(os.getenv('LAYER_VOLATILITY_MULTIPLIER', '1.1'))
LAYER_TREND_CONDITION_MULTIPLIER = float(os.getenv('LAYER_TREND_CONDITION_MULTIPLIER', '1.3'))
LAYER_MIN_GRID_SPACING = float(os.getenv('LAYER_MIN_GRID_SPACING', '0.6'))
LAYER_MAX_GRID_SPACING = float(os.getenv('LAYER_MAX_GRID_SPACING', '3.5'))

# AVAX-specific strategy settings
AVAX_GRID_LEVELS = int(os.getenv('AVAX_GRID_LEVELS', '6'))  # Higher volatility, more grid levels
AVAX_GRID_SPACING_PCT = float(os.getenv('AVAX_GRID_SPACING_PCT', '1.5'))  # Wider grid spacing for AVAX
AVAX_TREND_EMA_FAST = int(os.getenv('AVAX_TREND_EMA_FAST', '7'))  # Slightly faster trend detection
AVAX_TREND_EMA_SLOW = int(os.getenv('AVAX_TREND_EMA_SLOW', '21'))
AVAX_VOLATILITY_LOOKBACK = int(os.getenv('AVAX_VOLATILITY_LOOKBACK', '14'))  # Shorter lookback for more responsive
AVAX_VOLUME_MA_PERIOD = int(os.getenv('AVAX_VOLUME_MA_PERIOD', '20'))
# AVAX-specific advanced parameters
AVAX_VOLATILITY_MULTIPLIER = float(os.getenv('AVAX_VOLATILITY_MULTIPLIER', '1.2'))  # Higher volatility multiplier
AVAX_TREND_CONDITION_MULTIPLIER = float(os.getenv('AVAX_TREND_CONDITION_MULTIPLIER', '1.4'))  # More aggressive in trends
AVAX_MIN_GRID_SPACING = float(os.getenv('AVAX_MIN_GRID_SPACING', '0.8'))  # Higher min spacing
AVAX_MAX_GRID_SPACING = float(os.getenv('AVAX_MAX_GRID_SPACING', '4.0'))  # Higher max spacing

# Market condition detection settings
LAYER_ADX_PERIOD = int(os.getenv('LAYER_ADX_PERIOD', '14'))
LAYER_ADX_THRESHOLD = int(os.getenv('LAYER_ADX_THRESHOLD', '25'))
LAYER_SIDEWAYS_THRESHOLD = int(os.getenv('LAYER_SIDEWAYS_THRESHOLD', '15'))

# AVAX market condition detection settings
AVAX_ADX_PERIOD = int(os.getenv('AVAX_ADX_PERIOD', '14'))
AVAX_ADX_THRESHOLD = int(os.getenv('AVAX_ADX_THRESHOLD', '23'))  # Slightly more sensitive to trend
AVAX_SIDEWAYS_THRESHOLD = int(os.getenv('AVAX_SIDEWAYS_THRESHOLD', '17'))  # More tolerance in sideways

# Position sizing
INITIAL_BALANCE = float(os.getenv('INITIAL_BALANCE', '50.0'))
RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', '0.10'))
MAX_OPEN_POSITIONS = int(os.getenv('MAX_OPEN_POSITIONS', '6'))

# Multi-instance configuration for running separate bot instances per trading pair
MULTI_INSTANCE_MODE = os.getenv('MULTI_INSTANCE_MODE', 'True').lower() == 'true'
MAX_POSITIONS_PER_SYMBOL = int(os.getenv('MAX_POSITIONS_PER_SYMBOL', '3'))

# Auto-compounding settings
AUTO_COMPOUND = os.getenv('AUTO_COMPOUND', 'True').lower() == 'true'
COMPOUND_REINVEST_PERCENT = float(os.getenv('COMPOUND_REINVEST_PERCENT', '0.75'))
COMPOUND_INTERVAL = os.getenv('COMPOUND_INTERVAL', 'DAILY')

# Technical indicator parameters
RSI_PERIOD = int(os.getenv('RSI_PERIOD', '14'))
RSI_OVERBOUGHT = int(os.getenv('RSI_OVERBOUGHT', '70'))
RSI_OVERSOLD = int(os.getenv('RSI_OVERSOLD', '30'))
FAST_EMA = int(os.getenv('FAST_EMA', '8'))
SLOW_EMA = int(os.getenv('SLOW_EMA', '21'))
TIMEFRAME = os.getenv('TIMEFRAME', '15m')

# Risk management - Standard settings
USE_STOP_LOSS = os.getenv('USE_STOP_LOSS', 'True').lower() == 'true'
STOP_LOSS_PCT = float(os.getenv('STOP_LOSS_PCT', '0.02'))
USE_TAKE_PROFIT = os.getenv('USE_TAKE_PROFIT', 'True').lower() == 'true'
TAKE_PROFIT_PCT = float(os.getenv('TAKE_PROFIT_PCT', '0.06'))
TRAILING_STOP = os.getenv('TRAILING_STOP', 'True').lower() == 'true'
TRAILING_STOP_PCT = float(os.getenv('TRAILING_STOP_PCT', '0.02'))
TRAILING_TAKE_PROFIT = os.getenv('TRAILING_TAKE_PROFIT', 'True').lower() == 'true'
TRAILING_TAKE_PROFIT_PCT = float(os.getenv('TRAILING_TAKE_PROFIT_PCT', '0.03'))

# Adaptive risk management settings for different market conditions
STOP_LOSS_PCT_BULLISH = float(os.getenv('STOP_LOSS_PCT_BULLISH', '0.02'))
STOP_LOSS_PCT_BEARISH = float(os.getenv('STOP_LOSS_PCT_BEARISH', '0.015'))
STOP_LOSS_PCT_SIDEWAYS = float(os.getenv('STOP_LOSS_PCT_SIDEWAYS', '0.01'))

TAKE_PROFIT_PCT_BULLISH = float(os.getenv('TAKE_PROFIT_PCT_BULLISH', '0.06'))
TAKE_PROFIT_PCT_BEARISH = float(os.getenv('TAKE_PROFIT_PCT_BEARISH', '0.04'))
TAKE_PROFIT_PCT_SIDEWAYS = float(os.getenv('TAKE_PROFIT_PCT_SIDEWAYS', '0.02'))

TRAILING_STOP_PCT_BULLISH = float(os.getenv('TRAILING_STOP_PCT_BULLISH', '0.02'))
TRAILING_STOP_PCT_BEARISH = float(os.getenv('TRAILING_STOP_PCT_BEARISH', '0.015'))
TRAILING_STOP_PCT_SIDEWAYS = float(os.getenv('TRAILING_STOP_PCT_SIDEWAYS', '0.01'))

TRAILING_TAKE_PROFIT_PCT_BULLISH = float(os.getenv('TRAILING_TAKE_PROFIT_PCT_BULLISH', '0.03'))
TRAILING_TAKE_PROFIT_PCT_BEARISH = float(os.getenv('TRAILING_TAKE_PROFIT_PCT_BEARISH', '0.02'))
TRAILING_TAKE_PROFIT_PCT_SIDEWAYS = float(os.getenv('TRAILING_TAKE_PROFIT_PCT_SIDEWAYS', '0.015'))

# Backtesting parameters
BACKTEST_START_DATE = os.getenv('BACKTEST_START_DATE', '2023-01-01')
BACKTEST_END_DATE = os.getenv('BACKTEST_END_DATE', '')  # Empty means use current date
BACKTEST_INITIAL_BALANCE = float(os.getenv('BACKTEST_INITIAL_BALANCE', '50.0'))
BACKTEST_COMMISSION = float(os.getenv('BACKTEST_COMMISSION', '0.0004'))  # 0.04% taker fee
BACKTEST_USE_AUTO_COMPOUND = os.getenv('BACKTEST_USE_AUTO_COMPOUND', 'True').lower() == 'true'

# Pre-live backtest validation
BACKTEST_BEFORE_LIVE = os.getenv('BACKTEST_BEFORE_LIVE', 'True').lower() == 'true'
BACKTEST_MIN_PROFIT_PCT = float(os.getenv('BACKTEST_MIN_PROFIT_PCT', '5.0'))
BACKTEST_MIN_WIN_RATE = float(os.getenv('BACKTEST_MIN_WIN_RATE', '40.0'))
BACKTEST_PERIOD = os.getenv('BACKTEST_PERIOD', '15 days')

# Logging and notifications
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
USE_TELEGRAM = os.getenv('USE_TELEGRAM', 'True').lower() == 'true'
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
SEND_DAILY_REPORT = os.getenv('SEND_DAILY_REPORT', 'True').lower() == 'true'
DAILY_REPORT_TIME = os.getenv('DAILY_REPORT_TIME', '00:00')  # 24-hour format

# Other settings
RETRY_COUNT = int(os.getenv('RETRY_COUNT', '3'))
RETRY_DELAY = int(os.getenv('RETRY_DELAY', '5'))  # seconds