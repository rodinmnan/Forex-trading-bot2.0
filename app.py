import os
import logging
import time
import threading
import random
import numpy as np
from datetime import datetime, timedelta
import requests
import pandas as pd
import pytz
from telegram import Update, ParseMode
from telegram.ext import Updater, CommandHandler, CallbackContext
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import talib
from flask import Flask, jsonify, request
import gunicorn  # Required for fly.io deployment

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

# API Configuration
TRADEMADE_API_KEY = os.getenv("TRADEMADE_API_KEY")
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
ADMIN_ID = os.getenv("ADMIN_ID", "")
FLY_APP_URL = os.getenv("FLY_APP_URL", "")  # fly.io app URL

# Validate critical environment variables
if not TRADEMADE_API_KEY:
    logger.error("Missing TRADEMADE_API_KEY environment variable")
if not TWELVE_DATA_API_KEY:
    logger.error("Missing TWELVE_DATA_API_KEY environment variable")
if not TELEGRAM_TOKEN:
    logger.error("Missing TELEGRAM_TOKEN environment variable")

# Trading parameters
PAIRS = os.getenv("TRADING_PAIRS", "XAUUSD,GBPJPY,AUDUSD,GBPUSD,EURUSD,USDJPY").split(',')
NEW_YORK_TZ = pytz.timezone('America/New_York')
RISK_REWARD_RATIO = float(os.getenv("RISK_REWARD_RATIO", 3.0))
CACHE_DURATION = int(os.getenv("CACHE_DURATION", 60))
NEWS_CHECK_INTERVAL = int(os.getenv("NEWS_CHECK_INTERVAL", 1800))
VOLATILITY_LOOKBACK = int(os.getenv("VOLATILITY_LOOKBACK", 14))
TREND_FILTER_MODE = os.getenv("TREND_FILTER", "strict")
SCALPING_MODE = os.getenv("SCALPING_MODE", "hybrid")  # sniper-only/standard-only/hybrid
SNIPER_PAIRS = os.getenv("SNIPER_PAIRS", "XAUUSD,EURUSD").split(',')
STANDARD_PAIRS = os.getenv("STANDARD_PAIRS", "GBPJPY,AUDUSD").split(',')
DMC_STRENGTH_THRESHOLD = int(os.getenv("DMC_STRENGTH_THRESHOLD", 65))

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class HighProbabilityTradingBot:
    def __init__(self):
        # Initialize shared resources with thread safety
        self.data_lock = threading.RLock()
        self.live_prices = {pair: {'price': None, 'timestamp': None} for pair in PAIRS}
        self.market_open = False
        self.high_impact_news = False
        self.signal_cooldown = {}
        self.performance = {
            'total_signals': 0,
            'tp1_hits': 0,
            'tp2_hits': 0,
            'tp3_hits': 0,
            'sl_hits': 0,
            'win_rate': 0.0,
            'fakeouts_filtered': 0,
            'dmc_signals': 0,
            'sniper_signals': 0,
            'sniper_wins': 0
        }
        self.active_signals = []
        self.subscribed_users = set()
        self.running = True
        self.trend_filters = {pair: None for pair in PAIRS}
        self.volume_profile = {}
        self.dmc_values = {pair: None for pair in PAIRS}
        self.sniper_conditions = {pair: False for pair in PAIRS}
        self.liquidity_zones = {pair: {'support': [], 'resistance': []} for pair in PAIRS}  # New
        
        # Initialize caches
        self.technical_cache = {pair: {'data': None, 'timestamp': 0} for pair in PAIRS}
        self.volatility_cache = {pair: {'data': None, 'timestamp': 0} for pair in PAIRS}
        self.price_history_cache = {pair: {'data': None, 'timestamp': 0} for pair in PAIRS}
        self.tick_data_cache = {pair: {'data': [], 'timestamp': 0} for pair in PAIRS}
        self.liquidity_cache = {pair: {'data': None, 'timestamp': 0} for pair in PAIRS}  # New
        
        # Configure API session
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        
        # Initialize Telegram
        self.updater = Updater(TELEGRAM_TOKEN, use_context=True)
        
        # Configure for fly.io
        if FLY_APP_URL:
            try:
                self.updater.bot.set_webhook(f"{FLY_APP_URL}/{TELEGRAM_TOKEN}")
                logger.info(f"Telegram webhook configured for fly.io at {FLY_APP_URL}/{TELEGRAM_TOKEN}")
            except Exception as e:
                logger.error(f"Webhook setup failed: {str(e)}")
                logger.info("Falling back to polling mode")
                self.updater.start_polling()
        else:
            logger.info("Using polling mode for Telegram")
            self.updater.start_polling()
        
        # Start services
        self.start_services()

    # ======================
    # ENHANCED SCALPING STRATEGIES
    # ======================
    
    def liquidity_analyzer(self):
        """Identify key liquidity zones for precision entries"""
        while self.running:
            try:
                for pair in PAIRS:
                    if not self.market_open:
                        time.sleep(60)
                        continue
                    
                    # Get price history
                    df = self.get_price_history(pair, interval='15min', lookback=200)
                    if df is None or len(df) < 50:
                        continue
                    
                    # Calculate support/resistance using fractal method
                    highs = df['high'].values
                    lows = df['low'].values
                    
                    # Fractal detection
                    support_levels = []
                    resistance_levels = []
                    
                    for i in range(2, len(df)-2):
                        # Support fractal
                        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
                           lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                            support_levels.append(lows[i])
                        
                        # Resistance fractal
                        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
                           highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                            resistance_levels.append(highs[i])
                    
                    # Cluster similar levels
                    if support_levels:
                        support_clusters = self.cluster_levels(support_levels, tolerance=0.0005)
                        strongest_support = max(support_clusters, key=lambda x: len(x)) if support_clusters else []
                    else:
                        strongest_support = []
                        
                    if resistance_levels:
                        resistance_clusters = self.cluster_levels(resistance_levels, tolerance=0.0005)
                        strongest_resistance = max(resistance_clusters, key=lambda x: len(x)) if resistance_clusters else []
                    else:
                        strongest_resistance = []
                    
                    # Calculate fair value gap (FVG) for imbalance zones
                    fvg_zones = []
                    for i in range(1, len(df)-1):
                        if df['low'].iloc[i] > df['high'].iloc[i+1]:  # Bullish FVG
                            fvg_zones.append((
                                df['high'].iloc[i+1],
                                df['low'].iloc[i]
                            ))
                        elif df['high'].iloc[i] < df['low'].iloc[i+1]:  # Bearish FVG
                            fvg_zones.append((
                                df['high'].iloc[i],
                                df['low'].iloc[i+1]
                            ))
                    
                    # Update liquidity zones
                    with self.data_lock:
                        self.liquidity_zones[pair] = {
                            'support': strongest_support,
                            'resistance': strongest_resistance,
                            'fvg': fvg_zones
                        }
                        
            except Exception as e:
                logger.error(f"Liquidity analysis failed: {str(e)}")
            time.sleep(600)  # Update every 10 minutes

    def cluster_levels(self, levels, tolerance=0.0005):
        """Cluster similar price levels"""
        clusters = []
        levels.sort()
        
        current_cluster = [levels[0]]
        for price in levels[1:]:
            if price <= current_cluster[-1] + tolerance:
                current_cluster.append(price)
            else:
                clusters.append(current_cluster)
                current_cluster = [price]
                
        clusters.append(current_cluster)
        return clusters

    def check_sniper_conditions(self, pair):
        """Enhanced sniper entry with liquidity confirmation"""
        if self.is_cooldown(pair) or self.is_news_blackout(pair):
            return None
            
        # Get recent price action
        with self.data_lock:
            ticks = self.tick_data_cache[pair]['data']
            if len(ticks) < 10:
                return None
                
            prices = [tick['price'] for tick in ticks]
            liquidity = self.liquidity_zones.get(pair, {})
            current_price = prices[-1]
        
        # Calculate short-term volatility
        price_changes = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
        volatility = np.mean(price_changes) if price_changes else 0
        
        # Calculate momentum
        momentum = prices[-1] - prices[0]
        
        # Calculate EMA cross
        if len(prices) >= 5:
            ema5 = talib.EMA(np.array(prices), timeperiod=5)[-1]
            ema10 = talib.EMA(np.array(prices), timeperiod=10)[-1]
            ema_cross = ema5 - ema10
        else:
            ema_cross = 0
            
        # Determine sniper entry conditions
        direction = None
        confidence = 0.0
        
        # Condition 1: Liquidity Grab with Reversal
        for support in liquidity.get('support', []):
            if current_price <= support * 1.0003 and current_price >= support * 0.9997:
                # Price at support level
                if momentum > volatility * 1.5 and ema_cross > 0:
                    direction = 'BUY'
                    confidence = 0.88
                    break
                    
        for resistance in liquidity.get('resistance', []):
            if current_price >= resistance * 0.9997 and current_price <= resistance * 1.0003:
                # Price at resistance level
                if momentum < -volatility * 1.5 and ema_cross < 0:
                    direction = 'SELL'
                    confidence = 0.88
                    break
        
        # Condition 2: Fair Value Gap (FVG) Fill
        if not direction:
            for low, high in liquidity.get('fvg', []):
                if low <= current_price <= high:
                    # Price filling FVG
                    if momentum > volatility * 2:
                        direction = 'BUY'
                        confidence = 0.85
                    elif momentum < -volatility * 2:
                        direction = 'SELL'
                        confidence = 0.85
                    break
        
        # Condition 3: EMA Cross Momentum (fallback)
        if not direction:
            if momentum > (volatility * 3) and ema_cross > 0:
                direction = 'BUY'
                confidence = 0.82
            elif momentum < -(volatility * 3) and ema_cross < 0:
                direction = 'SELL'
                confidence = 0.82
        
        # Condition 4: Volume Confirmation (if available)
        if direction:
            # Get volume data
            df = self.get_price_history(pair, interval='1min', lookback=5)
            if df is not None and 'volume' in df.columns:
                current_volume = df['volume'].iloc[-1]
                avg_volume = df['volume'].iloc[:-1].mean()
                if current_volume > avg_volume * 1.5:
                    confidence = min(confidence * 1.1, 0.95)
        
        # If conditions met, create sniper signal
        if direction and confidence >= 0.82:
            return self.create_sniper_signal(pair, direction, current_price, confidence)
            
        return None

    def create_sniper_signal(self, pair, direction, entry, confidence):
        """Create sniper signal with liquidity-based targets"""
        now_ny = datetime.now(NEW_YORK_TZ)
        
        # Get liquidity zones for precision targets
        with self.data_lock:
            liquidity = self.liquidity_zones.get(pair, {})
        
        # Determine targets based on liquidity
        if direction == 'BUY':
            # Find nearest resistance above entry
            resistances = [r for r in liquidity.get('resistance', []) if r > entry]
            if resistances:
                nearest_resistance = min(resistances)
                tp = nearest_resistance * 0.9998  # Just below resistance
                sl = min(liquidity.get('support', [entry * 0.999]))  # Nearest support
            else:
                # Default if no liquidity zones
                tp = entry * 1.0003
                sl = entry * 0.9997
        else:  # SELL
            # Find nearest support below entry
            supports = [s for s in liquidity.get('support', []) if s < entry]
            if supports:
                nearest_support = max(supports)
                tp = nearest_support * 1.0002  # Just above support
                sl = max(liquidity.get('resistance', [entry * 1.001]))  # Nearest resistance
            else:
                # Default if no liquidity zones
                tp = entry * 0.9997
                sl = entry * 1.0003
        
        # Create signal object
        signal = {
            "pair": pair,
            "direction": direction,
            "strategy": "sniper",
            "entry": entry,
            "tp1": round(tp, 5) if pair != "XAUUSD" else round(tp, 2),
            "sl": round(sl, 5) if pair != "XAUUSD" else round(sl, 2),
            "expiry": (now_ny + timedelta(minutes=5)).isoformat(),
            "status": "active",
            "confidence": round(confidence, 2),
            "created_at": now_ny.isoformat(),
            "trailing_sl": {
                "activated": False,
                "activation_level": 0.7,
                "distance": abs(tp - entry) * 0.3  # 30% of TP distance
            }
        }
        
        # Add to active signals
        with self.data_lock:
            self.active_signals.append(signal)
            self.performance['total_signals'] += 1
            self.performance['sniper_signals'] += 1
            
        logger.info(f"⚡️ SNIPER signal for {pair} {direction} @ {entry}")
        return signal

    # ======================
    # WIN RATE BOOSTING TECHNIQUES
    # ======================
    
    def generate_signal(self, pair):
        """Enhanced signal generation with multi-confirmation"""
        # ... [existing code] ...
        
        # Add new confirmation: Liquidity Zone Alignment
        with self.data_lock:
            liquidity = self.liquidity_zones.get(pair, {})
        
        liquidity_boost = 0.0
        if direction == "BUY":
            # Check if near support
            for s in liquidity.get('support', []):
                if abs(s - current_price) / current_price < 0.0005:  # Within 0.05%
                    liquidity_boost = 0.1
                    break
        else:  # SELL
            # Check if near resistance
            for r in liquidity.get('resistance', []):
                if abs(r - current_price) / current_price < 0.0005:  # Within 0.05%
                    liquidity_boost = 0.1
                    break
        
        confidence = min(confidence + liquidity_boost, 0.95)
        
        # Add new confirmation: Time-of-Day Filter
        now_ny = datetime.now(NEW_YORK_TZ)
        hour = now_ny.hour
        
        # Peak volatility hours (NY-London overlap)
        if 8 <= hour <= 11 or 14 <= hour <= 17:
            confidence *= 1.05
        # Low volatility hours
        elif hour < 5 or hour >= 22:
            confidence *= 0.9
            
        # Only accept high-confidence signals
        if direction and confidence >= 0.85:  # Increased threshold
            return self.create_signal(pair, direction, 'technical', current_price, confidence)
                
        return None

    def create_signal(self, pair, direction, strategy, entry, confidence):
        """Create signal with liquidity-based targets"""
        # ... [existing code] ...
        
        # Get liquidity zones for precision targets
        with self.data_lock:
            liquidity = self.liquidity_zones.get(pair, {})
        
        # Determine targets based on liquidity
        if direction == 'BUY':
            # Find resistance levels above entry
            resistances = [r for r in liquidity.get('resistance', []) if r > entry]
            resistances.sort()
            
            if len(resistances) >= 3:
                tp1 = resistances[0] * 0.9998
                tp2 = resistances[1] * 0.9998
                tp3 = resistances[2] * 0.9998
            elif len(resistances) >= 2:
                tp1 = resistances[0] * 0.9998
                tp2 = resistances[1] * 0.9998
                tp3 = entry * (1 + 3 * volatility * volatility_factor)
            else:
                # Default strategy
                tp1 = entry * (1 + volatility * volatility_factor * tp_sl_ratio / 3)
                tp2 = entry * (1 + volatility * volatility_factor * tp_sl_ratio * 2 / 3)
                tp3 = entry * (1 + volatility * volatility_factor * tp_sl_ratio)
            
            # Set stop loss below nearest support
            supports = [s for s in liquidity.get('support', []) if s < entry]
            if supports:
                sl = max(supports) * 0.9995
            else:
                sl = entry * (1 - volatility * volatility_factor * tp_sl_ratio / 3)
                
        else:  # SELL
            # Find support levels below entry
            supports = [s for s in liquidity.get('support', []) if s < entry]
            supports.sort(reverse=True)
            
            if len(supports) >= 3:
                tp1 = supports[0] * 1.0002
                tp2 = supports[1] * 1.0002
                tp3 = supports[2] * 1.0002
            elif len(supports) >= 2:
                tp1 = supports[0] * 1.0002
                tp2 = supports[1] * 1.0002
                tp3 = entry * (1 - 3 * volatility * volatility_factor)
            else:
                # Default strategy
                tp1 = entry * (1 - volatility * volatility_factor * tp_sl_ratio / 3)
                tp2 = entry * (1 - volatility * volatility_factor * tp_sl_ratio * 2 / 3)
                tp3 = entry * (1 - volatility * volatility_factor * tp_sl_ratio)
            
            # Set stop loss above nearest resistance
            resistances = [r for r in liquidity.get('resistance', []) if r > entry]
            if resistances:
                sl = min(resistances) * 1.0005
            else:
                sl = entry * (1 + volatility * volatility_factor * tp_sl_ratio / 3)
        
        # ... [rest of existing code] ...

    # ======================
    # FLY.IO DEPLOYMENT OPTIMIZATION
    # ======================
    
    def start_services(self):
        """Initialize all background services with fly.io optimizations"""
        services = [
            self.market_hours_checker,
            self.batch_price_updater,
            self.news_monitor,
            self.trend_analyzer,
            self.volume_analyzer,
            self.liquidity_analyzer,  # New service
            self.signal_generator,
            self.signal_monitor,
            self.sniper_scalping_monitor
        ]
        
        # Start services with staggered initialization
        for i, service in enumerate(services):
            t = threading.Thread(target=self.run_service, args=(service,), daemon=True)
            t.start()
            time.sleep(1)  # Reduce CPU spike at startup

# ======================
# FLASK ROUTES & FLY.IO CONFIG
# ======================

bot = None

@app.route('/')
def home():
    return "High Probability Trading Bot is running on fly.io!"

@app.route('/health')
def health_check():
    return jsonify({
        "status": "running",
        "bot_initialized": bot is not None,
        "market_open": bot.market_open if bot else False,
        "active_signals": len(bot.active_signals) if bot else 0
    })

@app.route(f'/{TELEGRAM_TOKEN}', methods=['POST'])
def telegram_webhook():
    if not bot or not bot.running:
        return "Bot not initialized", 503
        
    json_data = request.get_json()
    update = Update.de_json(json_data, bot.updater.bot)
    bot.updater.dispatcher.process_update(update)
    return 'ok', 200

def initialize_bot():
    global bot
    bot = HighProbabilityTradingBot()
    logger.info("Bot initialized for fly.io")

# Initialize in separate thread
threading.Thread(target=initialize_bot, daemon=True).start()

# Run with Gunicorn on fly.io
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
