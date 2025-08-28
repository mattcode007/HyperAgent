# config.py
import os
import torch

# --- FILE PATHS (DO NOT CHANGE) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(SCRIPT_DIR, "trading_data.db")
MODEL_FILE = os.path.join(SCRIPT_DIR, "agent_model.pth")
TRAINING_HISTORY_FILE = os.path.join(SCRIPT_DIR, "training_history.json")
SCALER_FILE = os.path.join(SCRIPT_DIR, "scaler.joblib")
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
AGENT_LOG_FILE = os.path.join(LOG_DIR, "agent.log")

# --- EXECUTION & HARDWARE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- GITHUB CONFIG (Fill these in for automated pushes) ---
GITHUB_USERNAME = "mattcode007"
REPO_NAME = "HyperAgent"

# --- EXCHANGE & DATA ---
EXCHANGE_ID = 'bybit'
SYMBOL = 'BTC/USDT'
SIGNAL_TIMEFRAME = '15m'
FETCH_DAYS_INITIAL = 365

# --- RL & STRATEGY PARAMETERS ---
SEQUENCE_LENGTH = 96
TRAINING_EPISODES = 50
INITIAL_ACCOUNT_BALANCE = 10000.0
LEVERAGE = 10.0
ATR_PERIOD = 14
RISK_REWARD_RATIO = 2.0
RSI_LAG = 3
VOLATILITY_WINDOW = 96

# --- DQN AGENT PARAMETERS ---
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.0001
REPLAY_BUFFER_SIZE = 50000
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 10

# --- TRANSFORMER MODEL PARAMETERS ---
D_MODEL = 64
N_HEAD = 4
N_LAYERS = 2