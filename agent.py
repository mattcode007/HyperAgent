# agent.py
import torch, torch.nn as nn, torch.optim as optim, numpy as np
from collections import deque
import random, json, logging, argparse, time as a_time, os, sys
import joblib
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import config, database
from model import TransformerDQN
from environment import TradingEnv

# --- LOGGING SETUP ---
# ... (logging setup remains the same)

# --- REPLAY BUFFER & TRAINING ---
# ... (ReplayBuffer and train_agent functions remain the same)

# --- BACKTEST PROCESS ---
# ... (run_backtest_process function remains the same)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Apex Predator RL Trading Agent")
    parser.add_argument('--train', action='store_true', help='Train a new agent model.')
    parser.add_argument('--run', action='store_true', help='Run a backtest on the trained agent.')
    args = parser.parse_args()

    # FIX: Add robust error checking for data fetching
    success15m = database.fetch_and_store_data(config.SYMBOL, config.SIGNAL_TIMEFRAME)
    if not success15m:
        logging.error("Failed to fetch 15m data. Aborting process.")
        sys.exit(1) # Exit with an error code

    df_15m = database.get_data_from_db(config.SYMBOL, config.SIGNAL_TIMEFRAME)
    
    if df_15m.empty:
        logging.error("15m data is empty after fetch. Cannot proceed.")
        sys.exit(1)
    else:
        df_15m_featured = database.create_features(df_15m.copy())
        split_index = int(len(df_15m_featured) * 0.75)
        train_df, test_df = df_15m_featured.iloc[:split_index], df_15m_featured.iloc[split_index:]
        logging.info(f"Data split: {len(train_df)} training samples, {len(test_df)} testing samples.")

        if args.train:
            train_agent(train_df)
        elif args.run:
            if not os.path.exists(config.MODEL_FILE):
                logging.error("Model file not found. Please train the agent first with --train.")
            else:
                run_backtest_process(test_df)
        else:
            print("Please specify mode: --train or --run.")