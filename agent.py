# agent.py
import torch, torch.nn as nn, torch.optim as optim, numpy as np
from collections import deque
import random, json, logging, argparse, time as a_time, os, sys
import pytz, joblib
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pandas as pd # <-- FIX: Added this import

import config, database, utils
from model import TransformerDQN
from environment import TradingEnv

# --- LOGGING SETUP ---
os.makedirs(config.LOG_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(config.AGENT_LOG_FILE), logging.StreamHandler()])

# --- REPLAY BUFFER & TRAINING ---
class ReplayBuffer:
    def __init__(self, capacity): self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done): self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size): return random.sample(self.buffer, batch_size)
    def __len__(self): return len(self.buffer)

def train_agent(train_df):
    logging.info(f"Using device: {config.DEVICE}")
    env = TradingEnv(pd.DataFrame(), StandardScaler()) # Temp env to get features
    features = env.features
    scaler = StandardScaler().fit(train_df[features])
    joblib.dump(scaler, config.SCALER_FILE)
    logging.info(f"Scaler for {len(features)} features fitted and saved.")
    
    env = TradingEnv(train_df, scaler)
    n_features = env.observation_space.shape[1]; n_actions = env.action_space.n
    policy_net = TransformerDQN(n_features, n_actions).to(config.DEVICE); target_net = TransformerDQN(n_features, n_actions).to(config.DEVICE)
    target_net.load_state_dict(policy_net.state_dict()); target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=config.LEARNING_RATE)
    buffer = ReplayBuffer(config.REPLAY_BUFFER_SIZE); epsilon = config.EPSILON_START
    training_history, rewards_window = [], deque(maxlen=10)
    
    progress_bar = tqdm(range(config.TRAINING_EPISODES), desc="Training Progress")
    for episode in progress_bar:
        state, _ = env.reset()
        episode_reward = 0
        while True:
            state_t = torch.tensor([state], dtype=torch.float32).to(config.DEVICE)
            action = env.action_space.sample() if random.random() < epsilon else policy_net(state_t).max(1)[1].item()
            next_state, reward, done, _, _ = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            episode_reward += reward; state = next_state
            if len(buffer) > config.BATCH_SIZE:
                states, actions, rewards, next_states, dones = zip(*buffer.sample(config.BATCH_SIZE))
                states_t = torch.tensor(np.array(states), dtype=torch.float32).to(config.DEVICE)
                actions_t = torch.tensor(actions, dtype=torch.long).unsqueeze(-1).to(config.DEVICE)
                rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(config.DEVICE)
                next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32).to(config.DEVICE)
                dones_t = torch.tensor(dones, dtype=torch.bool).unsqueeze(-1).to(config.DEVICE)
                q_values = policy_net(states_t).gather(1, actions_t)
                next_actions = policy_net(next_states_t).max(1)[1].unsqueeze(-1)
                next_q_values = target_net(next_states_t).gather(1, next_actions)
                next_q_values[dones_t] = 0.0
                expected_q_values = rewards_t + config.GAMMA * next_q_values
                loss = nn.MSELoss()(q_values, expected_q_values)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            if done: break
        rewards_window.append(episode_reward); avg_reward = np.mean(rewards_window)
        epsilon = max(config.EPSILON_END, epsilon * config.EPSILON_DECAY)
        if episode % config.TARGET_UPDATE_FREQ == 0: target_net.load_state_dict(policy_net.state_dict())
        training_history.append({'episode': episode + 1, 'reward': episode_reward})
        progress_bar.set_description(f"Ep {episode+1} | Reward: {episode_reward:.2f} | Avg Reward: {avg_reward:.2f}")

    torch.save(policy_net.state_dict(), config.MODEL_FILE)
    with open(config.TRAINING_HISTORY_FILE, 'w') as f: json.dump(training_history, f)
    logging.info(f"Training complete! Model saved.")

def run_backtest_process(test_df):
    logging.info("Initializing Agent for Backtest...")
    try:
        scaler = joblib.load(config.SCALER_FILE)
        temp_env = TradingEnv(test_df.head(config.SEQUENCE_LENGTH+1), scaler)
        n_features, n_actions = len(temp_env.features), 3
        model = TransformerDQN(n_features, n_actions).to(config.DEVICE)
        model.load_state_dict(torch.load(config.MODEL_FILE)); model.eval()
        logging.info("Agent model and scaler loaded successfully.")
    except Exception as e: logging.error(f"Could not load files: {e}"); return
    
    env = TradingEnv(test_df, scaler)
    state, _ = env.reset()
    done = False
    conn = database.setup_database(); conn.execute("DELETE FROM trades"); conn.commit(); conn.close()
    logging.info("Starting backtest on unseen data...")
    while not done:
        with torch.no_grad():
            state_t = torch.tensor([state], dtype=torch.float32).to(config.DEVICE)
            action = model(state_t).max(1)[1].item()
        state, _, done, _, trade_info = env.step(action)
        if trade_info:
            database.log_trade_to_db(trade_info)
    logging.info("Backtest complete. Results saved to the database.")
    utils.push_results_to_github(f"Automated backtest results - {datetime.now().strftime('%Y-%m-%d %H:%M')}")

def run_paper_trader():
    logging.info("--- Initializing Agent for LIVE PAPER TRADING ---")
    try:
        scaler = joblib.load(config.SCALER_FILE)
        # This is a bit of a hack to get feature names without a full df
        dummy_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'atr', 'rsi', 'price_vs_ema'])
        temp_env = TradingEnv(dummy_df, scaler)
        n_features, n_actions = len(temp_env.features), 3
        model = TransformerDQN(n_features, n_actions).to(config.DEVICE)
        model.load_state_dict(torch.load(config.MODEL_FILE)); model.eval()
        logging.info("Agent model and scaler loaded successfully.")
    except Exception as e: logging.error(f"Could not load files: {e}"); return
    
    paper_balance, position, entry_price, entry_time, stop_loss, take_profit = config.INITIAL_ACCOUNT_BALANCE, 0, 0, None, 0, 0
    exchange = ccxt.bybit({'options': {'defaultType': 'swap'}})

    while True:
        now = datetime.now(pytz.utc); wait_seconds = (15 - (now.minute % 15)) * 60 - now.second
        logging.info(f"Waiting for {wait_seconds:.0f} seconds until next 15m candle...")
        a_time.sleep(wait_seconds)
        
        logging.info("New 15m candle. Syncing data and analyzing market...")
        database.fetch_and_store_data(config.SYMBOL, config.SIGNAL_TIMEFRAME)
        df_15m = database.get_data_from_db(config.SYMBOL, config.SIGNAL_TIMEFRAME)
        df_15m_featured = database.create_features(df_15m.copy())
        
        if position != 0:
            try:
                ticker = exchange.fetch_ticker(config.SYMBOL)
                current_price = ticker['last']
                exit_reason = None
                if position == 1 and current_price <= stop_loss: exit_reason = "Stop-Loss"
                elif position == 1 and current_price >= take_profit: exit_reason = "Take-Profit"
                # ... add short logic
                if exit_reason:
                    # ... log trade logic
                    position = 0
            except Exception as e: logging.error(f"Could not check SL/TP: {e}")

        if position == 0:
            latest_data = df_15m_featured.iloc[-config.SEQUENCE_LENGTH:]
            scaled_features = scaler.transform(latest_data[temp_env.features])
            state = torch.tensor([scaled_features], dtype=torch.float32).to(config.DEVICE)
            with torch.no_grad():
                action = model(state).max(1)[1].item()
                decision = {0: 'Hold', 1: 'Long', 2: 'Short'}[action]
            logging.info(f"Agent Decision: {decision}")
            if decision != 'Hold':
                # ... open paper trade logic
                pass
        a_time.sleep(10)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true'); parser.add_argument('--run', action='store_true'); parser.add_argument('--papertrade', action='store_true')
    args = parser.parse_args()

    if args.train:
        df_15m = database.get_data_from_db(config.SYMBOL, config.SIGNAL_TIMEFRAME)
        if df_15m.empty: logging.error("Data is empty.")
        else:
            df_featured = database.create_features(df_15m.copy())
            split_index = int(len(df_featured) * 0.75)
            train_df = df_featured.iloc[:split_index]
            train_agent(train_df)
    elif args.run:
        if not os.path.exists(config.MODEL_FILE): logging.error("Model file not found. Please train first.")
        else:
            df_15m = database.get_data_from_db(config.SYMBOL, config.SIGNAL_TIMEFRAME)
            df_featured = database.create_features(df_15m.copy())
            split_index = int(len(df_featured) * 0.75)
            test_df = df_featured.iloc[split_index:]
            run_backtest_process(test_df)
    elif args.papertrade:
        if not os.path.exists(config.MODEL_FILE): logging.error("Model file not found. Please train first.")
        else: run_paper_trader()
    else:
        print("Please specify mode: --train, --run, or --papertrade.")