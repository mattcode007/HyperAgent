# agent.py
import torch, torch.nn as nn, torch.optim as optim, numpy as np
from collections import deque
import random, json, logging, argparse, time as a_time, os, sys
import pytz, joblib
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import config, database, utils
from model import TransformerDQN
from environment import TradingEnv

os.makedirs(config.LOG_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(config.AGENT_LOG_FILE), logging.StreamHandler()])

class ReplayBuffer:
    def __init__(self, capacity): self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done): self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size): return random.sample(self.buffer, batch_size)
    def __len__(self): return len(self.buffer)

def train_agent(train_df):
    logging.info(f"Using device: {config.DEVICE}")
    env = TradingEnv(pd.DataFrame(), StandardScaler())
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Apex Predator RL Trading Agent")
    parser.add_argument('--train', action='store_true', help='Train a new agent model.')
    parser.add_argument('--run', action='store_true', help='Run a backtest on the trained agent.')
    args = parser.parse_args()

    success_fetch = database.fetch_and_store_data(config.SYMBOL, config.SIGNAL_TIMEFRAME)
    if not success_fetch:
        logging.error("Failed to fetch data. Aborting.")
        sys.exit(1)

    df_15m = database.get_data_from_db(config.SYMBOL, config.SIGNAL_TIMEFRAME)
    
    if df_15m.empty:
        logging.error("Data is empty after fetch. Cannot proceed.")
        sys.exit(1)
    
    df_15m_featured = database.create_features(df_15m.copy())
    split_index = int(len(df_15m_featured) * 0.75)
    train_df, test_df = df_15m_featured.iloc[:split_index], df_15m_featured.iloc[split_index:]
    logging.info(f"Data split: {len(train_df)} training samples, {len(test_df)} testing samples.")

    if args.train:
        train_agent(train_df)
    elif args.run:
        if not os.path.exists(config.MODEL_FILE):
            logging.error("Model file not found. Please train first.")
        else:
            run_backtest_process(test_df)
    else:
        print("Please specify mode: --train or --run.")