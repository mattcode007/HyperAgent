# environment.py
import pandas as pd, numpy as np, random, gymnasium as gym
from gymnasium import spaces
from sklearn.preprocessing import StandardScaler
import config

class TradingEnv(gym.Env):
    def __init__(self, df, scaler):
        super(TradingEnv, self).__init__()
        self.df = df
        self.features = ['open', 'high', 'low', 'close', 'volume', 'atr', 'rsi', 'price_vs_ema', 'volatility', 'rsi_lag_3']
        self.scaler = scaler
        self.df_scaled_features = self.scaler.transform(self.df[self.features])
        self.action_space = spaces.Discrete(3) # 0: Flat/Exit, 1: Long, 2: Short
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                            shape=(config.SEQUENCE_LENGTH, len(self.features)), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.current_step = config.SEQUENCE_LENGTH
        self.balance = config.INITIAL_ACCOUNT_BALANCE
        self.position, self.entry_price, self.entry_time, self.trade_type = 0, 0, None, None
        self.stop_loss, self.take_profit = 0, 0
        return self._get_observation(), {}

    def _get_observation(self):
        return self.df_scaled_features[self.current_step - config.SEQUENCE_LENGTH : self.current_step]

    def step(self, action):
        current_candle = self.df.iloc[self.current_step]
        current_price = current_candle['close']; current_time = current_candle.name
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        reward, trade_info = 0, None
        
        if self.position != 0:
            exit_price = None
            if self.position == 1:
                if current_candle['low'] <= self.stop_loss: exit_price = self.stop_loss
                elif current_candle['high'] >= self.take_profit: exit_price = self.take_profit
            elif self.position == -1:
                if current_candle['high'] >= self.stop_loss: exit_price = self.stop_loss
                elif current_candle['low'] <= self.take_profit: exit_price = self.take_profit
            if action == 0: exit_price = current_price

            if exit_price:
                pnl = (exit_price - self.entry_price) * self.position
                self.balance += pnl * config.LEVERAGE
                trade_info = {"Type": self.trade_type, "Entry Time": self.entry_time, "Entry Price": self.entry_price, 
                              "Exit Price": exit_price, "PnL": pnl * config.LEVERAGE, "Exit Time": current_time, "Balance": self.balance}
                self.position, self.entry_price, self.trade_type, self.entry_time, self.stop_loss, self.take_profit = 0, 0, None, None, 0, 0
            else:
                prev_price = self.df['close'].iloc[self.current_step - 2]
                reward = (current_price - prev_price) * self.position

        elif action in [1, 2] and self.position == 0:
            self.position = 1 if action == 1 else -1
            self.trade_type = "Long" if action == 1 else "Short"
            self.entry_price, self.entry_time = current_price, current_time
            atr = current_candle['atr']
            if self.position == 1:
                self.stop_loss = current_price - (atr * 2.0)
                self.take_profit = current_price + (atr * config.RISK_REWARD_RATIO * 2.0)
            else:
                self.stop_loss = current_price + (atr * 2.0)
                self.take_profit = current_price - (atr * config.RISK_REWARD_RATIO * 2.0)
        
        if self.balance <= 0: done = True
        return self._get_observation(), reward, done, False, trade_info