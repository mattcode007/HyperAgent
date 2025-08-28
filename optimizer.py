# optimizer.py
import optuna
import logging
import pandas as pd
import config, database, agent

# Mute the agent's verbose logging for the optimization study
logging.getLogger().setLevel(logging.WARNING)

def objective(trial):
    """Function for Optuna to optimize."""
    # Suggest hyperparameters to test
    trial_config = config
    trial_config.LEARNING_RATE = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    trial_config.GAMMA = trial.suggest_float("gamma", 0.95, 0.999)
    trial_config.EPSILON_DECAY = trial.suggest_float("epsilon_decay", 0.990, 0.999)
    trial_config.D_MODEL = trial.suggest_categorical("d_model", [32, 64, 128])
    trial_config.N_HEAD = trial.suggest_categorical("n_head", [2, 4])
    trial_config.N_LAYERS = trial.suggest_int("n_layers", 1, 3)

    # Run a shortened training session with these parameters
    # The train_agent function needs to be able to accept config and run in a headless mode
    final_avg_reward = agent.train_agent(train_df, num_episodes=20, is_optuna_trial=True)
    
    return final_avg_reward

if __name__ == '__main__':
    print("--- Starting Hyperparameter Optimization with Optuna ---")
    database.fetch_and_store_data(config.SYMBOL, config.SIGNAL_TIMEFRAME)
    df_15m = database.get_data_from_db(config.SYMBOL, config.SIGNAL_TIMEFRAME)
    df_15m_featured = database.create_features(df_15m.copy())
    
    # Use only the training portion for optimization
    split_index = int(len(df_15m_featured) * 0.75)
    train_df = df_15m_featured.iloc[:split_index]
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25) # Run 25 different trials
    
    print("\n--- Optimization Complete ---")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Avg Reward): {trial.value}")
    print("  Best Parameters: ")
    for key, value in trial.params.items():
        print(f"    '{key.upper()}': {value},")

    print("\nUpdate these values in your config.py file for the final training.")