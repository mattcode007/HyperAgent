# dashboard.py
import streamlit as st, pandas as pd, plotly.graph_objects as go, json, os, config, database, time, numpy as np, sys
from datetime import datetime # <-- FIX: Added this import

# Add the project directory to the Python path to ensure local modules are found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def display_dashboard():
    st.set_page_config(layout="wide", page_title="Apex Agent Dashboard")
    st.title("🤖 Apex Predator RL Agent - Monitoring Dashboard")
    st.text(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    tab1, tab2, tab3 = st.tabs(["Backtest Performance", "Full Trade History", "Model Training"])

    def load_trade_data():
        conn = database.setup_database()
        try:
            trades_df = pd.read_sql("SELECT * FROM trades ORDER BY entry_time", conn)
        except pd.io.sql.DatabaseError: 
            trades_df = pd.DataFrame()
        conn.close()
        return trades_df

    with tab1:
        st.header("Backtest Performance on Unseen Data")
        
        trades_df = load_trade_data()
        
        if trades_df.empty:
            st.warning("No backtest results found. Please run the agent with '--run' first.")
        else:
            equity_curve = pd.Series([config.INITIAL_ACCOUNT_BALANCE] + trades_df['balance'].tolist())
            total_return = (equity_curve.iloc[-1] / config.INITIAL_ACCOUNT_BALANCE - 1) * 100
            win_rate = (trades_df['pnl'] > 0).sum() / len(trades_df) * 100 if not trades_df.empty else 0
            
            kpi_cols = st.columns(4)
            kpi_cols[0].metric("Total Return", f"{total_return:.2f}%")
            kpi_cols[1].metric("Final Balance", f"${equity_curve.iloc[-1]:,.2f}")
            kpi_cols[2].metric("Total Trades", len(trades_df))
            kpi_cols[3].metric("Win Rate", f"{win_rate:.2f}%")

            fig_equity = go.Figure(go.Scatter(x=np.arange(len(equity_curve)), y=equity_curve, mode='lines', name='Equity'))
            fig_equity.update_layout(title="Backtest Equity Curve", template="plotly_dark")
            st.plotly_chart(fig_equity, use_container_width=True)

    with tab2:
        st.header("Full Trade History")
        trades_df_full = load_trade_data()
        if not trades_df_full.empty:
            st.dataframe(trades_df_full.sort_values(by="id", ascending=False), use_container_width=True)
        else: 
            st.info("No trades to display.")

    with tab3:
        st.header("Model Training Performance")
        if os.path.exists(config.TRAINING_HISTORY_FILE):
            with open(config.TRAINING_HISTORY_FILE, 'r') as f: 
                history = json.load(f)
            history_df = pd.DataFrame(history)
            
            fig_train = go.Figure(go.Scatter(x=history_df['episode'], y=history_df['reward'], mode='lines+markers'))
            fig_train.update_layout(title="Agent's Learning Curve (Total Reward per Episode)", template="plotly_dark")
            st.plotly_chart(fig_train, use_container_width=True)
            st.info("Upward trend indicates successful learning.")
        else:
            st.warning("No training history found. Train an agent first.")

    st.sidebar.button("Refresh Data")
    
if __name__ == '__main__':
    display_dashboard()