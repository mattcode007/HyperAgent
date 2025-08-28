#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e 

echo "--- Starting Apex Agent System ---"

# --- Step 1: Launch the Dashboard in the Background ---
echo "[PHASE 1/2] Launching monitoring dashboard in the background..."
python3 -m streamlit run dashboard.py &

sleep 5

# --- Step 2: Train the Agent ---
echo "[PHASE 2/2] Starting agent training... You can monitor the progress on the dashboard."
python3 agent.py --train

echo ""
echo "--- ✅ Training complete. ---"
echo "You can now run a backtest on the unseen data with the following command:"
echo "python3 agent.py --run"