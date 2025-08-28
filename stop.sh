#!/bin/bash

# This script stops the background agent process.

echo "--- Stopping Apex Agent System ---"
PID_FILE="agent.pid"

if [ -f "$PID_FILE" ]; then
    AGENT_PID=$(cat $PID_FILE)
    echo "Found agent process with PID: $AGENT_PID"
    kill $AGENT_PID
    rm $PID_FILE
    echo "Agent process stopped."
else
    echo "No running agent found (PID file does not exist)."
fi