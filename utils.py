# utils.py
import os, subprocess, config, logging

def get_github_token():
    try:
        env_path = os.path.join(config.SCRIPT_DIR, ".env")
        with open(env_path, "r") as f:
            for line in f:
                if "GITHUB_PAT" in line:
                    return line.strip().split("=")[1].replace('"', '')
    except FileNotFoundError: return None

def push_results_to_github(commit_message):
    token = get_github_token()
    if not token:
        logging.error("GITHUB_PAT not found in .env file. Cannot push results.")
        return

    remote_url = f"https://{config.GITHUB_USERNAME}:{token}@github.com/{config.GITHUB_USERNAME}/{config.REPO_NAME}.git"
    logging.info("Pushing results to GitHub...")
    try:
        os.chdir(config.SCRIPT_DIR)
        subprocess.run(["git", "add", config.DB_FILE], check=True)
        subprocess.run(["git", "add", config.TRAINING_HISTORY_FILE], check=True)
        status_result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        if not status_result.stdout.strip():
            logging.info("No new results to commit."); return
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        subprocess.run(["git", "push", remote_url], check=True)
        logging.info("Successfully pushed results to GitHub.")
    except Exception as e:
        logging.error(f"An error occurred during git push: {e}")