# database.py
import ccxt, pandas as pd, sqlite3, pytz
from datetime import datetime, timedelta, time
import config

def setup_database():
    conn = sqlite3.connect(config.DB_FILE)
    cursor = conn.cursor()
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS ohlcv (
            timestamp TEXT, symbol TEXT, timeframe TEXT,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            PRIMARY KEY (timestamp, symbol, timeframe))''')
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT, entry_time TEXT, exit_time TEXT,
            symbol TEXT, type TEXT, entry_price REAL, exit_price REAL,
            pnl REAL, balance REAL)''')
    conn.commit()
    return conn

def fetch_and_store_data(symbol, timeframe):
    exchange = getattr(ccxt, config.EXCHANGE_ID)({'options': {'defaultType': 'swap'}})
    conn = setup_database()
    table_name = 'ohlcv'
    last_ts_str = pd.read_sql(f"SELECT MAX(timestamp) FROM {table_name} WHERE symbol='{symbol}' AND timeframe='{timeframe}'", conn).iloc[0, 0]
    since = None
    if last_ts_str: since = int(pd.to_datetime(last_ts_str, utc=True).timestamp() * 1000)
    else: since = exchange.parse8601((datetime.now() - timedelta(days=config.FETCH_DAYS_INITIAL)).isoformat())
    if since: since += 1 

    all_ohlcv = []
    print(f"Syncing {timeframe} data for {symbol}...")
    while True:
        try:
            params = {'type': 'linear'}
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000, params=params)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            if len(ohlcv) < 1000: break
        except Exception as e: print(f"Error fetching data: {e}"); break
    
    if all_ohlcv:
        new_df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms', utc=True).astype(str)
        new_df['symbol'], new_df['timeframe'] = symbol, timeframe
        existing_ts = pd.read_sql(f"SELECT timestamp FROM {table_name} WHERE symbol='{symbol}' AND timeframe='{timeframe}'", conn)
        new_df = new_df[~new_df['timestamp'].isin(existing_ts['timestamp'])]
        if not new_df.empty:
            new_df.to_sql(table_name, conn, if_exists='append', index=False)
            print(f"Stored {len(new_df)} new {timeframe} candles.")
    conn.close()

def get_data_from_db(symbol, timeframe):
    conn = setup_database()
    df = pd.read_sql(f"SELECT * FROM ohlcv WHERE symbol='{symbol}' AND timeframe='{timeframe}' ORDER BY timestamp", conn, 
                      index_col='timestamp', parse_dates=['timestamp'])
    if df.index.tz is None: df.index = df.index.tz_localize('UTC')
    conn.close()
    return df.drop(columns=['symbol', 'timeframe'], errors='ignore')

def create_features(df):
    df['atr'] = (df['high'] - df['low']).rolling(window=config.ATR_PERIOD).mean()
    df['rsi'] = 100 - (100 / (1 + (df['close'].diff().clip(lower=0).rolling(14).mean() / df['close'].diff().clip(upper=0).abs().rolling(14).mean())))
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['price_vs_ema'] = (df['close'] - df['ema_200']) / df['ema_200']
    df['volatility'] = df['close'].pct_change().rolling(window=96).std()
    df['rsi_lag_3'] = df['rsi'].shift(3)
    df.dropna(inplace=True)
    return df

def log_trade_to_db(trade_info):
    conn = setup_database()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO trades (entry_time, exit_time, symbol, type, entry_price, exit_price, pnl, balance)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        str(trade_info['Entry Time']), str(trade_info['Exit Time']), config.SYMBOL, trade_info['Type'],
        trade_info['Entry Price'], trade_info['Exit Price'], trade_info['PnL'], trade_info['Balance']
    ))
    conn.commit()
    conn.close()