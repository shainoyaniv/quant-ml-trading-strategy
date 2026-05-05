import numpy as np
import pandas as pd
import yfinance as yf
from xgboost import XGBClassifier


# =========================
# DATA LOADING
# =========================
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df['return'] = df['Close'].pct_change()
    return df.dropna()


# =========================
# FEATURE ENGINEERING
# =========================
def add_features(df):
    df = df.copy()

    df['ma_10'] = df['Close'].rolling(10).mean()
    df['ma_50'] = df['Close'].rolling(50).mean()
    df['ma_200'] = df['Close'].rolling(200).mean()

    df['momentum'] = df['Close'] / df['Close'].shift(10) - 1
    df['volatility'] = df['return'].rolling(10).std()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # TREND FILTER 🔥
    df['trend'] = (df['ma_50'] > df['ma_200']).astype(int)

    return df.dropna()


# =========================
# DATASET
# =========================
def prepare_data(df):
    features = ['return', 'ma_10', 'ma_50', 'momentum', 'volatility', 'rsi']
    X = df[features]
    y = (df['return'].shift(-1) > 0).astype(int)

    return X.iloc[:-1], y.iloc[:-1], df.iloc[:-1]


# =========================
# WALK FORWARD TRAINING
# =========================
def walk_forward(X, y):
    preds = []

    window = 252  # 1 year

    for i in range(window, len(X)):
        X_train = X.iloc[i-window:i]
        y_train = y.iloc[i-window:i]

        model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss'
        )

        model.fit(X_train, y_train)

        pred = model.predict_proba(X.iloc[i:i+1])[:, 1][0]
        preds.append(pred)

    return np.array(preds)


# =========================
# STRATEGY (THE MAGIC 🔥)
# =========================
def apply_strategy(df, preds, long_th=0.6, short_th=0.4):
    df = df.iloc[-len(preds):].copy()
    df['confidence'] = preds

    df['position'] = 0.0

    # LONG only in bull market
    long_mask = (df['confidence'] > long_th) & (df['trend'] == 1)
    df.loc[long_mask, 'position'] = (
        (df.loc[long_mask, 'confidence'] - long_th) / (1 - long_th)
    )

    # SHORT only in bear market
    short_mask = (df['confidence'] < short_th) & (df['trend'] == 0)
    df.loc[short_mask, 'position'] = -(
        (short_th - df.loc[short_mask, 'confidence']) / short_th
    )

    # Avoid lookahead bias
    df['position'] = df['position'].shift(1)

    # Transaction cost
    cost = 0.001
    df['trade'] = df['position'].diff().abs()
    df['strategy_return'] = df['position'] * df['return'] - cost * df['trade']

    return df.dropna()

def optimize_thresholds(df, preds):
    """
    Find the best long/short confidence thresholds based on Sharpe Ratio.
    """

    best_sharpe = -999
    best_long = None
    best_short = None

    long_range = np.arange(0.55, 0.81, 0.05)
    short_range = np.arange(0.20, 0.46, 0.05)

    for long_th in long_range:
        for short_th in short_range:

            temp = apply_strategy(df, preds, long_th, short_th)

            if temp['strategy_return'].std() == 0:
                continue

            sharpe = (
                temp['strategy_return'].mean() /
                temp['strategy_return'].std()
            ) * np.sqrt(252)

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_long = long_th
                best_short = short_th

    print("\n=== THRESHOLD OPTIMIZATION ===")
    print(f"Best Long Threshold: {best_long:.2f}")
    print(f"Best Short Threshold: {best_short:.2f}")
    print(f"Best Sharpe: {best_sharpe:.2f}")

    return best_long, best_short

# =========================
# BACKTEST
# =========================
def backtest(df):
    df['cumulative_market'] = (1 + df['return']).cumprod()
    df['cumulative_strategy'] = (1 + df['strategy_return']).cumprod()

    sharpe = (
        df['strategy_return'].mean() /
        df['strategy_return'].std()
    ) * np.sqrt(252)

    print("\n=== RESULTS ===")
    print(f"Market: {df['cumulative_market'].iloc[-1]:.2f}")
    print(f"Strategy: {df['cumulative_strategy'].iloc[-1]:.2f}")
    print(f"Sharpe: {sharpe:.2f}")

    return df