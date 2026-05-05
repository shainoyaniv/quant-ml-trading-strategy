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
            n_estimators=30,
            max_depth=2,
            n_jobs=-1,
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
def apply_strategy(
    df,
    preds,
    long_rank=0.70,
    short_rank=0.10,
    use_trend_filter=True,
    verbose=False
):
    """
    Apply a trading strategy based on model prediction confidence.

    This function converts model prediction probabilities into trading positions
    using a ranking-based approach. Long positions are taken on high-confidence
    signals, while short positions are restricted to extreme low-confidence cases.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing market data and features.
        Must include at least:
        - 'Close': asset price
        - 'ma_200': long-term moving average
        - 'return': daily return

    preds : array-like
        Model prediction probabilities (e.g., probability of price increase).

    long_rank : float, default=0.70
        Percentile threshold for long positions.
        Example: 0.70 → top 30% highest confidence predictions.

    short_rank : float, default=0.10
        Percentile threshold for short positions.
        Example: 0.10 → bottom 10% lowest confidence predictions.

    use_trend_filter : bool, default=True
        Whether to apply a trend filter based on price vs. MA200.

    verbose : bool, default=False
        If True, prints debug information about signals and thresholds.

    Returns
    -------
    pandas.DataFrame
        DataFrame with added columns:
        - 'confidence': model predictions
        - 'confidence_rank': percentile ranking of predictions
        - 'position': trading position (-1, 0, 1)
        - 'strategy_return': strategy daily returns
    """

    import numpy as np

    # Align dataframe length with predictions
    df = df.iloc[-len(preds):].copy()
    df["confidence"] = preds

    # Convert confidence to percentile rank (0 → low, 1 → high)
    df["confidence_rank"] = df["confidence"].rank(pct=True)

    # Generate raw signals
    raw_long = df["confidence_rank"] > long_rank
    raw_short = df["confidence_rank"] < short_rank

    # Optional trend filter (soft)
    if use_trend_filter:
        trend_margin = 0.005  # 0.5% tolerance

        close = df["Close"].squeeze()
        ma_200 = df["ma_200"].squeeze()

        bullish = close > ma_200 * (1 - trend_margin)
        bearish = close < ma_200 * (1 + trend_margin)

        long_mask = raw_long & bullish
        short_mask = raw_short & bearish
    else:
        long_mask = raw_long
        short_mask = raw_short

    # Debug output
    if verbose:
        print("Long rank threshold:", long_rank)
        print("Short rank threshold:", short_rank)
        print("Raw long signals:", raw_long.sum())
        print("Raw short signals:", raw_short.sum())
        print("Final long signals:", long_mask.sum())
        print("Final short signals:", short_mask.sum())

    # Initialize positions
    df["position"] = 0.0

    # Apply positions
    df.loc[long_mask, "position"] = 1
    df.loc[short_mask, "position"] = -1

    # Avoid lookahead bias (use next day's return)
    df["position"] = df["position"].shift(1)

    # Transaction cost modeling
    cost = 0.001
    df["trade"] = df["position"].diff().abs()

    df["strategy_return"] = (
        df["position"] * df["return"] - cost * df["trade"]
    )

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

            if temp["strategy_return"].std() == 0:
                continue

            sharpe = (
                temp["strategy_return"].mean() /
                temp["strategy_return"].std()
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