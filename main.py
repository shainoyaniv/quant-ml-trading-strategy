import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from src.data_loader import (
    load_data,
    add_features,
    prepare_data,
    walk_forward,
    apply_strategy,
    optimize_thresholds,
    backtest
)


def run_single_asset(ticker):
    print(f"\nRunning strategy for {ticker}...")

    df = load_data(ticker, "2021-01-01", "2026-01-01")
    df = add_features(df)

    X, y, df = prepare_data(df)

    preds = walk_forward(X, y)

    best_long, best_short = optimize_thresholds(df, preds)

    df = apply_strategy(
        df,
        preds,
        long_rank=0.8,
        short_rank=0.005,
        use_trend_filter=True,
        verbose=True
    )

    results = backtest(df)

    print("\n=== DEBUG ===")
    print("Total trades:", (df["position"] != 0).sum())
    print("Long trades:", (df["position"] > 0).sum())
    print("Short trades:", (df["position"] < 0).sum())
    print("Strategy return:", df["cumulative_strategy"].iloc[-1])
    print("Market return:", df["cumulative_market"].iloc[-1])
    print(df['confidence'].describe())
    return results["strategy_return"].rename(ticker)


def main():
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

    all_returns = []

    for ticker in tickers:
        try:
            strategy_returns = run_single_asset(ticker)
            all_returns.append(strategy_returns)
        except Exception as e:
            print(f"Failed to process {ticker}: {e}")

    returns_df = pd.concat(all_returns, axis=1).dropna()

    # Equal-weight portfolio
    portfolio_returns = returns_df.mean(axis=1)

    cumulative_portfolio = (1 + portfolio_returns).cumprod()

    sharpe = (
                     portfolio_returns.mean() /
                     portfolio_returns.std()
             ) * np.sqrt(252)

    print("\n=== MULTI-ASSET PORTFOLIO RESULTS ===")
    print(f"Portfolio Return: {cumulative_portfolio.iloc[-1]:.2f}")
    print(f"Portfolio Sharpe: {sharpe:.2f}")

    plt.figure()
    plt.plot(cumulative_portfolio, label="Portfolio Strategy")
    plt.legend()
    plt.title("Multi-Asset ML Trading Strategy")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    print("Saving plot...")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    if BASE_DIR.endswith("src"):
        PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
    else:
        PROJECT_ROOT = BASE_DIR
    IMAGE_DIR = os.path.join(PROJECT_ROOT, "images")
    os.makedirs(IMAGE_DIR, exist_ok=True)
    save_path = os.path.join(IMAGE_DIR, "strategy_performance.png")
    print("Saving to:", save_path)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    plt.show()


if __name__ == "__main__":
    main()