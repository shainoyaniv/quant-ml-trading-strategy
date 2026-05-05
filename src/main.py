import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from data_loader import (
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

    df = load_data(ticker, "2015-01-01", "2024-01-01")
    df = add_features(df)

    X, y, df = prepare_data(df)

    preds = walk_forward(X, y)

    best_long, best_short = optimize_thresholds(df, preds)

    df = apply_strategy(df, preds, best_long, best_short)

    results = backtest(df)

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
    plt.show()


if __name__ == "__main__":
    main()