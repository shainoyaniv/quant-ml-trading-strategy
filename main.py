import warnings
from urllib3.exceptions import NotOpenSSLWarning

warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
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

    df = apply_strategy(
        df,
        preds,
        long_rank=0.80,
        short_rank=0.05,
        use_trend_filter=True,
        verbose=False
    )

    results = backtest(df)

    strategy_returns = results["strategy_return"].rename(ticker)
    market_returns = results["return"].rename(ticker)

    return strategy_returns, market_returns

def main():
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

    all_strategy_returns = []
    all_market_returns = []

    for ticker in tickers:
        try:
            strategy_returns, market_returns = run_single_asset(ticker)
            all_strategy_returns.append(strategy_returns)
            all_market_returns.append(market_returns)
        except Exception as e:
            print(f"Failed to process {ticker}: {e}")

    strategy_df = pd.concat(all_strategy_returns, axis=1).dropna()
    market_df = pd.concat(all_market_returns, axis=1).dropna()

    # Equal-weight portfolios
    portfolio_returns = strategy_df.mean(axis=1)
    market_returns = market_df.mean(axis=1)

    cumulative_portfolio = (1 + portfolio_returns).cumprod()
    cumulative_market = (1 + market_returns).cumprod()

    sharpe = (
        portfolio_returns.mean() /
        portfolio_returns.std()
    ) * np.sqrt(252)

    print("\n=== FINAL RESULTS ===")
    print(f"Portfolio Return: {cumulative_portfolio.iloc[-1]:.2f}")
    print(f"Market Return: {cumulative_market.iloc[-1]:.2f}")
    print(f"Portfolio Sharpe: {sharpe:.2f}")
    print(f"Total Trades: {(strategy_df != 0).sum().sum()}")

    plt.figure()
    plt.plot(cumulative_portfolio, label="Strategy")
    plt.plot(cumulative_market, label="Market", linestyle="--")
    plt.legend()
    plt.title("Strategy vs Market")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Returns")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..")) if BASE_DIR.endswith("src") else BASE_DIR
    IMAGE_DIR = os.path.join(PROJECT_ROOT, "images")
    os.makedirs(IMAGE_DIR, exist_ok=True)

    save_path = os.path.join(IMAGE_DIR, "strategy_performance.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Chart saved to images/strategy_performance.png")

    drawdown = cumulative_portfolio / cumulative_portfolio.cummax() - 1

    plt.figure()
    plt.plot(drawdown)
    plt.title("Strategy Drawdown")
    plt.xlabel("Time")
    plt.ylabel("Drawdown")
    plt.show()

if __name__ == "__main__":
    print("Running trading strategy...")
    main()
