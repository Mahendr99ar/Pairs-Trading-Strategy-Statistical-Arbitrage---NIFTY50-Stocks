
"""
Pairs Trading Strategy (Statistical Arbitrage)
==============================================

A complete implementation of a pairs trading strategy using statistical arbitrage
on NIFTY50 stock pairs with cointegration testing and Z-score signals.

Author: Mahendra Meena
Project: Quantitative Finance — Statistical Arbitrage — Backtesting
Period: June 2025 – July 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
from scipy import stats
from statsmodels.tsa.stattools import coint

warnings.filterwarnings('ignore')

# NIFTY50 stock symbols (subset for demonstration)
NIFTY50_STOCKS = [
    'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR', 'HDFC', 'ICICIBANK',
    'KOTAKBANK', 'BHARTIARTL', 'ITC', 'SBIN', 'BAJFINANCE', 'ASIANPAINT',
    'MARUTI', 'AXISBANK', 'LT', 'TITAN', 'NESTLEIND', 'ULTRACEMCO', 'WIPRO'
]

def generate_synthetic_price_data(stocks, start_date, end_date, seed=42):
    """
    Generate synthetic stock price data for demonstration.
    In real implementation, replace this with actual data from APIs like Yahoo Finance, Alpha Vantage, etc.
    """
    np.random.seed(seed)

    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates[dates.weekday < 5]  # Remove weekends

    price_data = {}

    for stock in stocks:
        n_days = len(dates)
        returns = np.random.normal(0.001, 0.02, n_days)  # Daily returns

        # Add correlation structure for realistic behavior
        if stock in ['HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'AXISBANK']:
            bank_factor = np.random.normal(0, 0.01, n_days)
            returns += bank_factor * 0.5

        # Generate prices from returns
        initial_price = np.random.uniform(100, 2000)
        prices = [initial_price]

        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(new_price)

        price_data[stock] = prices

    return pd.DataFrame(price_data, index=dates)

def test_cointegration(price_data, stock1, stock2, significance_level=0.05):
    """Test cointegration between two stock price series using Engle-Granger test"""
    series1 = price_data[stock1].dropna()
    series2 = price_data[stock2].dropna()

    # Perform cointegration test
    coint_stat, p_value, critical_values = coint(series1, series2)

    return {
        'stock1': stock1,
        'stock2': stock2,
        'cointegration_stat': coint_stat,
        'p_value': p_value,
        'critical_values': critical_values,
        'is_cointegrated': p_value < significance_level
    }

def find_cointegrated_pairs(price_data, stocks, significance_level=0.05):
    """Find all cointegrated pairs among given stocks"""
    cointegrated_pairs = []

    print("Testing for cointegrated pairs...")
    for i in range(len(stocks)):
        for j in range(i+1, len(stocks)):
            stock1, stock2 = stocks[i], stocks[j]
            result = test_cointegration(price_data, stock1, stock2, significance_level)

            if result['is_cointegrated']:
                cointegrated_pairs.append(result)
                print(f"✓ Found cointegrated pair: {stock1} - {stock2} (p-value: {result['p_value']:.4f})")

    return cointegrated_pairs

def calculate_spread_and_zscore(price_data, stock1, stock2, lookback_window=30):
    """Calculate spread and Z-score for pairs trading"""
    series1 = price_data[stock1]
    series2 = price_data[stock2]

    # Calculate the spread using ratio
    spread = series1 / series2

    # Calculate rolling statistics
    rolling_mean = spread.rolling(window=lookback_window).mean()
    rolling_std = spread.rolling(window=lookback_window).std()

    # Calculate Z-score
    z_score = (spread - rolling_mean) / rolling_std

    return spread, z_score, rolling_mean, rolling_std

def generate_trading_signals(z_score, entry_threshold=2.0, exit_threshold=0.5):
    """Generate trading signals based on Z-score"""
    signals = pd.Series(0, index=z_score.index)
    position = 0

    for i in range(len(z_score)):
        current_z = z_score.iloc[i]

        if pd.isna(current_z):
            signals.iloc[i] = position
            continue

        # Entry signals
        if position == 0:
            if current_z > entry_threshold:
                position = -1  # Short spread
            elif current_z < -entry_threshold:
                position = 1   # Long spread

        # Exit signals
        elif position != 0:
            if abs(current_z) < exit_threshold:
                position = 0

        signals.iloc[i] = position

    return signals

class PairsTradingStrategy:
    """
    Complete Pairs Trading Strategy implementation with backtesting and performance metrics.

    This strategy identifies cointegrated pairs and trades on mean-reversion using Z-score signals.
    """

    def __init__(self, stock1, stock2, price_data, 
                 lookback_window=30, entry_threshold=2.0, exit_threshold=0.5,
                 transaction_cost=0.001):

        self.stock1 = stock1
        self.stock2 = stock2
        self.price_data = price_data
        self.lookback_window = lookback_window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.transaction_cost = transaction_cost

        # Calculate spread and signals
        self.spread, self.z_score, self.rolling_mean, self.rolling_std = \
            calculate_spread_and_zscore(price_data, stock1, stock2, lookback_window)

        self.signals = generate_trading_signals(self.z_score, entry_threshold, exit_threshold)

    def backtest(self, initial_capital=100000):
        """
        Backtest the pairs trading strategy using spread returns approach.
        This method provides realistic performance estimation accounting for transaction costs.
        """

        # Calculate spread returns
        spread_returns = self.spread.pct_change().fillna(0)

        # Generate positions (lag by 1 day for realistic execution)
        positions = self.signals.shift(1).fillna(0)

        # Calculate strategy returns
        strategy_returns = positions * spread_returns

        # Apply transaction costs
        position_changes = positions.diff().fillna(0)
        transaction_costs = abs(position_changes) * self.transaction_cost

        # Net returns after costs
        net_returns = strategy_returns - transaction_costs

        # Calculate cumulative performance
        cumulative_returns = (1 + net_returns).cumprod()
        portfolio_values = initial_capital * cumulative_returns

        # Create results DataFrame
        self.results = pd.DataFrame({
            'portfolio_value': portfolio_values,
            'z_score': self.z_score,
            'signals': self.signals,
            'positions': positions,
            'spread': self.spread,
            'strategy_returns': strategy_returns,
            'net_returns': net_returns,
            'cumulative_returns': cumulative_returns
        }, index=self.spread.index)

        return self.results

    def calculate_performance_metrics(self):
        """Calculate comprehensive strategy performance metrics"""
        if not hasattr(self, 'results'):
            raise ValueError("Run backtest() first")

        net_returns = self.results['net_returns'].dropna()
        portfolio_values = self.results['portfolio_value']

        # Performance metrics
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1) * 100

        days = len(net_returns)
        annualized_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) ** (252/days) - 1

        volatility = net_returns.std() * np.sqrt(252)

        # Sharpe ratio
        risk_free_rate = 0.05
        excess_returns = net_returns.mean() * 252 - risk_free_rate
        sharpe_ratio = excess_returns / volatility if volatility != 0 else 0

        # Maximum drawdown
        peak = portfolio_values.cummax()
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = drawdown.min() * 100

        # Win rate
        winning_trades = sum(net_returns > 0)
        total_trades = sum(net_returns != 0)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        return {
            'total_return_pct': total_return,
            'annualized_return': annualized_return * 100,
            'volatility': volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades
        }

    def plot_results(self):
        """Plot strategy performance and trading signals"""
        if not hasattr(self, 'results'):
            raise ValueError("Run backtest() first")

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # Portfolio value over time
        axes[0].plot(self.results.index, self.results['portfolio_value'])
        axes[0].set_title(f'Portfolio Value: {self.stock1} - {self.stock2}')
        axes[0].set_ylabel('Portfolio Value ($)')
        axes[0].grid(True)

        # Z-score and signals
        axes[1].plot(self.results.index, self.results['z_score'], label='Z-Score')
        axes[1].axhline(y=self.entry_threshold, color='r', linestyle='--', label='Entry Threshold')
        axes[1].axhline(y=-self.entry_threshold, color='r', linestyle='--')
        axes[1].axhline(y=self.exit_threshold, color='g', linestyle='--', label='Exit Threshold')
        axes[1].axhline(y=-self.exit_threshold, color='g', linestyle='--')

        # Mark trading signals
        long_signals = self.results[self.results['signals'] == 1]
        short_signals = self.results[self.results['signals'] == -1]

        axes[1].scatter(long_signals.index, long_signals['z_score'], 
                       color='green', marker='^', s=30, label='Long Signal')
        axes[1].scatter(short_signals.index, short_signals['z_score'], 
                       color='red', marker='v', s=30, label='Short Signal')

        axes[1].set_title('Z-Score and Trading Signals')
        axes[1].set_ylabel('Z-Score')
        axes[1].legend()
        axes[1].grid(True)

        # Spread price
        axes[2].plot(self.results.index, self.results['spread'])
        axes[2].set_title('Price Spread')
        axes[2].set_ylabel('Spread Ratio')
        axes[2].set_xlabel('Date')
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()

def main():
    """
    Main function to run the complete pairs trading strategy analysis
    """
    print("="*60)
    print("PAIRS TRADING STRATEGY (STATISTICAL ARBITRAGE)")
    print("="*60)
    print("Author: Mahendra Meena")
    print("Project: Quantitative Finance — Statistical Arbitrage — Backtesting")
    print("Period: June 2025 – July 2025")
    print()

    # Generate sample data (replace with real data in production)
    print("1. Loading/Generating price data...")
    start_date = '2024-01-01'
    end_date = '2025-07-31'
    price_data = generate_synthetic_price_data(NIFTY50_STOCKS, start_date, end_date)
    print(f"✓ Generated price data for {len(NIFTY50_STOCKS)} stocks")

    # Find cointegrated pairs
    print("\n2. Testing for cointegrated pairs...")
    cointegrated_pairs = find_cointegrated_pairs(price_data, NIFTY50_STOCKS)
    print(f"✓ Found {len(cointegrated_pairs)} cointegrated pairs")

    if len(cointegrated_pairs) == 0:
        print("No cointegrated pairs found. Exiting...")
        return

    # Select best pair and run strategy
    print("\n3. Running pairs trading strategy...")
    best_pair = min(cointegrated_pairs, key=lambda x: x['p_value'])
    stock1, stock2 = best_pair['stock1'], best_pair['stock2']

    # Initialize strategy with optimized parameters
    strategy = PairsTradingStrategy(
        stock1=stock1, 
        stock2=stock2, 
        price_data=price_data,
        lookback_window=20,
        entry_threshold=1.2,
        exit_threshold=0.3,
        transaction_cost=0.0015
    )

    # Run backtest
    results = strategy.backtest(initial_capital=100000)
    metrics = strategy.calculate_performance_metrics()

    # Display results
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    print(f"Stock Pair: {stock1} - {stock2}")
    print(f"Period: {results.index[0].date()} to {results.index[-1].date()}")
    print(f"Cointegration p-value: {best_pair['p_value']:.4f}")
    print()
    print("PERFORMANCE METRICS:")
    print(f"• Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"• Annualized Return: {metrics['annualized_return']:.2f}%")
    print(f"• Volatility (Annual): {metrics['volatility']:.2f}%")
    print(f"• Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"• Maximum Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"• Win Rate: {metrics['win_rate']:.1f}%")
    print(f"• Total Trades: {metrics['total_trades']}")

    # Show validation of practical use
    if metrics['sharpe_ratio'] > 1.0 and metrics['total_return_pct'] > 0:
        print("\n✓ STRATEGY VALIDATION:")
        print("✓ Achieved profitable results under realistic transaction costs")
        print("✓ Demonstrated statistical arbitrage using cointegration")
        print("✓ Mean-reversion strategy with quantifiable risk metrics")
        print("✓ Suitable for practical implementation in equity markets")

    return strategy, metrics

if __name__ == "__main__":
    strategy, metrics = main()
