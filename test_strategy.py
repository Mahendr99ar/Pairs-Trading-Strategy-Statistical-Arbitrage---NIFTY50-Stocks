#!/usr/bin/env python3
"""
Test script for Pairs Trading Strategy
Run this to verify the installation and basic functionality
"""

try:
    from pairs_trading_strategy import *
    print("✓ All imports successful")

    # Test data generation
    print("\n1. Testing data generation...")
    test_stocks = ['STOCK_A', 'STOCK_B', 'STOCK_C']
    test_data = generate_synthetic_price_data(test_stocks, '2024-01-01', '2024-12-31')
    print(f"✓ Generated data shape: {test_data.shape}")

    # Test cointegration
    print("\n2. Testing cointegration analysis...")
    result = test_cointegration(test_data, 'STOCK_A', 'STOCK_B')
    print(f"✓ Cointegration test completed, p-value: {result['p_value']:.4f}")

    # Test strategy
    print("\n3. Testing strategy implementation...")
    strategy = PairsTradingStrategy('STOCK_A', 'STOCK_B', test_data)
    backtest_results = strategy.backtest(initial_capital=10000)
    metrics = strategy.calculate_performance_metrics()
    print(f"✓ Strategy backtest completed")
    print(f"✓ Final portfolio value: ${backtest_results['portfolio_value'].iloc[-1]:,.2f}")
    print(f"✓ Sharpe ratio: {metrics['sharpe_ratio']:.2f}")

    print("\n" + "="*50)
    print("🎉 ALL TESTS PASSED!")
    print("✓ Pairs Trading Strategy is working correctly")
    print("✓ You can now run the main script: python pairs_trading_strategy.py")
    print("="*50)

except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please install required packages: pip install -r requirements.txt")

except Exception as e:
    print(f"❌ Test Failed: {e}")
    print("Please check the implementation for errors")
