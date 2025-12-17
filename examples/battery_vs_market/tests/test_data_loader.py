"""
Quick test script to verify data loading works correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import load_imbalance_prices, get_data_path

# Test loading real Swissgrid data
data_path = get_data_path()
prices_file = data_path / 'imbalance_prices.csv'

if prices_file.exists():
    print("Loading Swissgrid imbalance prices...")
    p_buy, p_sell = load_imbalance_prices(str(prices_file))

    print(f"\nLoaded {len(p_buy)} timesteps")
    print(f"Time horizon: {len(p_buy) / 96:.1f} days (assuming 15-min resolution)")

    print("\nFirst 10 timesteps:")
    print("  t  | p_buy (CHF/kWh) | p_sell (CHF/kWh) | Spread")
    print("-----|-----------------|------------------|--------")
    for t in range(min(10, len(p_buy))):
        spread = p_buy[t] - p_sell[t]
        print(f"  {t:2d} | {p_buy[t]:15.4f} | {p_sell[t]:16.4f} | {spread:6.4f}")

    print("\nPrice statistics:")
    print(f"  Buy  - Min: {min(p_buy.values()):.4f}, Max: {max(p_buy.values()):.4f}, "
          f"Mean: {sum(p_buy.values())/len(p_buy):.4f} CHF/kWh")
    print(f"  Sell - Min: {min(p_sell.values()):.4f}, Max: {max(p_sell.values()):.4f}, "
          f"Mean: {sum(p_sell.values())/len(p_sell):.4f} CHF/kWh")

    # Check for negative prices
    neg_buy = sum(1 for v in p_buy.values() if v < 0)
    neg_sell = sum(1 for v in p_sell.values() if v < 0)
    print(f"\nNegative prices: Buy={neg_buy}, Sell={neg_sell}")

else:
    print(f"File not found: {prices_file}")
    print("Please add your Swissgrid data to examples/battery_vs_market/data/imbalance_prices.csv")