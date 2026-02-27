"""
Quick verification script for baseline calculator sign convention fix.

Tests that:
1. Positive imbalance (surplus) -> export revenue (negative cost)
2. Negative imbalance (deficit) -> import cost (positive cost)
"""

from examples.battery_vs_market.utils.baseline_calculator import calculate_baseline_cost

# Test case 1: Pure surplus (should export and earn revenue)
print("=" * 60)
print("Test 1: Pure Surplus (positive imbalance)")
print("=" * 60)
imbalance_surplus = {0: 100.0, 1: 50.0, 2: 75.0}  # All positive = surplus
p_buy = {0: 0.12, 1: 0.12, 2: 0.12}  # CHF/kWh
p_sell = {0: 0.08, 1: 0.08, 2: 0.08}  # CHF/kWh

baseline_cost, annual_cost = calculate_baseline_cost(imbalance_surplus, p_buy, p_sell)
print(f"Imbalance: {imbalance_surplus}")
print(f"Baseline cost: {baseline_cost:.4f} CHF")
print(f"Expected: NEGATIVE (earning revenue from exports)")
print(f"PASS" if baseline_cost < 0 else "FAIL")

# Test case 2: Pure deficit (should import and pay cost)
print("\n" + "=" * 60)
print("Test 2: Pure Deficit (negative imbalance)")
print("=" * 60)
imbalance_deficit = {0: -100.0, 1: -50.0, 2: -75.0}  # All negative = deficit
baseline_cost, annual_cost = calculate_baseline_cost(imbalance_deficit, p_buy, p_sell)
print(f"Imbalance: {imbalance_deficit}")
print(f"Baseline cost: {baseline_cost:.4f} CHF")
print(f"Expected: POSITIVE (paying for imports)")
print(f"PASS" if baseline_cost > 0 else "FAIL")

# Test case 3: Mixed (should net out)
print("\n" + "=" * 60)
print("Test 3: Mixed (both surplus and deficit)")
print("=" * 60)
imbalance_mixed = {0: 100.0, 1: -100.0, 2: 50.0}  # Mixed
# With 15-min timesteps (0.25h):
# t=0: Export 100kW * 0.25h * 0.08 CHF/kWh = +2.0 CHF revenue (cost = -2.0)
# t=1: Import 100kW * 0.25h * 0.12 CHF/kWh = +3.0 CHF cost
# t=2: Export 50kW * 0.25h * 0.08 CHF/kWh = +1.0 CHF revenue (cost = -1.0)
# Net: -2.0 + 3.0 - 1.0 = 0.0 CHF
baseline_cost, annual_cost = calculate_baseline_cost(imbalance_mixed, p_buy, p_sell)
print(f"Imbalance: {imbalance_mixed}")
print(f"Baseline cost: {baseline_cost:.4f} CHF")
print(f"Expected: ~0.0 CHF (net of imports and exports)")
expected = -2.0 + 3.0 - 1.0  # Should be 0.0
print(f"Calculated: {baseline_cost:.4f} CHF, Expected: {expected:.4f} CHF")
print(f"PASS" if abs(baseline_cost - expected) < 0.01 else "FAIL")

print("\n" + "=" * 60)
print("Summary: All sign convention tests completed")
print("=" * 60)
