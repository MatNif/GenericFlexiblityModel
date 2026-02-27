"""
Test with the full 33,120 timesteps (~345 days at 15-min resolution).

This matches the real data scenario mentioned in the plan.
"""

import numpy as np
from scipy import sparse

from flex_model.assets import BatteryUnit, BatteryCostModel, BatteryFlex
from flex_model.assets import BalancingMarketCost, BalancingMarketFlex
from flex_model.optimization import LPOptimizer


# Full dataset: 33,120 timesteps = ~345 days at 15-min resolution
n_timesteps = 33120

print("Large-Scale Sparse Matrix Test")
print("=" * 80)
print(f"Testing with {n_timesteps:,} timesteps (~345 days at 15-min resolution)")
print("=" * 80)

# Create battery
print("\nCreating battery asset...")
battery_unit = BatteryUnit(
    name="BESS_100kWh",
    capacity_kwh=100.0,
    power_kw=50.0,
    efficiency=0.95,
    soc_min=0.1,
    soc_max=0.9,
)

battery_cost = BatteryCostModel(
    name="battery_economics",
    c_inv=500.0,
    n_lifetime=10.0,
    p_int=0.05,
)

battery_flex = BatteryFlex(unit=battery_unit, cost_model=battery_cost)
battery_flex.reset(E_plus_init=50.0, E_minus_init=50.0)

# Create market
print("Creating market asset...")
market = BalancingMarketFlex(
    cost_model=BalancingMarketCost(
        name="market",
        p_E_buy=0.25,
        p_E_sell=0.15
    )
)

# Create simple imbalance profile
print("Creating imbalance profile...")
imbalance = {}
for t in range(n_timesteps):
    # Create a pattern that alternates between surplus and deficit
    imbalance[t] = 30.0 * np.sin(2 * np.pi * t / 96)  # ~24-hour cycle

# Build linear models
print("\nBuilding linear models...")
battery_lp = battery_flex.get_linear_model(n_timesteps, initial_soc=0.5)
market_lp = market.get_linear_model(n_timesteps)

# Analyze battery constraint matrix
print(f"\nBattery constraint matrix (A_eq):")
print(f"  Shape: {battery_lp.A_eq.shape}")
print(f"  Type: {type(battery_lp.A_eq)}")

if sparse.issparse(battery_lp.A_eq):
    nnz = battery_lp.A_eq.nnz
    total_elements = battery_lp.A_eq.shape[0] * battery_lp.A_eq.shape[1]
    sparsity = 100.0 * (1.0 - nnz / total_elements)

    # Memory estimates
    sparse_memory_mb = (nnz * 12) / (1024 * 1024)
    dense_memory_mb = (total_elements * 8) / (1024 * 1024)
    reduction = dense_memory_mb / sparse_memory_mb

    print(f"  Elements: {total_elements:,}")
    print(f"  Non-zeros: {nnz:,}")
    print(f"  Sparsity: {sparsity:.5f}% zeros")
    print(f"\n  Memory usage:")
    print(f"    Sparse: {sparse_memory_mb:.2f} MiB")
    print(f"    Dense: {dense_memory_mb:.2f} MiB ({dense_memory_mb / 1024:.2f} GiB)")
    print(f"    Reduction: {reduction:.1f}x")
else:
    print("  ERROR: A_eq is not sparse!")
    exit(1)

# Set up optimizer
print("\n" + "=" * 80)
print("Setting up optimizer...")
optimizer = LPOptimizer(n_timesteps=n_timesteps)
optimizer.add_asset(battery_lp)
optimizer.add_asset(market_lp)
optimizer.set_imbalance(imbalance)

# Solve
print("\nSolving optimization problem...")
print("(This may take a minute or two for large problems...)")
result = optimizer.solve()

if result['success']:
    print("\n" + "=" * 80)
    print("SUCCESS!")
    print("=" * 80)
    print(f"Optimal cost: {result['cost']:.2f} CHF")

    # Extract solution statistics
    solution = result['solution']
    battery_sol = solution['BESS_100kWh']

    # Calculate total charge/discharge
    total_charge = sum(battery_sol[f'BESS_100kWh_P_charge_{t}'] for t in range(min(100, n_timesteps)))
    total_discharge = sum(battery_sol[f'BESS_100kWh_P_discharge_{t}'] for t in range(min(100, n_timesteps)))

    print(f"\nFirst 100 timesteps:")
    print(f"  Total battery charge: {total_charge:.2f} kW")
    print(f"  Total battery discharge: {total_discharge:.2f} kW")

    print("\n" + "=" * 80)
    print("The sparse matrix implementation successfully handled the large-scale problem!")
    print("Without sparse matrices, this would have required ~81 GiB of memory.")
    print("With sparse matrices, it uses only a few MiB.")
    print("=" * 80)
else:
    print(f"\nOptimization FAILED: {result['message']}")
    exit(1)
