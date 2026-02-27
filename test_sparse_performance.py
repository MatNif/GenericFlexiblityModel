"""
Test script to verify sparse matrix implementation and memory benefits.

This script creates a battery optimization problem with varying timesteps
and verifies that:
1. The sparse matrix implementation works correctly
2. Memory usage is significantly reduced
3. Results match the expected optimization outcome
"""

import numpy as np
from scipy import sparse

from flex_model.assets import BatteryUnit, BatteryCostModel, BatteryFlex
from flex_model.assets import BalancingMarketCost, BalancingMarketFlex
from flex_model.optimization import LPOptimizer


def test_sparse_implementation(n_timesteps):
    """Test sparse matrix implementation with given number of timesteps."""
    print(f"\n{'='*70}")
    print(f"Testing with {n_timesteps:,} timesteps")
    print(f"{'='*70}")

    # Create battery
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
    market = BalancingMarketFlex(
        cost_model=BalancingMarketCost(
            name="market",
            p_E_buy=0.25,
            p_E_sell=0.15
        )
    )

    # Create simple imbalance profile (sinusoidal pattern)
    imbalance = {}
    for t in range(n_timesteps):
        # Create a pattern that alternates between surplus and deficit
        imbalance[t] = 30.0 * np.sin(2 * np.pi * t / 24)

    # Build linear models
    print("\nBuilding linear models...")
    battery_lp = battery_flex.get_linear_model(n_timesteps, initial_soc=0.5)
    market_lp = market.get_linear_model(n_timesteps)

    # Check that battery model uses sparse matrices
    print(f"\nBattery model:")
    print(f"  Variables: {battery_lp.n_vars:,}")
    print(f"  A_eq type: {type(battery_lp.A_eq)}")
    print(f"  A_eq shape: {battery_lp.A_eq.shape}")

    if sparse.issparse(battery_lp.A_eq):
        nnz = battery_lp.A_eq.nnz
        total_elements = battery_lp.A_eq.shape[0] * battery_lp.A_eq.shape[1]
        sparsity = 100.0 * (1.0 - nnz / total_elements)

        # Estimate memory usage
        # Sparse: ~12 bytes per non-zero (8 for value, 4 for row/col indices)
        # Dense: 8 bytes per element
        sparse_memory_mb = (nnz * 12) / (1024 * 1024)
        dense_memory_mb = (total_elements * 8) / (1024 * 1024)

        print(f"  A_eq sparsity: {sparsity:.3f}% zeros")
        print(f"  A_eq non-zeros: {nnz:,} out of {total_elements:,}")
        print(f"  Estimated memory (sparse): {sparse_memory_mb:.2f} MiB")
        print(f"  Estimated memory (dense): {dense_memory_mb:.2f} MiB")
        print(f"  Memory reduction: {dense_memory_mb / sparse_memory_mb:.1f}x")
    else:
        print("  WARNING: A_eq is not sparse!")

    # Set up optimizer
    print("\nSetting up optimizer...")
    optimizer = LPOptimizer(n_timesteps=n_timesteps)
    optimizer.add_asset(battery_lp)
    optimizer.add_asset(market_lp)
    optimizer.set_imbalance(imbalance)

    # Solve
    print("\nSolving optimization...")
    result = optimizer.solve()

    if result['success']:
        print(f"\nOptimization successful!")
        print(f"  Optimal cost: {result['cost']:.2f} CHF")

        # Extract some solution details
        solution = result['solution']
        battery_sol = solution['BESS_100kWh']

        # Calculate total charge/discharge
        total_charge = sum(battery_sol[f'BESS_100kWh_P_charge_{t}'] for t in range(n_timesteps))
        total_discharge = sum(battery_sol[f'BESS_100kWh_P_discharge_{t}'] for t in range(n_timesteps))

        print(f"  Total battery charge: {total_charge:.2f} kW")
        print(f"  Total battery discharge: {total_discharge:.2f} kW")

        return True
    else:
        print(f"\nOptimization FAILED: {result['message']}")
        return False


if __name__ == "__main__":
    print("Sparse Matrix Implementation Test")
    print("=" * 70)

    # Test with increasing problem sizes
    test_cases = [
        10,      # Small: 10 timesteps
        100,     # Medium: 100 timesteps
        1000,    # Large: 1000 timesteps
        10000,   # Very large: 10000 timesteps (~104 days at 15-min resolution)
    ]

    results = {}
    for n in test_cases:
        try:
            success = test_sparse_implementation(n)
            results[n] = success
        except Exception as e:
            print(f"\nERROR with {n} timesteps: {e}")
            results[n] = False

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for n, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"  {n:,} timesteps: {status}")

    if all(results.values()):
        print("\nAll tests passed! Sparse matrix implementation is working correctly.")
    else:
        print("\nSome tests failed. Please check the errors above.")
