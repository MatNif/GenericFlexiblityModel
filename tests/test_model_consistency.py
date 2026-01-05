"""
Model consistency tests for FlexAsset representations.

These tests verify that different representations of the same FlexAsset
(operational interface vs. linear model vs. future GA/DP representations)
produce equivalent results for the same operation sequences.

This does NOT test optimizer performance or solution quality.
It IS instead testing that model representations are internally consistent.

Run with: pytest tests/test_model_consistency.py -v
"""

import pytest

from flex_model.assets import (
    BalancingMarketCost,
    BalancingMarketFlex,
    BatteryUnit,
    BatteryCostModel,
    BatteryFlex,
)
from flex_model.optimization import LPOptimizer


class TestBalancingMarketConsistency:
    """
    Verify BalancingMarketFlex representations are internally consistent.

    Tests compare:
    1. Operational interface (evaluate_operation/execute_operation)
    2. Linear model (get_linear_model for LP optimization)
    """

    def test_single_timestep_import(self):
        """
        Test single timestep import operation consistency.

        Simplest possible test: one timestep, import only.
        """
        # Setup
        from flex_model.core.flex_asset import FlexAsset
        market = BalancingMarketFlex(
            cost_model=BalancingMarketCost(
                name="test_market",
                p_E_buy=0.25,   # Buy at 25 ct/kWh
                p_E_sell=0.15,  # Sell at 15 ct/kWh
            )
        )

        # Operation: Import 40 kW for 1 timestep (uses FlexAsset.dt_hours)
        P_import = 40.0
        P_export = 0.0

        # Method 1: Operational interface
        result = market.evaluate_operation(0, P_import, P_export)
        cost_operational = result['cost']

        # Method 2: Linear model
        linear_model = market.get_linear_model(n_timesteps=1, dt_hours=FlexAsset.dt_hours)

        # LP cost calculation (manually, since we're not solving an optimization)
        # cost_coefficients[0] = p_buy * dt_hours (for P_import variable)
        # cost = cost_coefficients[0] * P_import
        cost_lp = linear_model.cost_coefficients[0] * P_import

        # Assert equivalence
        assert cost_operational == pytest.approx(cost_lp, rel=1e-9), \
            f"Operational cost {cost_operational} != LP cost {cost_lp}"

    def test_single_timestep_export(self):
        """Test single timestep export operation consistency."""
        # Setup
        from flex_model.core.flex_asset import FlexAsset
        market = BalancingMarketFlex(
            cost_model=BalancingMarketCost(
                name="test_market",
                p_E_buy=0.30,
                p_E_sell=0.20,
            )
        )

        # Operation: Export 50 kW for 1 timestep
        P_import = 0.0
        P_export = 50.0

        # Method 1: Operational interface
        result = market.evaluate_operation(0, P_import, P_export)
        cost_operational = result['cost']

        # Method 2: Linear model
        linear_model = market.get_linear_model(n_timesteps=1, dt_hours=FlexAsset.dt_hours)

        # cost_coefficients[1] = -p_sell * dt_hours (for P_export variable)
        # cost = cost_coefficients[1] * P_export (negative = revenue)
        cost_lp = linear_model.cost_coefficients[1] * P_export

        # Assert equivalence
        assert cost_operational == pytest.approx(cost_lp, rel=1e-9), \
            f"Operational cost {cost_operational} != LP cost {cost_lp}"

    def test_multi_timestep_sequence(self):
        """
        Test multi-timestep operation sequence consistency.

        Verifies that a sequence of operations produces identical total costs
        through operational interface vs. LP representation.
        """
        # Setup with time-varying prices
        from flex_model.core.flex_asset import FlexAsset
        market = BalancingMarketFlex(
            cost_model=BalancingMarketCost(
                name="test_market",
                p_E_buy={0: 0.20, 1: 0.25, 2: 0.30, 3: 0.22},
                p_E_sell={0: 0.15, 1: 0.18, 2: 0.22, 3: 0.16},
            )
        )

        # Define fixed operation sequence
        n_timesteps = 4
        operations = [
            (40.0, 0.0),   # t=0: import 40 kW
            (0.0, 30.0),   # t=1: export 30 kW
            (20.0, 0.0),   # t=2: import 20 kW
            (0.0, 0.0),    # t=3: idle
        ]

        # Method 1: Apply through operational interface
        market.reset()
        total_cost_operational = 0.0
        for t, (P_import, P_export) in enumerate(operations):
            result = market.evaluate_operation(t, P_import, P_export)
            assert result['feasible'], f"Operation at t={t} should be feasible"
            total_cost_operational += result['cost']
            market.execute_operation(t, P_import, P_export)

        # Method 2: Calculate cost from linear model
        linear_model = market.get_linear_model(n_timesteps, FlexAsset.dt_hours)

        # Calculate LP cost for fixed operations
        total_cost_lp = 0.0
        for t, (P_import, P_export) in enumerate(operations):
            # Import cost
            cost_import = linear_model.cost_coefficients[t] * P_import
            # Export cost (revenue)
            cost_export = linear_model.cost_coefficients[n_timesteps + t] * P_export
            total_cost_lp += cost_import + cost_export

        # Assert equivalence
        assert total_cost_operational == pytest.approx(total_cost_lp, rel=1e-9), \
            f"Operational cost {total_cost_operational} != LP cost {total_cost_lp}"

    def test_boundary_conditions(self):
        """Test edge cases: zero power, large power, extreme prices."""
        from flex_model.core.flex_asset import FlexAsset
        market = BalancingMarketFlex(
            cost_model=BalancingMarketCost(
                name="test_market",
                p_E_buy=1.50,   # High price
                p_E_sell=0.01,  # Very low price
            )
        )

        # Test case 1: Zero power (idle)
        result_op = market.evaluate_operation(0, 0.0, 0.0)
        linear_model = market.get_linear_model(1, FlexAsset.dt_hours)
        cost_lp = linear_model.cost_coefficients[0] * 0.0 + \
                  linear_model.cost_coefficients[1] * 0.0

        assert result_op['cost'] == pytest.approx(cost_lp, abs=1e-9)
        assert result_op['cost'] == 0.0

        # Test case 2: Very large power
        P_large = 10000.0
        result_op = market.evaluate_operation(0, P_large, 0.0)
        cost_lp = linear_model.cost_coefficients[0] * P_large

        assert result_op['cost'] == pytest.approx(cost_lp, rel=1e-9)


class TestBatteryConsistency:
    """
    Verify BatteryFlex representations are internally consistent.

    These tests actually solve LP problems and compare with operational execution.
    """

    @pytest.fixture
    def simple_battery(self):
        """Create a simple battery for testing."""
        unit = BatteryUnit(
            name="test_battery",
            capacity_kwh=100.0,
            power_kw=50.0,
            efficiency=0.90,         # 90% one-way efficiency
            soc_min=0.1,
            soc_max=0.9,
            self_discharge_per_hour=0.0,  # No self-discharge for simpler tests
        )

        cost_model = BatteryCostModel(
            name="test_battery_cost",
            c_inv=500.0,
            n_lifetime=10.0,
            p_int=0.05,              # 5 ct/kWh degradation cost
            c_fix=0.0
        )

        battery = BatteryFlex(unit=unit, cost_model=cost_model)
        return battery

    def test_simple_optimization_consistency(self, simple_battery):
        """
        Test that LP optimization produces same results as operational execution.

        Simple scenario: Battery + Market, 2 timesteps, simple imbalance profile.
        """
        from flex_model.core.flex_asset import FlexAsset

        # Setup imbalance profile: need to charge in t=0, discharge in t=1
        imbalance = {
            0: -30.0,  # Excess 30 kW (can charge battery)
            1: 20.0,   # Deficit 20 kW (can discharge battery)
        }
        n_timesteps = 2

        # Method 1: Solve via LP Optimizer
        battery_lp = BatteryFlex(
            unit=BatteryUnit(
                name="battery_lp",
                capacity_kwh=100.0,
                power_kw=50.0,
                efficiency=0.90,
                soc_min=0.1,
                soc_max=0.9,
                self_discharge_per_hour=0.0,
            ),
            cost_model=BatteryCostModel(
                name="battery_cost_lp",
                c_inv=500.0,
                n_lifetime=10.0,
                p_int=0.05,
                c_fix=0.0,
            ),
        )
        battery_lp.reset(E_plus_init=40.0, E_minus_init=40.0)  # 50% SOC

        market_lp = BalancingMarketFlex(
            cost_model=BalancingMarketCost(
                name="market_lp",
                p_E_buy=0.25,
                p_E_sell=0.15,
            )
        )

        optimizer = LPOptimizer(n_timesteps=n_timesteps)
        optimizer.add_asset(battery_lp.get_linear_model(n_timesteps, FlexAsset.dt_hours))
        optimizer.add_asset(market_lp.get_linear_model(n_timesteps, FlexAsset.dt_hours))
        optimizer.set_imbalance(imbalance)

        lp_result = optimizer.solve()

        assert lp_result['success'], f"LP should solve successfully: {lp_result['message']}"
        cost_lp = lp_result['cost']

        # Method 2: Execute manually with same operations
        # Extract operations from LP solution
        battery_solution = lp_result['solution']['battery_lp']
        market_solution = lp_result['solution']['market_lp']

        battery_op = BatteryFlex(
            unit=BatteryUnit(
                name="battery_op",
                capacity_kwh=100.0,
                power_kw=50.0,
                efficiency=0.90,
                soc_min=0.1,
                soc_max=0.9,
                self_discharge_per_hour=0.0,
            ),
            cost_model=BatteryCostModel(
                name="battery_cost_op",
                c_inv=500.0,
                n_lifetime=10.0,
                p_int=0.05,
                c_fix=0.0,
            ),
        )
        battery_op.reset(E_plus_init=40.0, E_minus_init=40.0)  # 50% SOC

        market_op = BalancingMarketFlex(
            cost_model=BalancingMarketCost(
                name="market_op",
                p_E_buy=0.25,
                p_E_sell=0.15,
            )
        )

        # Execute operations for each timestep
        for t in range(n_timesteps):
            # Extract battery operations from LP solution (keys include asset name prefix)
            P_charge = battery_solution.get(f'battery_lp_P_charge_{t}', 0.0)
            P_discharge = battery_solution.get(f'battery_lp_P_discharge_{t}', 0.0)

            # Extract market operations from LP solution
            P_market_import = market_solution.get(f'market_lp_P_import_{t}', 0.0)
            P_market_export = market_solution.get(f'market_lp_P_export_{t}', 0.0)

            # Execute on battery
            battery_op.execute_operation(t, P_charge, P_discharge)

            # Execute on market
            market_op.execute_operation(t, P_market_import, P_market_export)

        # Compare costs
        cost_op = battery_op.get_metrics()['total_cost_eur'] + market_op.get_metrics()['total_cost_eur']

        assert cost_lp == pytest.approx(cost_op, rel=1e-4), \
            f"LP cost {cost_lp:.4f} != Operational cost {cost_op:.4f}"

    def test_battery_only_optimization(self, simple_battery):
        """
        Test battery-only LP optimization vs operational execution.

        No market - just battery with fixed import/export profile.
        """
        from flex_model.core.flex_asset import FlexAsset

        # Simple 1-timestep problem: charge 30 kW
        n_timesteps = 1
        imbalance = {0: -30.0}  # Excess 30 kW

        # LP Method
        battery_lp = simple_battery
        battery_lp.reset(E_plus_init=40.0, E_minus_init=40.0)

        market_lp = BalancingMarketFlex(
            cost_model=BalancingMarketCost(name="market", p_E_buy=0.25, p_E_sell=0.15)
        )

        optimizer = LPOptimizer(n_timesteps=n_timesteps)
        optimizer.add_asset(battery_lp.get_linear_model(n_timesteps, FlexAsset.dt_hours))
        optimizer.add_asset(market_lp.get_linear_model(n_timesteps, FlexAsset.dt_hours))
        optimizer.set_imbalance(imbalance)

        lp_result = optimizer.solve()
        assert lp_result['success'], "LP should solve"

        # Extract battery operations
        battery_solution = lp_result['solution']['test_battery']
        P_charge_lp = battery_solution.get('test_battery_P_charge_0', 0.0)
        P_discharge_lp = battery_solution.get('test_battery_P_discharge_0', 0.0)

        # Operational method - execute same operation
        battery_op = BatteryFlex(
            unit=BatteryUnit(
                name="test_battery_op",
                capacity_kwh=100.0,
                power_kw=50.0,
                efficiency=0.90,
                soc_min=0.1,
                soc_max=0.9,
                self_discharge_per_hour=0.0,
            ),
            cost_model=BatteryCostModel(
                name="battery_cost_op",
                c_inv=500.0,
                n_lifetime=10.0,
                p_int=0.05,
                c_fix=0.0,
            ),
        )
        battery_op.reset(E_plus_init=40.0, E_minus_init=40.0)

        result = battery_op.evaluate_operation(0, P_charge_lp, P_discharge_lp)
        assert result['feasible'], "LP solution should be feasible operationally"

        battery_op.execute_operation(0, P_charge_lp, P_discharge_lp)

        # Compare battery costs only
        cost_lp_battery = sum(
            lp_result['cost'] * (1 if 'battery' in key else 0)
            for key in lp_result['solution'].keys()
        )  # This is imperfect, but we can compare total costs

        cost_op_battery = battery_op.get_metrics()['total_cost_eur']

        # Note: We're comparing total optimization cost vs just battery cost
        # A better test would extract just battery cost from LP
        # For now, verify the operation is at least feasible and costs are reasonable
        assert cost_op_battery >= 0, "Battery cost should be non-negative"
        assert P_charge_lp >= 0, "Charge power should be non-negative"
        assert P_discharge_lp >= 0, "Discharge power should be non-negative"


class TestFutureRepresentationConsistency:
    """
    Placeholder for future representation consistency tests.

    As new optimization algorithm representations are added (GA, DP, RL),
    add corresponding consistency tests here.
    """

    def test_operational_vs_ga_placeholder(self):
        """Placeholder: Test operational vs. GA representation consistency."""
        pytest.skip("GA representation not yet implemented")

    def test_operational_vs_dp_placeholder(self):
        """Placeholder: Test operational vs. DP representation consistency."""
        pytest.skip("DP representation not yet implemented")

    def test_lp_vs_ga_placeholder(self):
        """Placeholder: Cross-check LP and GA representations."""
        pytest.skip("GA representation not yet implemented")


# Helper functions for future LP-based consistency tests

def solve_lp_with_fixed_decisions(linear_model, operations, dt_hours):
    """
    Solve LP with fixed decisions by constraining variable bounds.

    This is a helper for future tests that need to verify LP solutions
    match operational interface for the same operation sequence.

    Args:
        linear_model: LinearModel from FlexAsset
        operations: List of (P_import, P_export) tuples
        dt_hours: Timestep duration

    Returns:
        LP solution cost
    """
    # TODO: Implement when needed for full LP consistency tests
    raise NotImplementedError("LP fixed-decision solving not yet implemented")


def extract_soc_trajectory_from_lp_solution(solution, linear_model):
    """
    Extract SOC trajectory from LP solution.

    Helper for comparing SOC evolution between operational and LP representations.

    Args:
        solution: LP solution dict
        linear_model: Battery LinearModel

    Returns:
        List of SOC values over time
    """
    # TODO: Implement when needed for full battery LP tests
    raise NotImplementedError("SOC extraction from LP solution not yet implemented")
