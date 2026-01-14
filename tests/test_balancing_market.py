"""
Unit tests for the BalancingMarket flexibility asset.

Tests the balancing market implementation including:
- BalancingMarketCost: Cost model for market prices
- BalancingMarketFlex: FlexAsset without physical constraints

Run with: pytest tests/test_balancing_market.py -v
"""

import pytest

from flex_model.assets import BalancingMarketCost, BalancingMarketFlex


class TestBalancingMarketCost:
    """Tests for BalancingMarketCost economic model."""

    def test_initialization(self):
        """Test basic initialization with constant prices."""
        cost = BalancingMarketCost(
            name="test_market",
            p_E_buy=0.25,
            p_E_sell=0.15,
        )

        assert cost.name == "test_market"
        assert cost.c_inv == 0.0  # No investment cost
        assert cost.c_fix == 0.0  # No fixed cost
        assert cost.p_E_buy(0) == 0.25
        assert cost.p_E_sell(0) == 0.15

    def test_time_varying_prices(self):
        """Test time-varying market prices using dict."""
        cost = BalancingMarketCost(
            name="test_market",
            p_E_buy={0: 0.20, 1: 0.25, 2: 0.30},
            p_E_sell={0: 0.15, 1: 0.18, 2: 0.22},
        )

        assert cost.p_E_buy(0) == 0.20
        assert cost.p_E_buy(1) == 0.25
        assert cost.p_E_buy(2) == 0.30
        assert cost.p_E_sell(0) == 0.15
        assert cost.p_E_sell(1) == 0.18
        assert cost.p_E_sell(2) == 0.22

    def test_step_cost_import_only(self):
        """Test cost calculation for pure import (buying from market)."""
        cost = BalancingMarketCost(
            name="test_market",
            p_E_buy=0.25,
            p_E_sell=0.15,
        )

        activation = {
            'P_grid_import': 40.0,  # Buy 40 kW
            'P_grid_export': 0.0,
        }

        step_cost = cost.step_cost(t=0, flex_state=None, activation=activation)

        # Cost = 40 kW * 0.25 h * 0.25 CHF/kWh = 2.5 CHF
        expected_cost = 40.0 * 0.25 * 0.25
        assert step_cost == pytest.approx(expected_cost)

    def test_step_cost_export_only(self):
        """Test cost calculation for pure export (selling to market)."""
        cost = BalancingMarketCost(
            name="test_market",
            p_E_buy=0.25,
            p_E_sell=0.15,
        )

        activation = {
            'P_grid_import': 0.0,
            'P_grid_export': 30.0,  # Sell 30 kW
        }

        step_cost = cost.step_cost(t=0, flex_state=None, activation=activation)

        # Revenue = 30 kW * 0.25 h * 0.15 CHF/kWh = 1.125 CHF (negative cost)
        expected_cost = -30.0 * 0.25 * 0.15
        assert step_cost == pytest.approx(expected_cost)

    def test_step_cost_with_spread(self):
        """Test that market spread (buy > sell) is captured correctly."""
        cost = BalancingMarketCost(
            name="test_market",
            p_E_buy=0.30,  # Higher buy price
            p_E_sell=0.10,  # Lower sell price
        )

        # Buy 10 kWh
        activation_buy = {
            'P_grid_import': 40.0,
            'P_grid_export': 0.0,
        }
        cost_buy = cost.step_cost(t=0, flex_state=None, activation=activation_buy)

        # Sell 10 kWh
        activation_sell = {
            'P_grid_import': 0.0,
            'P_grid_export': 40.0,
        }
        cost_sell = cost.step_cost(t=0, flex_state=None, activation=activation_sell)

        # Buying should be more expensive than selling revenue
        E = 40.0 * 0.25  # 10 kWh
        assert cost_buy == pytest.approx(E * 0.30)  # 3.0 CHF
        assert cost_sell == pytest.approx(-E * 0.10)  # -1.0 CHF
        assert abs(cost_buy) > abs(cost_sell)  # Market spread


class TestBalancingMarketFlex:
    """Tests for BalancingMarketFlex operational interface."""

    def test_initialization(self):
        """Test basic initialization."""
        cost = BalancingMarketCost(
            name="test_market",
            p_E_buy=0.25,
            p_E_sell=0.15,
        )
        market = BalancingMarketFlex(cost_model=cost)

        assert market.name == "test_market"
        assert market.cost_model is cost
        assert market._total_cost_eur == 0.0
        assert market._total_throughput_kwh == 0.0
        assert market._num_activations == 0

    def test_custom_name(self):
        """Test initialization with custom name."""
        cost = BalancingMarketCost(name="cost_model", p_E_buy=0.25)
        market = BalancingMarketFlex(cost_model=cost, name="custom_market")

        assert market.name == "custom_market"

    def test_evaluate_always_feasible(self):
        """Test that market operations are always feasible."""
        cost = BalancingMarketCost(name="test_market", p_E_buy=0.25, p_E_sell=0.15)
        market = BalancingMarketFlex(cost_model=cost)

        # Large import - still feasible
        result = market.evaluate_operation(
            t=0,
            P_grid_import=1000.0,
            P_grid_export=0.0,
        )
        assert result['feasible'] is True
        assert len(result['violations']) == 0

        # Large export - still feasible
        result = market.evaluate_operation(
            t=0,
            P_grid_import=0.0,
            P_grid_export=1000.0,
        )
        assert result['feasible'] is True
        assert len(result['violations']) == 0

    def test_evaluate_cost_calculation(self):
        """Test cost calculation in evaluate_operation."""
        cost = BalancingMarketCost(name="test_market", p_E_buy=0.25, p_E_sell=0.15)
        market = BalancingMarketFlex(cost_model=cost)

        result = market.evaluate_operation(
            t=0,
            P_grid_import=40.0,
            P_grid_export=0.0,
        )

        assert result['feasible'] is True
        # Cost = 40 kW * 0.25 h * 0.25 CHF/kWh = 2.5 CHF
        expected_cost = 40.0 * 0.25 * 0.25
        assert result['cost'] == pytest.approx(expected_cost)
        assert result['E_import'] == pytest.approx(10.0)
        assert result['E_export'] == pytest.approx(0.0)

    def test_evaluate_is_stateless(self):
        """Test that evaluate_operation doesn't modify state."""
        cost = BalancingMarketCost(name="test_market", p_E_buy=0.25)
        market = BalancingMarketFlex(cost_model=cost)

        # Call evaluate multiple times
        for _ in range(5):
            market.evaluate_operation(
                t=0,
                P_grid_import=40.0,
                P_grid_export=0.0,
            )

        # State should still be zero
        assert market._total_cost_eur == 0.0
        assert market._total_throughput_kwh == 0.0
        assert market._num_activations == 0

    def test_execute_updates_tracking(self):
        """Test that execute_operation updates metrics."""
        cost = BalancingMarketCost(name="test_market", p_E_buy=0.25, p_E_sell=0.15)
        market = BalancingMarketFlex(cost_model=cost)

        # Execute import operation
        market.execute_operation(
            t=0,
            P_grid_import=40.0,
            P_grid_export=0.0,
        )

        # Check tracking
        expected_cost = 40.0 * 0.25 * 0.25
        expected_throughput = 40.0 * 0.25
        assert market._total_cost_eur == pytest.approx(expected_cost)
        assert market._total_throughput_kwh == pytest.approx(expected_throughput)
        assert market._num_activations == 1

    def test_execute_accumulates_metrics(self):
        """Test that multiple executions accumulate correctly."""
        cost = BalancingMarketCost(name="test_market", p_E_buy=0.25, p_E_sell=0.15)
        market = BalancingMarketFlex(cost_model=cost)

        # Execute 3 operations
        for _ in range(3):
            market.execute_operation(
                t=0,
                P_grid_import=40.0,
                P_grid_export=0.0,
            )

        # Check accumulated values
        expected_cost = 3 * (40.0 * 0.25 * 0.25)
        expected_throughput = 3 * (40.0 * 0.25)
        assert market._total_cost_eur == pytest.approx(expected_cost)
        assert market._total_throughput_kwh == pytest.approx(expected_throughput)
        assert market._num_activations == 3

    def test_power_limits_infinite(self):
        """Test that market has no power limits."""
        cost = BalancingMarketCost(name="test_market", p_E_buy=0.25)
        market = BalancingMarketFlex(cost_model=cost)

        P_import_max, P_export_max = market.power_limits(t=0)

        assert P_import_max == float('inf')
        assert P_export_max == float('inf')

    def test_reset(self):
        """Test reset functionality."""
        cost = BalancingMarketCost(name="test_market", p_E_buy=0.25)
        market = BalancingMarketFlex(cost_model=cost)

        # Execute some operations
        for _ in range(3):
            market.execute_operation(
                t=0,
                P_grid_import=40.0,
                P_grid_export=0.0,
            )

        # Reset
        market.reset()

        # Check everything is zero
        assert market._total_cost_eur == 0.0
        assert market._total_throughput_kwh == 0.0
        assert market._num_activations == 0

    def test_get_metrics(self):
        """Test get_metrics returns correct values."""
        cost = BalancingMarketCost(name="test_market", p_E_buy=0.25)
        market = BalancingMarketFlex(cost_model=cost)

        # Execute operations
        market.execute_operation(t=0, P_grid_import=40.0, P_grid_export=0.0)
        market.execute_operation(t=1, P_grid_import=20.0, P_grid_export=0.0)

        metrics = market.get_metrics()

        expected_cost = (40.0 * 0.25 * 0.25) + (20.0 * 0.25 * 0.25)
        expected_throughput = (40.0 * 0.25) + (20.0 * 0.25)

        assert metrics['total_cost_eur'] == pytest.approx(expected_cost)
        assert metrics['total_throughput_kwh'] == pytest.approx(expected_throughput)
        assert metrics['num_activations'] == 2


class TestBalancingMarketIntegration:
    """Integration tests comparing market with battery."""

    def test_market_vs_battery_interface(self):
        """Test that market and battery have compatible interfaces."""
        from flex_model.assets import BatteryUnit, BatteryCostModel, BatteryFlex

        # Create battery
        battery_unit = BatteryUnit(
            name="test_battery",
            capacity_kwh=100.0,
            power_kw=50.0,
            efficiency=0.95,
        )
        battery_cost = BatteryCostModel(
            name="battery_cost",
            c_inv=500.0,
            n_lifetime=10.0,
            p_int=0.05
        )
        battery = BatteryFlex(unit=battery_unit, cost_model=battery_cost)
        battery.reset(E_plus_init=50.0, E_minus_init=50.0)

        # Create market
        market_cost = BalancingMarketCost(
            name="market",
            p_E_buy=0.25,
            p_E_sell=0.15,
        )
        market = BalancingMarketFlex(cost_model=market_cost)

        # Both should have same interface methods
        assert hasattr(battery, 'evaluate_operation')
        assert hasattr(market, 'evaluate_operation')
        assert hasattr(battery, 'execute_operation')
        assert hasattr(market, 'execute_operation')
        assert hasattr(battery, 'power_limits')
        assert hasattr(market, 'power_limits')
        assert hasattr(battery, 'reset')
        assert hasattr(market, 'reset')
        assert hasattr(battery, 'get_metrics')
        assert hasattr(market, 'get_metrics')

    def test_time_varying_prices(self):
        """Test market with time-varying prices over multiple steps."""
        # Peak/off-peak pricing
        cost = BalancingMarketCost(
            name="test_market",
            p_E_buy={0: 0.15, 1: 0.15, 2: 0.30, 3: 0.30, 4: 0.15},  # Peak at t=2,3
            p_E_sell={0: 0.10, 1: 0.10, 2: 0.20, 3: 0.20, 4: 0.10},
        )
        market = BalancingMarketFlex(cost_model=cost)

        # Buy during off-peak (cheaper)
        result_offpeak = market.evaluate_operation(
            t=0,
            P_grid_import=10.0,
            P_grid_export=0.0,
        )

        # Buy during peak (expensive)
        result_peak = market.evaluate_operation(
            t=2,
            P_grid_import=10.0,
            P_grid_export=0.0,
        )

        # Peak should be more expensive
        assert result_peak['cost'] > result_offpeak['cost']
        assert result_offpeak['cost'] == pytest.approx(10.0 * 1.0 * 0.15)  # 1.5 CHF
        assert result_peak['cost'] == pytest.approx(10.0 * 1.0 * 0.30)  # 3.0 CHF