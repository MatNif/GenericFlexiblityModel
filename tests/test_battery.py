"""
Unit tests for battery energy storage implementation.

Tests cover:
    1. BatteryUnit: Physical behavior, power limits, SOC tracking, efficiency
    2. BatteryCostModel: Cost calculation, degradation, energy arbitrage
    3. BatteryFlex: Operational interface, feasibility checks, execution

Run tests with: pytest tests/test_battery.py -v
"""

import pytest
from flex_model.assets.battery import BatteryUnit, BatteryCostModel, BatteryFlex


class TestBatteryUnit:
    """Tests for BatteryUnit physical model."""

    def test_initialization(self):
        """Test battery initialization with valid parameters."""
        battery = BatteryUnit(
            name="test_battery",
            capacity_kwh=100.0,
            power_kw=50.0,
            efficiency=0.95,
        )

        assert battery.name == "test_battery"
        assert battery.C_spec == 100.0
        assert battery.power_kw == 50.0
        assert battery.efficiency == 0.95
        assert battery.soc_min == 0.0
        assert battery.soc_max == 1.0

    def test_invalid_efficiency(self):
        """Test that invalid efficiency raises ValueError."""
        with pytest.raises(ValueError, match="Efficiency"):
            BatteryUnit(name="bad", capacity_kwh=100, power_kw=50, efficiency=1.5)

        with pytest.raises(ValueError, match="Efficiency"):
            BatteryUnit(name="bad", capacity_kwh=100, power_kw=50, efficiency=0.0)

    def test_invalid_soc_limits(self):
        """Test that invalid SOC limits raise ValueError."""
        with pytest.raises(ValueError, match="SOC limits"):
            BatteryUnit(name="bad", capacity_kwh=100, power_kw=50, soc_min=0.8, soc_max=0.2)

        with pytest.raises(ValueError, match="SOC limits"):
            BatteryUnit(name="bad", capacity_kwh=100, power_kw=50, soc_min=-0.1, soc_max=1.0)

    def test_power_limits_empty_battery(self):
        """Test power limits when battery is empty."""
        battery = BatteryUnit(name="test", capacity_kwh=100, power_kw=50)
        battery.reset_state(E_plus_init=0.0, E_minus_init=100.0)

        P_draw_max, P_inject_max = battery.power_limits(t=0)

        # Can charge at full power
        assert P_draw_max == pytest.approx(50.0, rel=0.1)
        # Cannot discharge (no energy stored)
        assert P_inject_max == pytest.approx(0.0, abs=0.1)

    def test_power_limits_full_battery(self):
        """Test power limits when battery is full."""
        battery = BatteryUnit(name="test", capacity_kwh=100, power_kw=50)
        battery.reset_state(E_plus_init=100.0, E_minus_init=0.0)

        P_draw_max, P_inject_max = battery.power_limits(t=0)

        # Cannot charge (no capacity remaining)
        assert P_draw_max == pytest.approx(0.0, abs=0.1)
        # Can discharge at full power
        assert P_inject_max == pytest.approx(50.0, rel=0.1)

    def test_power_limits_half_full(self):
        """Test power limits at 50% SOC."""
        battery = BatteryUnit(name="test", capacity_kwh=100, power_kw=50)
        battery.reset_state(E_plus_init=50.0, E_minus_init=50.0)

        P_draw_max, P_inject_max = battery.power_limits(t=0)

        # Can charge and discharge at full power (50 kWh / 0.25h = 200 kW > 50 kW)
        assert P_draw_max == pytest.approx(50.0, rel=0.1)
        assert P_inject_max == pytest.approx(50.0, rel=0.1)

    def test_soc_calculation(self):
        """Test SOC calculation."""
        battery = BatteryUnit(name="test", capacity_kwh=100, power_kw=50)

        # Empty
        battery.reset_state(E_plus_init=0.0, E_minus_init=100.0)
        assert battery.soc() == pytest.approx(0.0)

        # Half full
        battery.reset_state(E_plus_init=50.0, E_minus_init=50.0)
        assert battery.soc() == pytest.approx(0.5)

        # Full
        battery.reset_state(E_plus_init=100.0, E_minus_init=0.0)
        assert battery.soc() == pytest.approx(1.0)

    def test_charging_with_efficiency(self):
        """Test that charging respects efficiency."""
        battery = BatteryUnit(name="test", capacity_kwh=100, power_kw=50, efficiency=0.9)
        battery.reset_state(E_plus_init=50.0, E_minus_init=50.0)

        # Charge at 40 kW for 0.25 hours = 10 kWh
        # With 90% efficiency, only 9 kWh is stored
        battery.update_state(t=0, dt_hours=0.25, P_draw_cmd=40.0, P_inject_cmd=0.0)

        # SOC should increase by approximately 9 kWh
        assert battery.E_plus == pytest.approx(50.0 + 9.0, abs=1.0)

    def test_discharging_with_efficiency(self):
        """Test that discharging respects efficiency."""
        battery = BatteryUnit(name="test", capacity_kwh=100, power_kw=50, efficiency=0.9)
        battery.reset_state(E_plus_init=50.0, E_minus_init=50.0)

        # Discharge at 40 kW for 0.25 hours = 10 kWh delivered to grid
        # With 90% efficiency, need to take 10/0.9 = 11.1 kWh from storage
        battery.update_state(t=0, dt_hours=0.25, P_draw_cmd=0.0, P_inject_cmd=40.0)

        # E_plus should decrease by approximately 11.1 kWh
        assert battery.E_plus == pytest.approx(50.0 - 11.1, abs=1.0)

    def test_self_discharge(self):
        """Test self-discharge over time."""
        battery = BatteryUnit(
            name="test",
            capacity_kwh=100,
            power_kw=50,
            self_discharge_per_hour=0.01,  # 1% per hour
        )
        battery.reset_state(E_plus_init=100.0, E_minus_init=0.0)

        # Wait 1 hour with no operation
        battery.update_state(t=0, dt_hours=1.0, P_draw_cmd=0.0, P_inject_cmd=0.0)

        # Should lose approximately 1% = 1 kWh
        assert battery.E_plus == pytest.approx(99.0, abs=0.5)


class TestBatteryCostModel:
    """Tests for BatteryCostModel economic evaluation."""

    def test_initialization(self):
        """Test cost model initialization."""
        cost = BatteryCostModel(
            name="test_cost",
            c_inv=500.0,
            n_lifetime=10.0,
            p_int=0.05,
            p_E_buy=0.25,
            p_E_sell=0.20,
        )

        assert cost.name == "test_cost"
        assert cost.c_inv == 500.0
        assert cost.n_lifetime == 10.0
        assert cost.p_int(0) == 0.05
        assert cost.p_E_buy(0) == 0.25
        assert cost.p_E_sell(0) == 0.20

    def test_step_cost_charging(self):
        """Test cost calculation when charging."""
        cost = BatteryCostModel(
            name="test",
            c_inv=500.0,
            n_lifetime=10.0,
            p_int=0.05,  # CHF/kWh degradation
            p_E_buy=0.25,  # CHF/kWh buy price
            p_E_sell=0.20,  # CHF/kWh sell price
        )

        flex_state = {'soc': 0.5, 'E_plus': 50.0, 'E_minus': 50.0}
        activation = {'P_draw': 40.0, 'P_inject': 0.0, 'dt_hours': 0.25}

        step_cost = cost.step_cost(t=0, flex_state=flex_state, activation=activation)

        # Degradation: 40 kW * 0.25 h * 0.05 CHF/kWh = 0.50 CHF
        # Energy cost: 40 kW * 0.25 h * 0.25 CHF/kWh = 2.50 CHF
        # Total: 3.00 CHF
        assert step_cost == pytest.approx(3.00, abs=0.01)

    def test_step_cost_discharging(self):
        """Test cost calculation when discharging (revenue)."""
        cost = BatteryCostModel(
            name="test",
            c_inv=500.0,
            n_lifetime=10.0,
            p_int=0.05,  # CHF/kWh degradation
            p_E_buy=0.25,  # CHF/kWh buy price
            p_E_sell=0.20,  # CHF/kWh sell price
        )

        flex_state = {'soc': 0.5, 'E_plus': 50.0, 'E_minus': 50.0}
        activation = {'P_draw': 0.0, 'P_inject': 40.0, 'dt_hours': 0.25}

        step_cost = cost.step_cost(t=0, flex_state=flex_state, activation=activation)

        # Degradation: 40 kW * 0.25 h * 0.05 CHF/kWh = 0.50 CHF
        # Energy revenue: 40 kW * 0.25 h * 0.20 CHF/kWh = 2.00 CHF (negative cost)
        # Total: 0.50 - 2.00 = -1.50 CHF (net revenue)
        assert step_cost == pytest.approx(-1.50, abs=0.01)

    def test_time_varying_prices(self):
        """Test cost calculation with time-varying prices."""
        cost = BatteryCostModel(
            name="test",
            c_inv=500.0,
            n_lifetime=10.0,
            p_int=0.05,
            p_E_buy={0: 0.20, 1: 0.30, 2: 0.25},  # Time-varying
            p_E_sell={0: 0.18, 1: 0.28, 2: 0.23},
        )

        flex_state = {'soc': 0.5, 'E_plus': 50.0, 'E_minus': 50.0}
        activation = {'P_draw': 40.0, 'P_inject': 0.0, 'dt_hours': 0.25}

        # At t=0: buy price = 0.20
        cost_t0 = cost.step_cost(t=0, flex_state=flex_state, activation=activation)
        # Degradation: 0.50, Energy: 40*0.25*0.20 = 2.00, Total: 2.50
        assert cost_t0 == pytest.approx(2.50, abs=0.01)

        # At t=1: buy price = 0.30
        cost_t1 = cost.step_cost(t=1, flex_state=flex_state, activation=activation)
        # Degradation: 0.50, Energy: 40*0.25*0.30 = 3.00, Total: 3.50
        assert cost_t1 == pytest.approx(3.50, abs=0.01)

    def test_annualized_investment(self):
        """Test annualized investment calculation."""
        cost = BatteryCostModel(
            name="test",
            c_inv=500.0,  # CHF/kWh
            n_lifetime=10.0,
            p_int=0.05,
        )

        # 100 kWh battery, 10 years, 5% discount rate
        capex_annual = cost.annualized_investment(capacity=100.0, discount_rate=0.05)

        # Total investment: 500 * 100 = 50,000 CHF
        # Annuity factor (r=0.05, n=10): 0.1295
        # Annual: 50,000 * 0.1295 = 6,475 CHF/a
        assert capex_annual == pytest.approx(6475.0, rel=0.01)


class TestBatteryFlex:
    """Tests for BatteryFlex operational composition."""

    def setup_method(self):
        """Create battery flex asset for each test."""
        self.unit = BatteryUnit(
            name="test_battery",
            capacity_kwh=100.0,
            power_kw=50.0,
            efficiency=0.95,
        )

        self.cost_model = BatteryCostModel(
            name="test_cost",
            c_inv=500.0,
            n_lifetime=10.0,
            p_int=0.05,
            p_E_buy=0.25,
            p_E_sell=0.20,
        )

        self.battery_flex = BatteryFlex(unit=self.unit, cost_model=self.cost_model)
        self.battery_flex.reset(E_plus_init=50.0, E_minus_init=50.0)

    def test_evaluate_feasible_charging(self):
        """Test evaluation of feasible charging operation."""
        result = self.battery_flex.evaluate_operation(
            t=0, dt_hours=0.25, P_draw_cmd=40.0, P_inject_cmd=0.0
        )

        assert result['feasible'] is True
        assert len(result['violations']) == 0
        assert result['cost'] > 0  # Charging costs money
        assert 0 <= result['soc'] <= 1
        assert result['throughput'] == pytest.approx(10.0)  # 40 kW * 0.25 h

    def test_evaluate_feasible_discharging(self):
        """Test evaluation of feasible discharging operation."""
        result = self.battery_flex.evaluate_operation(
            t=0, dt_hours=0.25, P_draw_cmd=0.0, P_inject_cmd=40.0
        )

        assert result['feasible'] is True
        assert len(result['violations']) == 0
        assert result['cost'] < 0  # Discharging generates revenue
        assert 0 <= result['soc'] <= 1

    def test_evaluate_exceeds_power_limit(self):
        """Test evaluation when power exceeds limits."""
        result = self.battery_flex.evaluate_operation(
            t=0, dt_hours=0.25, P_draw_cmd=100.0, P_inject_cmd=0.0  # Exceeds 50 kW limit
        )

        assert result['feasible'] is False
        assert any('exceeds limit' in v for v in result['violations'])
        assert result['cost'] == 0.0  # Infeasible operations have zero cost

    def test_evaluate_simultaneous_charge_discharge(self):
        """Test that simultaneous charging and discharging is infeasible."""
        result = self.battery_flex.evaluate_operation(
            t=0, dt_hours=0.25, P_draw_cmd=20.0, P_inject_cmd=20.0
        )

        assert result['feasible'] is False
        assert any('simultaneously' in v for v in result['violations'])

    def test_evaluate_soc_min_violation(self):
        """Test detection of SOC minimum violation."""
        # Set up battery with minimal charge (5 kWh stored)
        # Power limit will be: min(50, 5*4) = 20 kW
        self.unit.reset_state(E_plus_init=5.0, E_minus_init=95.0)

        # Try to discharge 15 kW for 0.25h = 3.75 kWh gross
        # With 95% efficiency, this needs 3.75/0.95 = 3.95 kWh from storage
        # This is within power limit (20 kW) but would violate soc_min
        result = self.battery_flex.evaluate_operation(
            t=0, dt_hours=0.25, P_draw_cmd=0.0, P_inject_cmd=15.0
        )

        # Should fail either due to power limit or SOC limit
        # Both are valid reasons for infeasibility
        assert result['feasible'] is False
        assert len(result['violations']) > 0

    def test_execute_operation(self):
        """Test execution of operation updates state."""
        initial_soc = self.unit.soc()
        initial_cost = self.battery_flex._total_cost_eur
        initial_throughput = self.battery_flex._total_throughput_kwh

        # Execute charging operation
        self.battery_flex.execute_operation(
            t=0, dt_hours=0.25, P_draw_cmd=40.0, P_inject_cmd=0.0
        )

        # Check state updated
        assert self.unit.soc() > initial_soc  # SOC increased
        assert self.battery_flex._total_cost_eur > initial_cost  # Cost accumulated
        assert self.battery_flex._total_throughput_kwh > initial_throughput  # Throughput tracked

    def test_get_metrics(self):
        """Test metrics retrieval."""
        # Execute some operations
        self.battery_flex.execute_operation(t=0, dt_hours=0.25, P_draw_cmd=40.0, P_inject_cmd=0.0)
        self.battery_flex.execute_operation(t=1, dt_hours=0.25, P_draw_cmd=0.0, P_inject_cmd=30.0)

        metrics = self.battery_flex.get_metrics()

        assert 'total_throughput_kwh' in metrics
        assert 'total_cost_eur' in metrics
        assert 'num_activations' in metrics
        assert 'current_soc' in metrics
        assert 'current_energy_stored' in metrics

        assert metrics['num_activations'] == 2
        assert metrics['total_throughput_kwh'] > 0

    def test_reset(self):
        """Test reset functionality."""
        # Do some operations
        self.battery_flex.execute_operation(t=0, dt_hours=0.25, P_draw_cmd=40.0, P_inject_cmd=0.0)

        # Reset
        self.battery_flex.reset(E_plus_init=50.0, E_minus_init=50.0)

        # Check all metrics cleared
        assert self.battery_flex._total_cost_eur == 0.0
        assert self.battery_flex._total_throughput_kwh == 0.0
        assert self.battery_flex._num_activations == 0
        assert self.unit.soc() == pytest.approx(0.5)


class TestBatteryIntegration:
    """Integration tests for complete battery workflow."""

    def test_charge_discharge_cycle(self):
        """Test a complete charge-discharge cycle."""
        # Setup
        unit = BatteryUnit(name="battery", capacity_kwh=100, power_kw=50, efficiency=0.95)
        cost = BatteryCostModel(
            name="cost", c_inv=500, n_lifetime=10, p_int=0.05,
            p_E_buy=0.25, p_E_sell=0.20
        )
        battery = BatteryFlex(unit=unit, cost_model=cost)
        battery.reset(E_plus_init=50.0, E_minus_init=50.0)

        initial_soc = unit.soc()

        # Charge
        battery.execute_operation(t=0, dt_hours=0.25, P_draw_cmd=40.0, P_inject_cmd=0.0)
        soc_after_charge = unit.soc()
        assert soc_after_charge > initial_soc

        # Discharge
        battery.execute_operation(t=1, dt_hours=0.25, P_draw_cmd=0.0, P_inject_cmd=40.0)
        soc_after_discharge = unit.soc()

        # Due to round-trip efficiency, SOC should be slightly lower than initial
        assert soc_after_discharge < soc_after_charge
        assert soc_after_discharge < initial_soc

    def test_arbitrage_scenario(self):
        """Test energy arbitrage: charge when cheap, discharge when expensive."""
        unit = BatteryUnit(name="battery", capacity_kwh=100, power_kw=50, efficiency=0.95)
        cost = BatteryCostModel(
            name="cost",
            c_inv=500,
            n_lifetime=10,
            p_int=0.05,
            p_E_buy={0: 0.15, 1: 0.35},  # Low at t=0, high at t=1
            p_E_sell={0: 0.13, 1: 0.33},
        )
        battery = BatteryFlex(unit=unit, cost_model=cost)
        battery.reset(E_plus_init=50.0, E_minus_init=50.0)

        # Charge at t=0 (cheap)
        result_charge = battery.evaluate_operation(t=0, dt_hours=0.25, P_draw_cmd=40.0, P_inject_cmd=0.0)
        battery.execute_operation(t=0, dt_hours=0.25, P_draw_cmd=40.0, P_inject_cmd=0.0)

        # Discharge at t=1 (expensive)
        result_discharge = battery.evaluate_operation(t=1, dt_hours=0.25, P_draw_cmd=0.0, P_inject_cmd=40.0)
        battery.execute_operation(t=1, dt_hours=0.25, P_draw_cmd=0.0, P_inject_cmd=40.0)

        # Revenue from discharge should exceed cost of charge (minus degradation)
        metrics = battery.get_metrics()
        # Net cost should be relatively small or negative (profit)
        # Exact value depends on efficiency and degradation
        assert metrics['total_cost_eur'] < result_charge['cost']  # Some revenue earned