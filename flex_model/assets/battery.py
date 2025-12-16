"""
Battery Energy Storage System (BESS) implementation.

This module provides a complete implementation of battery energy storage as a flexibility
asset, demonstrating the three-layer architecture:
    1. BatteryUnit: Physical model (power limits, efficiency, SOC tracking)
    2. BatteryCostModel: Economic model (degradation, energy arbitrage)
    3. BatteryFlex: Operational composition (feasibility + cost evaluation)

BATTERY PHYSICAL MODEL
-----------------------
The BatteryUnit models a battery with:
    - Energy capacity [kWh]: Total storage capacity
    - Power rating [kW]: Maximum charge/discharge power (C-rate based)
    - Round-trip efficiency: Energy loss during charge/discharge cycles
    - Self-discharge: Passive energy loss over time
    - SOC limits: Optional minimum/maximum state of charge

State of Charge (SOC):
    - SOC = E_plus / capacity
    - E_plus: Energy available to discharge
    - E_minus: Energy capacity available to charge
    - For a battery: E_plus + E_minus ≈ capacity (accounting for efficiency)

Power limits:
    - Charge power limited by: power rating, available capacity (E_minus), C-rate
    - Discharge power limited by: power rating, stored energy (E_plus), C-rate
    - Both respect SOC limits if configured

Efficiency modeling:
    - Charging: E_stored = E_charged * efficiency
    - Discharging: E_discharged = E_stored * efficiency
    - Round-trip efficiency = charge_eff * discharge_eff

BATTERY ECONOMIC MODEL
----------------------
The BatteryCostModel captures:
    - Investment cost [CHF/kWh]: Upfront battery cost (amortized via c_inv/n_lifetime)
    - Variable O&M cost [CHF/kWh]: Usage-dependent maintenance
    - Energy arbitrage: Buy low (charge), sell high (discharge)

Cost components:
    - Fixed: Investment amortization (handled outside step_cost)
    - Variable utilization: p_int * throughput (cycling-dependent maintenance)
    - Energy: Asymmetric buy/sell prices capture arbitrage opportunities

Note on degradation:
    - Battery degradation is captured via investment cost amortization (c_inv/n_lifetime)
    - The p_int parameter represents variable O&M, NOT replacement costs
    - This separates CAPEX-related degradation from OPEX-related maintenance

BATTERY OPERATIONAL MODEL
--------------------------
The BatteryFlex combines physics and economics:
    - evaluate_operation(): Check feasibility, calculate cost
    - execute_operation(): Update SOC, track degradation and costs

Typical use case:
    1. Optimizer proposes: "Charge at 30 kW for 15 minutes"
    2. BatteryFlex.evaluate_operation(): Check if feasible, compute cost
    3. If optimal: BatteryFlex.execute_operation(): Update battery state

EXAMPLE USAGE
-------------
# 1. Create physical unit
battery_unit = BatteryUnit(
    name="BESS_100kWh",
    capacity_kwh=100.0,
    power_kw=50.0,
    efficiency=0.95,
    self_discharge_per_hour=0.0001,
)

# 2. Create cost model
battery_cost = BatteryCostModel(
    name="battery_economics",
    c_inv=500.0,  # CHF/kWh
    n_lifetime=10.0,
    p_int=0.05,  # CHF/kWh degradation
    p_E_buy={0: 0.20, 1: 0.25, ...},  # Time-varying prices
    p_E_sell={0: 0.18, 1: 0.23, ...},
)

# 3. Compose flex asset
battery_flex = BatteryFlex(unit=battery_unit, cost_model=battery_cost)

# 4. Initialize state (50% SOC)
battery_flex.reset(E_plus_init=50.0, E_minus_init=50.0)

# 5. Evaluate operation
result = battery_flex.evaluate_operation(t=10, dt_hours=0.25, P_draw=0.0, P_inject=30.0)
if result['feasible']:
    print(f"Cost: {result['cost']:.2f} CHF, SOC after: {result['soc']:.1%}")

# 6. Execute operation
battery_flex.execute_operation(t=10, dt_hours=0.25, P_draw=0.0, P_inject=30.0)

# 7. Get metrics
metrics = battery_flex.get_metrics()
print(f"Total throughput: {metrics['total_throughput_kwh']:.1f} kWh")
print(f"Total cost: {metrics['total_cost_eur']:.2f} EUR")
"""

from __future__ import annotations

from typing import Callable, Dict, Any, Optional, Tuple

from flex_model.core.flex_unit import FlexUnit
from flex_model.core.cost_model import CostModel, TimeDependentValue
from flex_model.core.flex_asset import FlexAsset
from flex_model.settings import DT_HOURS


class BatteryUnit(FlexUnit):
    """
    Physical model of a battery energy storage system.

    Implements power limits, efficiency, SOC tracking, and self-discharge.
    """

    def __init__(
        self,
        name: str,
        capacity_kwh: float,
        power_kw: float,
        efficiency: float = 0.95,
        self_discharge_per_hour: float = 0.0,
        soc_min: float = 0.0,
        soc_max: float = 1.0,
        availability_fn: Optional[Callable[[int], float]] = None,
    ) -> None:
        """
        Args:
            name:
                Human-readable identifier for this battery.

            capacity_kwh:
                Total energy capacity [kWh].

            power_kw:
                Maximum charge/discharge power [kW] (nameplate rating).

            efficiency:
                One-way efficiency for charge or discharge (default: 0.95).
                Round-trip efficiency = efficiency^2.
                Applied symmetrically to charging and discharging.

            self_discharge_per_hour:
                Self-discharge rate as fraction of capacity per hour (default: 0.0).
                Example: 0.0001 = 0.01% per hour = ~2.4% per day.

            soc_min:
                Minimum allowed state of charge [0-1] (default: 0.0).
                Battery cannot discharge below this level.

            soc_max:
                Maximum allowed state of charge [0-1] (default: 1.0).
                Battery cannot charge above this level.

            availability_fn:
                Optional time-dependent availability function.
                If provided, scales capacity and power at each time step.
        """
        # Validate parameters BEFORE calling super().__init__
        if not 0.0 < efficiency <= 1.0:
            raise ValueError(f"Efficiency must be in (0, 1], got {efficiency}")
        if not 0.0 <= soc_min < soc_max <= 1.0:
            raise ValueError(f"SOC limits must satisfy 0 <= soc_min < soc_max <= 1, got [{soc_min}, {soc_max}]")

        # Physical parameters (set before super().__init__ to use in properties)
        self._soc_min = soc_min
        self._soc_max = soc_max
        self._power_kw = power_kw
        self._efficiency = efficiency
        self._self_discharge_per_hour = self_discharge_per_hour

        # Initialize private energy variables BEFORE calling super().__init__
        # This prevents property setters from failing when FlexUnit tries to set E_plus/E_minus
        self._E_plus = 0.0
        self._E_minus = 0.0

        super().__init__(
            name=name,
            C_spec=capacity_kwh,
            availability_fn=availability_fn,
        )

    # ------------------------------------------------------------------
    # Properties with validation
    # ------------------------------------------------------------------

    @property
    def soc_min(self) -> float:
        """Minimum allowed state of charge [0-1]."""
        return self._soc_min

    @property
    def soc_max(self) -> float:
        """Maximum allowed state of charge [0-1]."""
        return self._soc_max

    @property
    def power_kw(self) -> float:
        """Nameplate power rating [kW]."""
        return self._power_kw

    @property
    def efficiency(self) -> float:
        """One-way efficiency [0-1]."""
        return self._efficiency

    @property
    def self_discharge_per_hour(self) -> float:
        """Self-discharge rate per hour [0-1]."""
        return self._self_discharge_per_hour

    @property
    def usable_capacity(self) -> float:
        """
        Usable capacity between SOC limits [kWh].

        Returns:
            Capacity available between soc_min and soc_max.
        """
        return self.C_spec * (self.soc_max - self.soc_min)

    def _validate_state(self) -> None:
        """
        Validate internal state consistency.

        Checks:
            1. E_plus >= 0
            2. E_minus >= 0
            3. E_plus <= capacity * soc_max
            4. E_minus <= capacity * (1 - soc_min)
            5. E_plus + E_minus ≈ usable_capacity (energy balance)

        Raises:
            ValueError: If any constraint is violated beyond tolerance.
        """
        tolerance = 1e-6

        # Check non-negativity
        if self._E_plus < -tolerance:
            raise ValueError(f"E_plus={self._E_plus:.6f} is negative")
        if self._E_minus < -tolerance:
            raise ValueError(f"E_minus={self._E_minus:.6f} is negative")

        # Check SOC limits
        max_E_plus = self.C_spec * self.soc_max
        if self._E_plus > max_E_plus + tolerance:
            raise ValueError(
                f"E_plus={self._E_plus:.2f} exceeds maximum {max_E_plus:.2f} "
                f"(soc_max={self.soc_max:.2%})"
            )

        max_E_minus = self.C_spec * (1.0 - self.soc_min)
        if self._E_minus > max_E_minus + tolerance:
            raise ValueError(
                f"E_minus={self._E_minus:.2f} exceeds maximum {max_E_minus:.2f} "
                f"(soc_min={self.soc_min:.2%})"
            )

        # Check energy balance
        energy_sum = self._E_plus + self._E_minus
        expected_sum = self.usable_capacity
        if abs(energy_sum - expected_sum) > tolerance:
            raise ValueError(
                f"Energy balance violated: E_plus + E_minus = {energy_sum:.2f} kWh, "
                f"but usable_capacity = {expected_sum:.2f} kWh "
                f"(difference: {energy_sum - expected_sum:.6f} kWh)"
            )

    @property
    def E_plus(self) -> float:
        """Energy available to discharge [kWh]."""
        return self._E_plus

    @E_plus.setter
    def E_plus(self, value: float) -> None:
        """
        Set E_plus with validation.

        Enforces:
            1. 0 <= E_plus <= capacity * soc_max
            2. E_plus + E_minus = usable_capacity (energy balance)

        Args:
            value: New E_plus value [kWh].

        Raises:
            ValueError: If energy balance or SOC constraints violated.
        """
        # Allow negative values to be clipped to zero (tolerance for numerical errors)
        value = max(0.0, value)

        # Check SOC max constraint
        max_E_plus = self.usable_capacity
        if value > max_E_plus + 1e-6:  # Small tolerance for floating point errors
            raise ValueError(
                f"E_plus={value:.2f} kWh exceeds maximum {max_E_plus:.2f} kWh "
                f"(capacity={self.C_spec:.2f} kWh, soc_max={self.soc_max:.2%})"
            )

        # Clip to max if within tolerance
        value = min(value, max_E_plus)

        # Update E_plus and adjust E_minus to maintain energy balance
        # Energy balance: E_plus + E_minus = usable_capacity
        self._E_plus = value
        self._E_minus = self.usable_capacity - value

        # Ensure E_minus is non-negative (should be guaranteed by constraints)
        if self._E_minus < -1e-6:
            raise ValueError(
                f"Setting E_plus={value:.2f} kWh results in negative E_minus={self._E_minus:.2f} kWh. "
                f"This violates energy balance."
            )
        self._E_minus = max(0.0, self._E_minus)

    @property
    def E_minus(self) -> float:
        """Energy capacity available to charge [kWh]."""
        return self._E_minus

    @E_minus.setter
    def E_minus(self, value: float) -> None:
        """
        Set E_minus with validation.

        Enforces:
            1. 0 <= E_minus <= capacity * (1 - soc_min)
            2. E_plus + E_minus = usable_capacity (energy balance)

        Args:
            value: New E_minus value [kWh].

        Raises:
            ValueError: If energy balance or SOC constraints violated.
        """
        # Allow negative values to be clipped to zero
        value = max(0.0, value)

        # Check SOC min constraint
        max_E_minus = self.usable_capacity
        if value > max_E_minus + 1e-6:
            raise ValueError(
                f"E_minus={value:.2f} kWh exceeds maximum {max_E_minus:.2f} kWh "
                f"(capacity={self.C_spec:.2f} kWh, soc_min={self.soc_min:.2%})"
            )

        # Clip to max if within tolerance
        value = min(value, max_E_minus)

        # Update E_minus and adjust E_plus to maintain energy balance
        self._E_minus = value
        self._E_plus = self.usable_capacity - value

        # Ensure E_plus is non-negative
        if self._E_plus < -1e-6:
            raise ValueError(
                f"Setting E_minus={value:.2f} kWh results in negative E_plus={self._E_plus:.2f} kWh. "
                f"This violates energy balance."
            )
        self._E_plus = max(0.0, self._E_plus)



    def power_limits(self, t: int) -> Tuple[float, float]:
        """
        Return maximum charge/discharge power at time t.

        Power limits are determined by:
            1. Nameplate power rating (self.power_kw)
            2. Available energy capacity (E_minus for charging, E_plus for discharging)
            3. Availability factor (if configured)

        Returns:
            (P_draw_max, P_inject_max) in [kW].
            - P_draw_max: Maximum power to draw (charge battery)
            - P_inject_max: Maximum power to inject (discharge battery)
        """
        # Base power limit from nameplate rating and availability
        P_rated = self.power_kw * self.availability(t)

        # Charge limit: limited by available capacity (E_minus)
        # Can't charge more than remaining capacity allows
        # P_max = E_available / dt (energy constraint on power)
        P_draw_max = min(P_rated, self.E_minus / DT_HOURS)

        # Discharge limit: limited by stored energy (E_plus)
        # Can't discharge more than stored energy allows
        # P_max = E_available / dt (energy constraint on power)
        P_inject_max = min(P_rated, self.E_plus / DT_HOURS)

        return P_draw_max, P_inject_max

    def soc(self) -> float:
        """
        Return current state of charge [0-1].

        SOC is defined as the ratio of stored energy (E_plus) to total capacity.
        """
        capacity = self.capacity(t=0)  # Use t=0 as current capacity
        if capacity > 0:
            return self.soc_min + self.E_plus / capacity
        return 0.0

    def update_state(
        self,
        t: int,
        P_grid_import: float,
        P_grid_export: float,
        P_loss: float = 0.0,
        P_gain: float = 0.0,
        n_timesteps: int = 1,
    ) -> None:
        """
        Update battery state with efficiency and self-discharge.

        For a battery, the energy headroom interpretation is:
            - E_plus: Stored energy available to export (discharge) [kWh]
            - E_minus: Remaining capacity available to import (charge) [kWh]

        Charging (P_grid_import > 0): Increases E_plus, decreases E_minus
        Discharging (P_grid_export > 0): Decreases E_plus, increases E_minus

        Args:
            t: Time index.
            P_grid_import: Charging power [kW] (import from grid).
            P_grid_export: Discharging power [kW] (export to grid).
            P_loss: Additional losses [kW] (added to self-discharge).
            P_gain: Additional gains [kW] (rarely used for batteries).
            n_timesteps: Number of time steps.
        """
        # Clip commands to be non-negative
        P_import = max(0.0, P_grid_import)
        P_export = max(0.0, P_grid_export)

        self.P_grid_import = P_import
        self.P_grid_export = P_export

        # Calculate self-discharge loss for this time step
        capacity = self.capacity(t)
        P_self_discharge = capacity * self.self_discharge_per_hour  # [kW]

        # Convert powers to energies
        delta_t = n_timesteps * FlexAsset.dt_hours
        E_charge_gross = P_import * delta_t  # Energy imported from grid (charging)
        E_discharge_gross = P_export * delta_t  # Energy exported to grid (discharging)
        E_self_discharge = P_self_discharge * delta_t
        E_loss_external = P_loss * delta_t
        E_gain_external = P_gain * delta_t

        # Apply efficiency
        E_charge_net = E_charge_gross * self.efficiency  # Energy actually stored
        E_discharge_net = E_discharge_gross / self.efficiency  # Energy taken from storage

        # Update stored energy (E_plus)
        # Increases with charging, decreases with discharging and losses
        delta_E_plus = E_charge_net - E_discharge_net - E_self_discharge - E_loss_external + E_gain_external
        new_E_plus = self._E_plus + delta_E_plus
        new_E_plus = max(0.0, min(new_E_plus, capacity))

        # Update available capacity (E_minus)
        # Decreases with charging, increases with discharging and losses
        delta_E_minus = E_discharge_net - E_charge_net + E_self_discharge + E_loss_external - E_gain_external
        new_E_minus = self._E_minus + delta_E_minus
        new_E_minus = max(0.0, min(new_E_minus, capacity))

        # Apply updates directly to private variables (avoid property setter interference)
        self._E_plus = new_E_plus
        self._E_minus = new_E_minus

        # Validate state consistency
        self._validate_state()


class BatteryCostModel(CostModel):
    """
    Economic model for battery energy storage.

    Captures investment cost, degradation, and energy arbitrage.
    """

    def __init__(
        self,
        name: str,
        c_inv: float,
        n_lifetime: float,
        c_fix: float = 0.0,
        p_int: TimeDependentValue = 0.0,
        p_E_buy: TimeDependentValue = 0.0,
        p_E_sell: TimeDependentValue = 0.0,
    ) -> None:
        """
        Args:
            name:
                Human-readable identifier for this cost model.

            c_inv:
                Specific investment cost [CHF/kWh].
                Typical range: 300-800 CHF/kWh depending on technology and scale.

            n_lifetime:
                Economic lifetime [years].
                Typical range: 10-20 years for stationary batteries.

            c_fix:
                Fixed annual O&M cost [CHF/a].
                Typically small for batteries: 0-2% of CAPEX per year.

            p_int:
                Internal utilization cost [CHF/kWh of throughput].
                Represents variable O&M costs proportional to usage (cycling wear, etc.).
                This is separate from degradation, which is captured via investment cost
                amortization (c_inv / n_lifetime).
                Example: 0.01-0.05 CHF/kWh for maintenance triggered by cycling

            p_E_buy:
                Energy import price [CHF/kWh] when charging.
                Can be time-varying to capture spot price fluctuations.

            p_E_sell:
                Energy export price [CHF/kWh] when discharging.
                Typically lower than p_E_buy due to market spreads.
        """
        super().__init__(
            name=name,
            c_inv=c_inv,
            n_lifetime=n_lifetime,
            c_fix=c_fix,
            p_int=p_int,
            C_event=0.0,  # Batteries typically have no start-up costs
            p_E_buy=p_E_buy,
            p_E_sell=p_E_sell,
            p_P=0.0,  # Reserve capacity payments can be added via p_P
            p_CO2=0.0,  # CO2 costs handled externally
        )

    def step_cost(
        self,
        t: int,
        flex_state: Dict[str, Any],
        activation: Dict[str, Any],
    ) -> float:
        """
        Calculate cost [CHF] for one time step of battery operation.

        Args:
            t:
                Time index.

            flex_state:
                Physical state: {'soc': float, 'E_plus': float, 'E_minus': float}

            activation:
                Operation: {'P_grid_import': float, 'P_grid_export': float, 'dt_hours': float}

        Returns:
            Total cost [CHF] for this time step.
            Positive = net cost, Negative = net revenue.

        Cost components:
            1. Internal usage: throughput * p_int(t)
            2. Energy cost: E_net * p_E_buy(t) or E_net * p_E_sell(t)
        """
        # Validate and extract activation parameters (REQUIRED - fail fast if missing)
        self._validate_activation_keys(activation, {'P_grid_import', 'P_grid_export', 'dt_hours'})

        P_import = activation['P_grid_import']
        P_export = activation['P_grid_export']
        dt_hours = activation['dt_hours']

        # 1. Internal utilization cost (proportional to throughput)
        throughput_kwh = (P_import + P_export) * dt_hours
        cost_usage = throughput_kwh * self.p_int(t)

        # 2. Energy cost (buy when importing, sell when exporting)
        E_net = (P_import - P_export) * dt_hours  # Positive = net import (charging)

        if E_net > 0:
            # Net import (charging): buying energy
            cost_energy = E_net * self.p_E_buy(t)
        else:
            # Net export (discharging): selling energy (revenue is negative cost)
            cost_energy = E_net * self.p_E_sell(t)  # E_net is negative, so this is negative

        return cost_usage + cost_energy

    # Note: annualized_investment() is inherited from CostModel base class
    # Note: total_cost() is inherited from CostModel base class


class BatteryFlex(FlexAsset):
    """
    Operational composition of BatteryUnit and BatteryCostModel.

    Provides evaluate_operation() and execute_operation() interface for optimization.
    """

    def __init__(self, unit: BatteryUnit, cost_model: BatteryCostModel, name: str = None) -> None:
        """
        Args:
            unit:
                BatteryUnit providing physical behavior.

            cost_model:
                BatteryCostModel providing economic evaluation.

            name:
                Optional name for this asset. If None, uses unit.name.
        """
        super().__init__(unit=unit, name=name)
        self.cost_model = cost_model

    def evaluate_operation(
        self,
        t: int,
        P_grid_import: float,
        P_grid_export: float,
        n_timesteps: int = 1,
    ) -> Dict[str, Any]:
        """
        Evaluate proposed battery operation without executing it.

        Checks:
            1. Power limits: import/export within rated capacity
            2. Mutual exclusivity: Not charging and discharging simultaneously
            3. SOC limits: Operation doesn't violate soc_min or soc_max

        Args:
            t: Time index.
            P_grid_import: Proposed charging power [kW] (import from grid).
            P_grid_export: Proposed discharging power [kW] (export to grid).
            n_timesteps: Number of time steps.

        Returns:
            {
                'feasible': bool,
                'cost': float [CHF],
                'violations': List[str],
                'soc': float [0-1],  # SOC after operation
                'throughput': float [kWh],  # Energy throughput
            }
        """
        violations = []

        # Get current power limits
        P_import_max, P_export_max = self.unit.power_limits(t)

        # Check power limits
        if P_grid_import > P_import_max + 1e-6:  # Small tolerance for numerical errors
            violations.append(f"P_grid_import={P_grid_import:.2f} kW exceeds limit {P_import_max:.2f} kW")

        if P_grid_export > P_export_max + 1e-6:
            violations.append(f"P_grid_export={P_grid_export:.2f} kW exceeds limit {P_export_max:.2f} kW")

        # Check mutual exclusivity (can't charge and discharge simultaneously)
        if P_grid_import > 1e-6 and P_grid_export > 1e-6:
            violations.append("Cannot import and export simultaneously")

        # Calculate expected SOC after operation
        # Importing (charging): E_plus increases by P_import * dt * eff
        # Exporting (discharging): E_plus decreases by P_export * dt / eff
        delta_t = n_timesteps * FlexAsset.dt_hours
        current_soc = self.unit.soc()
        delta_E = (P_grid_import * delta_t * self.unit.efficiency -
                   P_grid_export * delta_t / self.unit.efficiency)
        soc_after = current_soc + delta_E / self.unit.capacity(t)

        # Check SOC limits
        if soc_after < self.unit.soc_min - 1e-6:
            violations.append(f"SOC would drop to {soc_after:.1%}, below minimum {self.unit.soc_min:.1%}")

        if soc_after > self.unit.soc_max + 1e-6:
            violations.append(f"SOC would rise to {soc_after:.1%}, above maximum {self.unit.soc_max:.1%}")

        feasible = len(violations) == 0

        # Calculate cost if feasible
        if feasible:
            flex_state = {
                'soc': current_soc,
                'E_plus': self.unit.E_plus,
                'E_minus': self.unit.E_minus,
            }
            activation = {
                'P_grid_import': P_grid_import,
                'P_grid_export': P_grid_export,
                'dt_hours': delta_t,
            }
            cost = self.cost_model.step_cost(t, flex_state, activation)
        else:
            cost = 0.0  # Infeasible operations have zero cost (handled by constraints)

        return {
            'feasible': feasible,
            'cost': cost,
            'violations': violations,
            'soc': soc_after,
            'throughput': (P_grid_import + P_grid_export) * delta_t,
        }

    def execute_operation(
        self,
        t: int,
        P_grid_import: float,
        P_grid_export: float,
        n_timesteps: int = 1,
    ) -> None:
        """
        Execute battery operation, updating physical state and tracking metrics.

        Args:
            t: Time index.
            P_grid_import: Charging power [kW] to execute (import from grid).
            P_grid_export: Discharging power [kW] to execute (export to grid).
            n_timesteps: Number of time steps.

        Notes:
            - Caller must ensure operation is feasible before calling.
            - Updates unit state via unit.update_state().
            - Tracks metrics: throughput, cost, activations.
        """
        # Update physical state
        self.unit.update_state(
            t=t,
            P_grid_import=P_grid_import,
            P_grid_export=P_grid_export,
            n_timesteps=n_timesteps,
        )

        # Calculate total time delta
        delta_t = n_timesteps * FlexAsset.dt_hours

        # Calculate cost for tracking
        flex_state = {
            'soc': self.unit.soc(),
            'E_plus': self.unit.E_plus,
            'E_minus': self.unit.E_minus,
        }
        activation = {
            'P_grid_import': P_grid_import,
            'P_grid_export': P_grid_export,
            'dt_hours': delta_t,
        }
        cost = self.cost_model.step_cost(t, flex_state, activation)

        # Update tracking metrics
        throughput = (P_grid_import + P_grid_export) * delta_t
        self._total_throughput_kwh += throughput
        self._total_cost_eur += cost

        if P_grid_import > 1e-6 or P_grid_export > 1e-6:
            self._num_activations += 1

    def get_metrics(self) -> Dict[str, Any]:
        """
        Return operational metrics including battery-specific values.

        Returns:
            Dictionary with standard metrics plus:
                - 'current_soc': Current state of charge [0-1]
                - 'current_energy_stored': Current stored energy [kWh]
        """
        metrics = super().get_metrics()
        metrics['current_soc'] = self.unit.soc()
        metrics['current_energy_stored'] = self.unit.E_plus
        return metrics

