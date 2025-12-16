"""
CostModel: Economic modeling of flexibility assets and market options.

This module provides the base class for modeling the economic behavior of flexibility
assets and market options. CostModel is designed to capture ONLY the economic aspects,
independent of physical/technical constraints (which are handled by FlexUnit).

DESIGN PHILOSOPHY
-----------------
Separation of concerns:
    - FlexUnit handles PHYSICS: power limits, energy capacity, ramp rates, efficiency
    - CostModel handles ECONOMICS: prices, degradation costs, revenues
    - FlexAsset handles OPERATIONS: combines physics + economics for decision-making

Physics-agnostic:
    - CostModel does not depend on FlexUnit internals
    - Communication happens through agreed data structures (flex_state, activation)
    - This allows the same CostModel framework to work with any FlexUnit type

Horizon-agnostic:
    - Time index t is abstract (integer), no assumptions about actual timestamps
    - No dependency on pandas, numpy, or specific time series libraries
    - Works with any time resolution (15-min, hourly, daily, etc.)

Minimal parameter set:
    - All economic aspects captured by generic price signals
    - Time-dependent prices can be constant, lookup tables, or dynamic functions
    - Flexible enough to model any value stream without hard-coding market designs

TIME-DEPENDENT PARAMETERS
-------------------------
All price signals are functions of time index t (integer). They can be specified as:

1. Scalar (constant over time):
    p_E_buy = 0.25  # constant 0.25 CHF/kWh at all times

2. Dictionary (lookup table):
    p_E_buy = {0: 0.20, 1: 0.25, 2: 0.30, ...}  # varies by time step

3. Callable (dynamic function):
    import math
    p_E_buy = lambda t: 0.20 + 0.05 * math.sin(t * 2 * math.pi / 96)  # daily pattern

All are automatically normalized to callables internally for consistent access.

ANALOGY TO FLEXUNIT
--------------------
FlexUnit provides time-dependent physical properties:
    - availability(t), power_limits(t), capacity(t)

CostModel provides time-dependent economic properties:
    - p_E_buy(t), p_E_sell(t), p_int(t), p_P(t), p_CO2(t)

Both use the same design pattern: methods that take time index t and return floats.

GENERIC ECONOMIC PARAMETER SET
-------------------------------
Structural (time-independent):
    - c_inv: Specific investment cost [CHF/kW] or [CHF/kWh]
    - n_lifetime: Economic lifetime [years]
    - c_fix: Fixed annual O&M cost [CHF/a]

Internal usage-based cost (time-dependent):
    - p_int(t): Internal marginal cost [CHF/kWh] - degradation, fuel, auxiliary energy
    - C_event(t): Event-based cost [CHF/event] - start-up, state-change costs

External price signals (time-dependent):
    - p_E_buy(t): Energy import price [CHF/kWh] - wholesale + network + taxes + imbalance
    - p_E_sell(t): Energy export price [CHF/kWh] - feed-in tariff or wholesale
    - p_P(t): Capacity/power price [CHF/kW] - reserve payments, network tariffs
    - p_CO2(t): Carbon price [CHF/tCO2] - emissions cost or savings

The asymmetry between p_E_buy and p_E_sell naturally captures:
    - Different buy/sell tariffs
    - Imbalance price asymmetry
    - Transaction costs and market inefficiencies

TYPICAL WORKFLOW
----------------
1. Create a CostModel subclass instance:
    cost = BatteryCostModel(
        name="battery_cost",
        c_inv=500,  # CHF/kWh
        n_lifetime=10,
        p_int=0.05,  # CHF/kWh degradation
        p_E_buy={0: 0.20, 1: 0.25, ...},  # time-varying price
    )

2. Query time-dependent prices:
    energy_price = cost.p_E_buy(t=10)
    internal_cost = cost.p_int(t=10)

3. Evaluate step cost:
    flex_state = {'soc': 0.7, 'E_plus': 30.0, 'E_minus': 70.0}
    activation = {'P_grid_import': 20.0, 'P_grid_export': 0.0, 'dt_hours': 0.25}
    cost_t = cost.step_cost(t=10, flex_state=flex_state, activation=activation)

4. Aggregate total cost:
    total = cost.total_cost(time_indices, flex_trajectory, activation_trajectory)

5. Calculate annualized investment:
    capex_annual = cost.annualized_investment(capacity=100.0, discount_rate=0.05)

APPLICABLE TO BOTH PHYSICAL ASSETS AND MARKET OPTIONS
------------------------------------------------------
Physical assets (combined with FlexUnit):
    - BatteryUnit + BatteryCostModel
    - HeatPumpUnit + HeatPumpCostModel
    - PVUnit + PVCostModel

Pure market options (no FlexUnit needed):
    - Imbalance settlement (buy/sell at imbalance prices)
    - Reserve capacity products (FCR, aFRR, mFRR)
    - Intraday trading (arbitrage between markets)

IMPLEMENTATION REQUIREMENTS FOR SUBCLASSES
-------------------------------------------
Abstract method that MUST be implemented:
    - step_cost(t, flex_state, activation): Calculate cost [CHF] for one time step

The step_cost() implementation should:
    1. Extract relevant info from flex_state and activation
    2. Query time-dependent prices: self.p_E_buy(t), self.p_int(t), etc.
    3. Calculate cost components:
        - Energy cost: E_net * p_E_buy or p_E_sell
        - Internal cost: throughput * p_int
        - Capacity revenue: P_offered * p_P
        - Event cost: C_event (if state change occurred)
        - CO2 cost: emissions * p_CO2
    4. Return total cost (positive = cost, negative = revenue)

EXAMPLE: Battery Cost Model
----------------------------
    class BatteryCostModel(CostModel):
        def step_cost(self, t, flex_state, activation):
            P_import = activation['P_grid_import']
            P_export = activation['P_grid_export']
            dt_hours = activation['dt_hours']

            # Internal degradation cost
            throughput = (P_import + P_export) * dt_hours
            cost_internal = throughput * self.p_int(t)

            # External energy cost
            E_net = (P_import - P_export) * dt_hours
            if E_net > 0:  # net import (buying)
                cost_energy = E_net * self.p_E_buy(t)
            else:  # net export (selling)
                cost_energy = E_net * self.p_E_sell(t)  # negative = revenue

            return cost_internal + cost_energy
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Sequence, Union


# Type alias for flexible time-dependent parameters
TimeDependentValue = Union[float, Dict[int, float], Callable[[int], float]]


class CostModel(ABC):
    """
    Base class for economic modeling of flexibility assets and market options.

    Subclasses must implement: step_cost(t, flex_state, activation).
    """

    def __init__(
        self,
        name: str,
        c_inv: float,
        n_lifetime: float,
        c_fix: float = 0.0,
        p_int: TimeDependentValue = 0.0,
        C_event: TimeDependentValue = 0.0,
        p_E_buy: TimeDependentValue = 0.0,
        p_E_sell: TimeDependentValue = 0.0,
        p_P: TimeDependentValue = 0.0,
        p_CO2: TimeDependentValue = 0.0,
    ) -> None:
        """
        Args:
            name:
                Human-readable identifier for this cost model.

            c_inv:
                Specific investment cost [CHF/kW] or [CHF/kWh], depending on asset type.

            n_lifetime:
                Economic lifetime [years].

            c_fix:
                Fixed annual operation & maintenance cost [CHF/a].

            p_int:
                Internal marginal operational cost [CHF/kWh] at time t.
                Aggregates degradation, fuel, auxiliary energy, wear.
                Can be scalar (constant), dict, or callable.

            C_event:
                Event-based cost [CHF/event] at time t.
                Start-up, state-change, or switching penalty.
                Can be scalar (constant), dict, or callable.

            p_E_buy:
                Effective energy IMPORT price [CHF/kWh] at time t.
                Wholesale + network charges + taxes + imbalance.
                Can be scalar (constant), dict, or callable.

            p_E_sell:
                Effective energy EXPORT price [CHF/kWh] at time t.
                Feed-in tariff or wholesale price.
                Can be scalar (constant), dict, or callable.

            p_P:
                Capacity / power price [CHF/kW] at time t.
                Reserve payments, power-based tariffs.
                Can be scalar (constant), dict, or callable.

            p_CO2:
                Carbon price [CHF/tCO2] at time t.
                Can be scalar (constant), dict, or callable.
        """
        self.name = name

        # Structural parameters (time-independent)
        self.c_inv = c_inv
        self.n_lifetime = n_lifetime
        self.c_fix = c_fix

        # Normalize all time-dependent parameters to callables
        self._p_int = self._normalize_time_param(p_int)
        self._C_event = self._normalize_time_param(C_event)
        self._p_E_buy = self._normalize_time_param(p_E_buy)
        self._p_E_sell = self._normalize_time_param(p_E_sell)
        self._p_P = self._normalize_time_param(p_P)
        self._p_CO2 = self._normalize_time_param(p_CO2)

    # ------------------------------------------------------------------
    # Internal helper: normalize time-dependent parameters
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_time_param(param: TimeDependentValue) -> Callable[[int], float]:
        """
        Convert a time-dependent parameter to a callable.

        Args:
            param: Can be float (constant), dict (lookup table), or callable.

        Returns:
            A function that maps time index t -> float.
        """
        if callable(param):
            return param
        elif isinstance(param, dict):
            # Return a lookup function with default 0.0 for missing keys
            return lambda t: param.get(t, 0.0)
        else:
            # Scalar value (constant over time)
            return lambda t: float(param)

    @staticmethod
    def _validate_activation_keys(activation: Dict, required_keys: set) -> None:
        """
        Validate that activation dict contains all required keys.

        Args:
            activation:
                Dictionary with operational decision parameters.

            required_keys:
                Set of keys that must be present in activation dict.

        Raises:
            ValueError: If any required keys are missing from activation dict.

        Example:
            >>> self._validate_activation_keys(activation, {'P_grid_import', 'P_grid_export', 'dt_hours'})
        """
        if not isinstance(activation, dict):
            raise TypeError(f"activation must be a dict, got {type(activation).__name__}")

        if not required_keys.issubset(activation.keys()):
            missing = required_keys - activation.keys()
            raise ValueError(
                f"Activation dict missing required keys: {missing}. "
                f"Got keys: {set(activation.keys())}"
            )

    # ------------------------------------------------------------------
    # Public time-dependent price accessors
    # ------------------------------------------------------------------

    def p_int(self, t: int) -> float:
        """Internal marginal operational cost [CHF/kWh] at time t."""
        return self._p_int(t)

    def C_event(self, t: int) -> float:
        """Event-based cost [CHF/event] at time t."""
        return self._C_event(t)

    def p_E_buy(self, t: int) -> float:
        """Effective energy import price [CHF/kWh] at time t."""
        return self._p_E_buy(t)

    def p_E_sell(self, t: int) -> float:
        """Effective energy export price [CHF/kWh] at time t."""
        return self._p_E_sell(t)

    def p_P(self, t: int) -> float:
        """Capacity / power price [CHF/kW] at time t."""
        return self._p_P(t)

    def p_CO2(self, t: int) -> float:
        """Carbon price [CHF/tCO2] at time t."""
        return self._p_CO2(t)

    # ------------------------------------------------------------------
    # Investment & fixed cost helpers
    # ------------------------------------------------------------------

    def annualized_investment(
        self,
        capacity: float,
        annuity_factor: Optional[float] = None,
        discount_rate: float = 0.05,
    ) -> float:
        """
        Calculate annualized investment cost (CAPEX) for given capacity.

        Args:
            capacity:
                Installed capacity [kW] or [kWh], consistent with c_inv units.

            annuity_factor:
                Annuity factor for capital recovery. If None, calculated from
                discount_rate and n_lifetime using: a = r*(1+r)^n / ((1+r)^n - 1)

            discount_rate:
                Discount rate (default: 0.05 = 5% per year).
                Only used if annuity_factor is None.

        Returns:
            Annualized investment cost [CHF/a].
        """
        if annuity_factor is None:
            # Calculate annuity factor: a = r*(1+r)^n / ((1+r)^n - 1)
            r = discount_rate
            n = self.n_lifetime
            if r == 0:
                # Special case: no discounting
                annuity_factor = 1.0 / n
            else:
                annuity_factor = (r * (1 + r) ** n) / ((1 + r) ** n - 1)

        total_investment = self.c_inv * capacity
        return total_investment * annuity_factor

    def annual_fixed_cost(self) -> float:
        """Return fixed annual O&M cost [CHF/a]."""
        return self.c_fix

    # ------------------------------------------------------------------
    # Abstract per-step cost method
    # ------------------------------------------------------------------

    @abstractmethod
    def step_cost(
        self,
        t: int,
        flex_state: object,
        activation: object,
    ) -> float:
        """
        Compute the cost [CHF] incurred by this asset in time step t.

        Args:
            t:
                Time index (integer).

            flex_state:
                Arbitrary object describing the physical state of the FlexUnit at time t.
                Structure is agreed between FlexUnit and CostModel implementations.

            activation:
                Arbitrary object describing the operational decision at time t.
                Structure is agreed between the optimizer and CostModel.

        Returns:
            Cost incurred in this time step [CHF].
            Positive = cost (money spent), Negative = revenue (money earned).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Aggregated cost helper
    # ------------------------------------------------------------------

    def total_cost(
        self,
        time_indices: Sequence[int],
        flex_trajectory: Sequence[object],
        activation_trajectory: Sequence[object],
    ) -> float:
        """
        Aggregate step_cost over a time horizon.

        Args:
            time_indices:
                Sequence of time indices [t_0, t_1, ..., t_N].

            flex_trajectory:
                Sequence of flex_state objects [state_0, state_1, ..., state_N],
                one per time step.

            activation_trajectory:
                Sequence of activation objects [act_0, act_1, ..., act_N],
                one per time step.

        Returns:
            Total cost over the horizon [CHF].
        """
        if len(time_indices) != len(flex_trajectory) or len(time_indices) != len(activation_trajectory):
            raise ValueError(
                f"Length mismatch: time_indices={len(time_indices)}, "
                f"flex_trajectory={len(flex_trajectory)}, "
                f"activation_trajectory={len(activation_trajectory)}"
            )

        total = 0.0
        for t, flex_state, activation in zip(time_indices, flex_trajectory, activation_trajectory):
            total += self.step_cost(t, flex_state, activation)
        return total