"""
BalancingMarket: Imbalance settlement with TSO (e.g., Swissgrid).

This module implements imbalance settlement as a FlexAsset without physical
constraints. It represents the cost of deviating from your scheduled position
and settling imbalances at real-time balancing energy prices.

CONTEXT
-------
When you have an energy imbalance (deviation from schedule), the TSO settles
this imbalance at balancing energy prices (Ausgleichsenergie):
    - Positive imbalance: You consumed more than scheduled → pay import price
    - Negative imbalance: You consumed less than scheduled → receive export price

In Switzerland, Swissgrid publishes these as "Preise für Ausgleichsenergie".

USE CASE
--------
Compare battery investment versus accepting imbalances:
    - Battery scenario: Invest in storage to avoid imbalances
    - Imbalance scenario: No investment, settle all imbalances with TSO

The BalancingMarket asset represents the imbalance settlement baseline.

DESIGN
------
Unlike physical flexibility assets, imbalance settlement has:
    - NO capacity limits (TSO always settles imbalances)
    - NO efficiency losses (financial settlement only)
    - NO degradation costs (no physical asset)
    - NO state tracking (stateless transactions)

Cost structure:
    - p_E_buy(t): Imbalance price for positive deviations [CHF/kWh]
    - p_E_sell(t): Imbalance price for negative deviations [CHF/kWh]
    - Typically asymmetric and volatile: buying often > selling

TYPICAL WORKFLOW
----------------
1. Create market cost model:
    market_cost = BalancingMarketCost(
        name="balancing_market",
        p_E_buy={0: 0.25, 1: 0.30, ...},  # Time-varying buy prices
        p_E_sell={0: 0.15, 1: 0.20, ...}, # Time-varying sell prices
    )

2. Create market flex asset:
    market = BalancingMarketFlex(cost_model=market_cost)

3. Evaluate operation (always feasible):
    result = market.evaluate_operation(
        t=10,
        dt_hours=0.25,
        P_grid_import=50.0,  # Buy 50 kW from market
        P_grid_export=0.0,
    )
    # result['feasible'] is always True
    # result['cost'] is the market procurement cost

4. Execute operation:
    market.execute_operation(t=10, dt_hours=0.25, P_grid_import=50.0, P_grid_export=0.0)

COMPARISON EXAMPLE
------------------
# Scenario: Cover 50 kW load for 15 minutes

# Option 1: Battery (if available)
battery_result = battery.evaluate_operation(t=10, dt_hours=0.25,
                                           P_grid_import=0, P_grid_export=50)
if battery_result['feasible']:
    battery_cost = battery_result['cost']  # Degradation + opportunity cost
else:
    battery_cost = float('inf')

# Option 2: Market
market_result = market.evaluate_operation(t=10, dt_hours=0.25,
                                          P_grid_import=50, P_grid_export=0)
market_cost = market_result['cost']  # Direct market price

# Optimizer chooses cheaper option
"""

from __future__ import annotations

from typing import Any, Dict

from flex_model.core.cost_model import CostModel, TimeDependentValue
from flex_model.core.flex_asset import FlexAsset
from flex_model.optimization import LinearModel


class BalancingMarketCost(CostModel):
    """
    Cost model for imbalance settlement with TSO.

    Represents the cost of settling energy imbalances at balancing energy prices
    (Ausgleichsenergie). No investment costs, no degradation - just real-time
    imbalance prices from the TSO (e.g., Swissgrid).
    """

    def __init__(
        self,
        name: str,
        p_E_buy: TimeDependentValue = 0.0,
        p_E_sell: TimeDependentValue = 0.0,
    ) -> None:
        """
        Args:
            name:
                Human-readable identifier for this market.

            p_E_buy:
                Imbalance price for positive deviations [CHF/kWh] at time t.
                (You consumed more than scheduled → must pay)
                Can be scalar (constant), dict (time-varying), or callable.

            p_E_sell:
                Imbalance price for negative deviations [CHF/kWh] at time t.
                (You consumed less than scheduled → receive payment)
                Can be scalar (constant), dict (time-varying), or callable.
                Typically lower than p_E_buy (asymmetric settlement).
        """
        # Market has no investment cost or lifetime
        super().__init__(
            name=name,
            c_inv=0.0,
            n_lifetime=1.0,  # Arbitrary, not used
            c_fix=0.0,
            p_int=0.0,  # No internal/degradation cost
            p_E_buy=p_E_buy,
            p_E_sell=p_E_sell,
        )

    def step_cost(
        self,
        t: int,
        flex_state: object,
        activation: Dict[str, float],
    ) -> float:
        """
        Calculate imbalance settlement cost for one time step.

        Args:
            t:
                Time index (integer).

            flex_state:
                Not used for imbalance settlement (no physical state).

            activation:
                Dict with keys:
                    - 'P_grid_import': Positive imbalance power [kW]
                    - 'P_grid_export': Negative imbalance power [kW]
                    - 'dt_hours': Time step duration [h]

        Returns:
            Settlement cost [CHF]. Positive = cost, Negative = revenue.
        """
        # Validate required keys
        self._validate_activation_keys(activation, {'P_grid_import', 'P_grid_export', 'dt_hours'})

        # Extract activation parameters
        P_import = activation['P_grid_import']
        P_export = activation['P_grid_export']
        dt_hours = activation['dt_hours']

        # Calculate energies
        E_import = P_import * dt_hours  # Energy bought [kWh]
        E_export = P_export * dt_hours  # Energy sold [kWh]

        # Calculate costs
        cost_import = E_import * self.p_E_buy(t)  # Cost of buying
        revenue_export = E_export * self.p_E_sell(t)  # Revenue from selling

        # Net cost (positive = cost, negative = revenue)
        return cost_import - revenue_export


class BalancingMarketFlex(FlexAsset):
    """
    FlexAsset for imbalance settlement without physical constraints.

    Represents settling energy imbalances with the TSO at real-time balancing
    prices. Always feasible (TSO always settles) with no physical state.
    """

    def __init__(
        self,
        cost_model: BalancingMarketCost,
        name: str = None,
    ) -> None:
        """
        Args:
            cost_model:
                BalancingMarketCost instance defining market prices.

            name:
                Optional name for this market. Defaults to cost_model.name.
        """
        # Market has no physical unit, but FlexAsset expects one
        # We pass None and handle it specially
        self.cost_model = cost_model
        self.name = name or cost_model.name

        # Operational tracking (inherited from FlexAsset pattern)
        self._total_throughput_kwh: float = 0.0
        self._total_cost_eur: float = 0.0
        self._num_activations: int = 0

    def evaluate_operation(
        self,
        t: int,
        dt_hours: float,
        P_grid_import: float,
        P_grid_export: float,
    ) -> Dict[str, Any]:
        """
        Evaluate imbalance settlement operation.

        Imbalance settlement is always feasible (TSO always settles).

        Args:
            t: Time index (integer).
            dt_hours: Duration of the time step [h].
            P_grid_import: Positive imbalance power [kW].
            P_grid_export: Negative imbalance power [kW].

        Returns:
            Dictionary:
                {
                    'feasible': True (always),
                    'cost': float [CHF],
                    'violations': [] (always empty),
                }
        """
        # Prepare activation dict
        activation = {
            'P_grid_import': P_grid_import,
            'P_grid_export': P_grid_export,
            'dt_hours': dt_hours,
        }

        # Calculate cost (no physical state for market)
        cost = self.cost_model.step_cost(t=t, flex_state=None, activation=activation)

        return {
            'feasible': True,  # Market always available
            'cost': cost,
            'violations': [],
            'E_import': P_grid_import * dt_hours,
            'E_export': P_grid_export * dt_hours,
        }

    def execute_operation(
        self,
        t: int,
        dt_hours: float,
        P_grid_import: float,
        P_grid_export: float,
    ) -> None:
        """
        Execute imbalance settlement operation.

        Updates tracking metrics but no physical state (settlement is stateless).

        Args:
            t: Time index (integer).
            dt_hours: Duration of the time step [h].
            P_grid_import: Positive imbalance power [kW].
            P_grid_export: Negative imbalance power [kW].
        """
        # Prepare activation dict
        activation = {
            'P_grid_import': P_grid_import,
            'P_grid_export': P_grid_export,
            'dt_hours': dt_hours,
        }

        # Calculate cost
        cost = self.cost_model.step_cost(t=t, flex_state=None, activation=activation)

        # Update tracking
        throughput = (P_grid_import + P_grid_export) * dt_hours
        self._total_throughput_kwh += throughput
        self._total_cost_eur += cost
        self._num_activations += 1

    def power_limits(self, t: int) -> tuple[float, float]:
        """
        Imbalance settlement has no power limits (TSO always settles).

        Returns:
            (P_import_max, P_export_max) = (inf, inf)
        """
        return float('inf'), float('inf')

    def reset(self, E_plus_init: float = 0.0, E_minus_init: float = 0.0) -> None:
        """
        Reset operational tracking.

        Note: E_plus_init and E_minus_init are ignored (settlement has no state).
        They're kept for interface compatibility with FlexAsset.
        """
        self._total_throughput_kwh = 0.0
        self._total_cost_eur = 0.0
        self._num_activations = 0

    def get_metrics(self) -> Dict[str, Any]:
        """
        Return operational metrics tracked during simulation.

        Returns:
            Dictionary with keys:
                - 'total_throughput_kwh': Total imbalance energy settled [kWh]
                - 'total_cost_eur': Total settlement cost [EUR]
                - 'num_activations': Number of settlement periods
        """
        return {
            'total_throughput_kwh': self._total_throughput_kwh,
            'total_cost_eur': self._total_cost_eur,
            'num_activations': self._num_activations,
        }

    def get_linear_model(self, n_timesteps: int, dt_hours: float) -> LinearModel:
        """
        Convert market settlement to linear optimization model.

        Creates decision variables for market import/export at each timestep
        with associated costs. No internal constraints (market is unlimited).

        Args:
            n_timesteps: Number of timesteps in optimization horizon.
            dt_hours: Duration of each timestep [h].

        Returns:
            LinearModel representing this market asset.
        """
        import numpy as np

        # Decision variables: P_import[t], P_export[t] for each timestep
        # Total: 2 * n_timesteps variables
        n_vars = 2 * n_timesteps

        # Variable names
        var_names = []
        for t in range(n_timesteps):
            var_names.append(f"{self.name}_P_import_{t}")
        for t in range(n_timesteps):
            var_names.append(f"{self.name}_P_export_{t}")

        # Variable bounds: all >= 0, no upper limit (market unlimited)
        var_bounds = [(0.0, None) for _ in range(n_vars)]

        # Cost coefficients
        # Cost = sum_t [p_buy[t] * P_import[t] * dt - p_sell[t] * P_export[t] * dt]
        cost_coefficients = np.zeros(n_vars)
        for t in range(n_timesteps):
            cost_coefficients[t] = self.cost_model.p_E_buy(t) * dt_hours  # Import cost
            cost_coefficients[n_timesteps + t] = -self.cost_model.p_E_sell(t) * dt_hours  # Export revenue (negative cost)

        # No internal constraints (market is unlimited)
        A_eq = None
        b_eq = None
        A_ub = None
        b_ub = None

        # Power mapping for energy balance
        # net_power[t] = P_import[t] - P_export[t]
        # (importing means positive power demand, exporting means negative demand)
        power_indices = {}
        for t in range(n_timesteps):
            power_indices[t] = [
                (t, 1.0),  # P_import contributes +1.0
                (n_timesteps + t, -1.0),  # P_export contributes -1.0
            ]

        return LinearModel(
            name=self.name,
            n_timesteps=n_timesteps,
            n_vars=n_vars,
            var_names=var_names,
            var_bounds=var_bounds,
            cost_coefficients=cost_coefficients,
            A_eq=A_eq,
            b_eq=b_eq,
            A_ub=A_ub,
            b_ub=b_ub,
            power_indices=power_indices,
        )