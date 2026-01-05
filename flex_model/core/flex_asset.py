"""
FlexAsset: Operational composition of physical units and economic models.

This module provides the operational layer that combines physical behavior (FlexUnit)
with economic evaluation (CostModel) to create a unified interface for optimization
and decision-making.

DESIGN PHILOSOPHY
-----------------
Three-layer architecture:
    1. FlexUnit (physical layer): Techno-physical constraints and behavior
    2. CostModel (economic layer): Prices, costs, and revenues
    3. FlexAsset (operational layer): Combines physics + economics for decision-making

A FlexAsset is the composition of a FlexUnit and a CostModel, bridging the gap between
what is physically possible and what is economically optimal. It provides:
    - Feasibility checks from physics
    - Cost evaluation from economics
    - Operational decision interface for optimizers
    - Metrics tracking for reporting and analysis

Separation of concerns:
    - Physical models can be validated independently against datasheets/measurements
    - Economic models can be swapped or updated without changing physics
    - Different cost scenarios can be compared using the same physical model
    - Same FlexUnit can be used with multiple cost assumptions

THE ROLE OF OPERATIONAL DECISIONS
----------------------------------
The key insight is that OPERATIONAL DECISIONS are what bind physics to economics:
    - Physical model: "What CAN I do?" (power limits, energy capacity, ramp rates)
    - Cost model: "What does it COST to do it?" (prices, degradation, revenues)
    - FlexAsset: "What SHOULD I do?" (evaluate and execute optimal decisions)

FlexAsset is the primary interface for optimization because it can:
    1. Evaluate proposed operations: feasible? how much does it cost?
    2. Execute confirmed operations: update physical state, track costs
    3. Provide limits and constraints: delegate to FlexUnit
    4. Report metrics: throughput, cost, activations

TYPICAL WORKFLOW
----------------
1. Create physical unit:
    battery_unit = BatteryUnit(name="BESS_1", capacity_kwh=100, power_kw=50, ...)

2. Create cost model:
    battery_cost = BatteryCostModel(
        name="battery_cost",
        c_inv=500,  # CHF/kWh
        p_int=0.05,  # CHF/kWh degradation
        p_E_buy={0: 0.20, 1: 0.25, ...},  # time-varying prices
    )

3. Compose flex asset:
    battery_flex = BatteryFlex(unit=battery_unit, cost_model=battery_cost)

4. Use in optimization:
    optimizer = Optimizer()
    optimizer.add_asset(battery_flex)
    optimal_schedule = optimizer.solve()

5. Evaluate operation (during optimization):
    result = battery_flex.evaluate_operation(t=10, P_grid_import=20.0, P_grid_export=0.0)
    if result['feasible']:
        cost = result['cost']
        # ... use in objective function ...

6. Execute operation (after optimization):
    battery_flex.execute_operation(t=10, P_grid_import=20.0, P_grid_export=0.0)

7. Get metrics:
    metrics = battery_flex.get_metrics()
    print(f"Total cost: {metrics['total_cost_eur']} EUR")
    print(f"Throughput: {metrics['total_throughput_kwh']} kWh")

EXAMPLES OF FLEXASSET SUBCLASSES
---------------------------------
Physical assets:
    - BatteryFlex: Battery storage with degradation costs and energy arbitrage
    - HeatPumpFlex: Heat pump with COP-dependent efficiency and thermal storage
    - DHWFlex: Domestic hot water with thermal losses and demand profiles
    - EVChargingFlex: EV charging with arrival/departure constraints
    - PVFlex: PV curtailment with opportunity costs

Market options (may not need FlexUnit):
    - ImbalanceFlex: Imbalance settlement without physical constraints
    - ReserveFlex: Reserve capacity products (FCR, aFRR, mFRR)
    - IntradayFlex: Intraday trading between markets

IMPLEMENTATION REQUIREMENTS FOR SUBCLASSES
-------------------------------------------
Abstract methods that MUST be implemented:
    - evaluate_operation(t, dt_hours, P_grid_import, P_grid_export):
        Check feasibility and calculate cost WITHOUT modifying state.

    - execute_operation(t, dt_hours, P_grid_import, P_grid_export):
        Execute the operation, updating physical state and tracking metrics.

Typical implementation pattern:
    1. evaluate_operation():
        a. Check physical feasibility: P_grid_import <= P_import_max, etc.
        b. Calculate cost using cost_model.step_cost()
        c. Return dict with 'feasible', 'cost', 'violations'
        d. Must be STATELESS (no modifications to self.unit or tracking)

    2. execute_operation():
        a. Call self.unit.update_state() to update physics
        b. Update operational tracking: _total_cost_eur, _total_throughput_kwh
        c. Caller ensures feasibility before calling this method

Optional methods to override:
    - get_metrics(): Add asset-specific metrics beyond the base set
    - reset(): Override if additional state needs resetting

COMMUNICATION PROTOCOL
----------------------
FlexAsset communicates with CostModel via two data structures:

1. flex_state: Physical state of the unit
    Example: {'soc': 0.7, 'E_plus': 30.0, 'E_minus': 70.0}

2. activation: Operational decision
    Example: {'P_grid_import': 20.0, 'P_grid_export': 0.0, 'dt_hours': 0.25}

The structure of these is agreed between FlexAsset and CostModel implementations.
The base classes (FlexUnit, CostModel, FlexAsset) don't dictate the structure,
allowing flexibility for different asset types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any

from flex_model.core.flex_unit import FlexUnit
from flex_model.settings import DT_HOURS


class FlexAsset(ABC):
    """
    Operational composition that combines FlexUnit (physics) with CostModel (economics).

    Subclasses must implement: evaluate_operation(...), execute_operation(...).
    """

    dt_hours: float = DT_HOURS

    def __init__(self, unit: FlexUnit, name: str = None) -> None:
        """
        Args:
            unit:
                The physical FlexUnit that provides techno-physical behavior.

            name:
                Optional human-readable name for this asset. If None, uses unit.name.
        """
        self.unit = unit
        self.name = name or unit.name

        # Operational tracking (can be used for reporting, validation, etc.)
        self._total_throughput_kwh: float = 0.0
        self._total_cost_eur: float = 0.0
        self._num_activations: int = 0

    # ------------------------------------------------------------------
    # Core operational interface
    # ------------------------------------------------------------------

    @abstractmethod
    def evaluate_operation(
        self,
        t: int,
        P_grid_import: float,
        P_grid_export: float,
        n_timesteps: int = 1,
    ) -> Dict[str, Any]:
        """
        Evaluate an operational decision without executing it.

        Args:
            t: Time index (integer).
            P_grid_import: Proposed power import from grid [kW] for this time step.
            P_grid_export: Proposed power export to grid [kW] for this time step.
            n_timesteps: Number of time steps (default: 1).

        Returns:
            Dictionary containing at minimum:
                {
                    'feasible': bool,
                    'cost': float [EUR],
                    'violations': List[str],
                }
            Subclasses may include additional info (soc, temperature, etc.).

        Notes:
            - Must be STATELESS (no modifications to self.unit or tracking).
            - Called repeatedly during optimization.
            - Time duration is calculated as: delta_t = n_timesteps * FlexAsset.dt_hours
        """
        raise NotImplementedError

    @abstractmethod
    def execute_operation(
        self,
        t: int,
        P_grid_import: float,
        P_grid_export: float,
        n_timesteps: int = 1,
    ) -> None:
        """
        Execute an operational decision, updating physical state and metrics.

        Args:
            t: Time index (integer).
            P_grid_import: Power import from grid command [kW] to execute.
            P_grid_export: Power export to grid command [kW] to execute.
            n_timesteps: Number of time steps (default: 1).

        Notes:
            - DOES modify state (calls self.unit.update_state()).
            - Updates tracking: _total_throughput_kwh, _total_cost_eur, _num_activations.
            - Caller ensures feasibility before calling.
            - Time duration is calculated as: delta_t = n_timesteps * FlexAsset.dt_hours
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Convenience methods for optimization interface
    # ------------------------------------------------------------------

    def power_limits(self, t: int) -> tuple[float, float]:
        """
        Delegate to underlying FlexUnit for physical power limits.

        Returns:
            (P_import_max, P_export_max) in [kW].
        """
        return self.unit.power_limits(t)

    def reset(self, E_plus_init: float, E_minus_init: float) -> None:
        """
        Reset the asset to initial state.

        Args:
            E_plus_init: Initial 'draw headroom' [kWh].
            E_minus_init: Initial 'inject headroom' [kWh].

        Resets both the physical unit state and operational tracking metrics.
        """
        self.unit.reset_state(E_plus_init, E_minus_init)
        self._total_throughput_kwh = 0.0
        self._total_cost_eur = 0.0
        self._num_activations = 0

    # ------------------------------------------------------------------
    # Operational metrics (for reporting and analysis)
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """
        Return operational metrics tracked during simulation.

        Returns:
            Dictionary with keys:
                - 'total_throughput_kwh': Total energy throughput [kWh]
                - 'total_cost_eur': Total operational cost [EUR]
                - 'num_activations': Number of times asset was activated

        Subclasses may add additional metrics specific to their type.
        """
        return {
            'total_throughput_kwh': self._total_throughput_kwh,
            'total_cost_eur': self._total_cost_eur,
            'num_activations': self._num_activations,
        }