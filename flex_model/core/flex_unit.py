"""
FlexUnit: Physical modeling of flexibility-providing units.

This module provides the base class for modeling the techno-physical behavior of
flexibility assets in energy systems. FlexUnit is designed to capture ONLY the
physical and technical constraints, independent of any economic or market considerations.

DESIGN PHILOSOPHY
-----------------
Separation of concerns:
    - FlexUnit handles PHYSICS: power limits, energy capacity, ramp rates, efficiency
    - CostModel handles ECONOMICS: prices, degradation costs, revenues
    - FlexAsset handles OPERATIONS: combines physics + economics for decision-making

Time-dependent interface:
    - All physical properties are functions of time index t (integer)
    - Methods like availability(t), power_limits(t), capacity(t) allow time-varying behavior
    - Time resolution is not fixed (DT_HOURS can be modified in the settings); the model is horizon-agnostic

Energy headroom concept:
    - E_plus(t): energy that can still be DRAWN (upward flexibility)
    - E_minus(t): energy that can still be INJECTED (downward flexibility)
    - These evolve over time based on operational commands via update_state()

KEY PHYSICAL QUESTIONS ANSWERED BY FLEXUNIT
--------------------------------------------
1. How much extra energy can I draw or inject at time t?
2. What are the instantaneous power limits?
3. How fast can I change power (ramp limits)?
4. For how long can I sustain a given activation (duration constraints)?
5. Which discrete operating states are actually available?

CONCRETE EXAMPLES OF FLEXUNIT SUBCLASSES
-----------------------------------------
- BatteryUnit: Electrical storage with SOC, C-rate limits, efficiency losses
- ThermalStorageUnit: Hot water tank with temperature stratification, thermal losses
- HeatPumpUnit: COP-dependent operation, temperature constraints
- EVChargingUnit: Time-dependent availability (plugged in/out), SOC constraints
- PVCurtailmentUnit: Discrete curtailment states, irradiance-dependent availability
- IndustrialDRUnit: Load shedding with minimum rest periods, event costs

TYPICAL WORKFLOW
----------------
1. Create a FlexUnit subclass instance:
    battery = BatteryUnit(name="BESS_1", capacity_kwh=100, power_kw=50, efficiency=0.95)

2. Initialize state:
    battery.reset_state(E_plus_init=50.0, E_minus_init=50.0)  # 50% SOC

3. Query physical limits:
    P_import_max, P_export_max = battery.power_limits(t=10)

4. Execute operation:
    battery.update_state(t=10, P_grid_import_cmd=20.0, P_grid_export_cmd=0.0)

5. Check new state:
    soc = battery.E_plus / battery.C_spec

IMPLEMENTATION REQUIREMENTS FOR SUBCLASSES
-------------------------------------------
Abstract methods that MUST be implemented:
    - power_limits(t): Return (P_import_max, P_export_max) based on current state

Optional methods to override:
    - ramp_limits(t): Return (dP_import_max, dP_export_max) if ramp constraints exist
    - duration_limits(): Return (t_max_active, t_min_rest) for min up/down times
    - capacity(t): Override if capacity is time-dependent (e.g., temperature effects)
    - feasible_access_states(t): Override to filter discrete states dynamically

State management:
    - Call update_state() from subclass after computing losses/gains
    - Use reset_state() to initialize or restart simulation
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Sequence, Tuple

from flex_model.core.access_state import AccessState
from flex_model.settings import DT_HOURS


class FlexUnit(ABC):
    """
    Base class for physical modeling of flexibility-providing units.

    Subclasses must implement: power_limits(t).
    """

    dt_hours: float = DT_HOURS

    # ------------------------------------------------------------------
    # 1. Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        name: str,
        C_spec: float,
        availability_fn: Optional[Callable[[int], float]] = None,
        access_states: Optional[Sequence[AccessState]] = None,
    ) -> None:
        """
        Args:
            name:
                Human-readable identifier of the unit.

            C_spec:
                Nameplate capacity of the unit. Interpretation depends on subclass:
                    - [kWh]  for storage-like units (battery, TES)
                    - [kWh_th] or thermal mass equivalent for building TES
                    - [kW]   for purely power-based flexibility (e.g. curtailable load)

            availability_fn:
                Function alpha(t) -> [0, 1] describing time-dependent availability.
                Examples:
                    - EV connected only between arrival and departure time.
                    - Demand response allowed only during working hours.
                    - Asset offline during maintenance.
                If None, the unit is assumed fully available (alpha = 1) at all times.

            access_states:
                Discrete operating states (e.g. [0%, 30%, 70%, 100%] for PV curtailment).
                If None or empty, the unit is assumed to allow *continuous* control
                between its physical bounds.
        """
        self.name: str = name
        self.C_spec: float = C_spec

        # Availability: alpha(t) in [0, 1]
        self._availability_fn: Callable[[int], float] = (
            availability_fn if availability_fn is not None else (lambda t: 1.0)
        )

        # Discrete states of access (may be empty => continuous control)
        self.access_states: List[AccessState] = list(access_states or [])

        # ------------------------------------------------------------------
        # Internal energy headroom (interpreted by subclasses):
        #
        #   E_plus(t)  >= 0  : energy that can still be DRAWN (upward flexibility)
        #   E_minus(t) >= 0  : energy that can still be INJECTED (downward flexibility)
        #
        # Units are typically [kWh] or [kWh_th] depending on context.
        # These evolve over time via the update_state() method.
        # ------------------------------------------------------------------
        self.E_plus: float = 0.0
        self.E_minus: float = 0.0

        # ------------------------------------------------------------------
        # Internal power state (optional tracking of current activation).
        # P_grid_import: power imported from grid [kW]
        # P_grid_export: power exported to grid [kW]
        # Net power = P_grid_export - P_grid_import
        # ------------------------------------------------------------------
        self.P_grid_import: float = 0.0
        self.P_grid_export: float = 0.0

        # Duration tracking for min up/down time constraints (if used)
        self._time_in_active_state: int = 0  # [time steps]

    # ------------------------------------------------------------------
    # 2. Basic properties
    # ------------------------------------------------------------------

    def availability(self, t: int) -> float:
        """
        Effective availability factor alpha(t) in [0, 1].

        It scales the usable capacity and power at time t.
        """
        return max(0.0, min(1.0, self._availability_fn(t)))

    def capacity(self, t: int) -> float:
        """
        Effective capacity C(t) at time t:

            C(t) = C_spec * availability(t)

        Subclasses may override if capacity itself is time-dependent
        (e.g. temperature-dependent battery capacity).
        """
        return self.C_spec * self.availability(t)

    # ------------------------------------------------------------------
    # 3. Physical limits (to be implemented/overridden by subclasses)
    # ------------------------------------------------------------------

    @abstractmethod
    def power_limits(self, t: int) -> Tuple[float, float]:
        """
        Return instantaneous power limits at time t.

        Returns:
            (P_import_max, P_export_max), both >= 0, in [kW].

        Interpretation:
            - P_import_max(t): maximum power that can be imported from grid
            - P_export_max(t): maximum power that can be exported to grid

        These limits should already take availability and internal energy state
        (E_plus / E_minus) into account.
        """
        raise NotImplementedError

    def ramp_limits(self, t: int) -> Tuple[Optional[float], Optional[float]]:
        """
        Optional ramp rate limits between time steps.

        Returns:
            (dP_import_max, dP_export_max) in [kW per time step], or (None, None)
            if ramp limits are not modelled.

        Interpretation:
            - |P_grid_import(t) - P_grid_import(t-1)| <= dP_import_max
            - |P_grid_export(t) - P_grid_export(t-1)| <= dP_export_max
        """
        return None, None  # by default: no ramp constraints

    def duration_limits(self) -> Tuple[Optional[int], Optional[int]]:
        """
        Optional duration constraints for activation.

        Returns:
            (t_max_active, t_min_rest) in [time steps], or (None, None) if not used.

        Examples:
            - Industrial DR process can only be shed for at most N steps
              and then must rest for M steps.
            - A CHP unit might have minimum up/down times.

        Enforcing these constraints is typically done at the optimisation layer,
        using self._time_in_active_state and unit-specific logic.
        """
        return None, None

    # ------------------------------------------------------------------
    # 4. Discrete states-of-access
    # ------------------------------------------------------------------

    def has_discrete_states(self) -> bool:
        """Return True if this unit uses discrete states-of-access."""
        return len(self.access_states) > 0

    def feasible_access_states(self, t: int) -> Sequence[AccessState]:
        """
        Return the list of access states that are feasible at time t.

        The default implementation simply returns all configured states.
        Subclasses may override this to:
            - disable certain states based on E_plus/E_minus,
            - depend on operating temperature, SOC, etc.
        """
        return self.access_states

    # ------------------------------------------------------------------
    # 5. State update (energy headroom)
    # ------------------------------------------------------------------

    def update_state(
        self,
        t: int,
        P_grid_import_cmd: float,
        P_grid_export_cmd: float,
        n_timesteps: int = 1,
        P_loss: float = 0.0,
        P_gain: float = 0.0,
    ) -> None:
        """
        Generic energy headroom update based on commanded power at grid connection.

        Args:
            t:
                Time index (integer). Used only for availability / capacity
                checks here; physics are time-homogeneous.

            P_grid_import_cmd:
                Power imported from grid [kW] during this time step.
                For batteries: charging power.
                For loads: consumption power.
                Positive = importing from grid.

            P_grid_export_cmd:
                Power exported to grid [kW] during this time step.
                For batteries: discharging power.
                For generators: production power.
                Positive = exporting to grid.

            n_timesteps:
                Number of timesteps [-].

            P_loss:
                Power lost during this time step [kW] due to inefficiencies
                (self-discharge, thermal losses, etc.).

            P_gain:
                Power gained during this time step [kW] (e.g. passive solar gains
                in a building, unavoidable waste heat, etc.).

        Notes:
            - This method updates E_plus and E_minus **in-place**.
            - Subclasses are responsible for calling this with physically
              meaningful commands and loss/gain values.
            - Grid perspective: import = flow INTO site, export = flow OUT OF site
        """
        # Clip commands to be non-negative (interpretation of signs is external)
        P_import = max(0.0, P_grid_import_cmd)
        P_export = max(0.0, P_grid_export_cmd)

        self.P_grid_import = P_import
        self.P_grid_export = P_export

        # Convert powers to energies for this time step
        delta_t = n_timesteps * DT_HOURS
        E_import = P_import * delta_t
        E_export = P_export * delta_t
        E_gain = P_gain * delta_t
        E_loss = P_loss * delta_t

        # Upper accessible energy: how much can still be exported (E_plus)
        # For generators: increases when producing (export), decreases when importing
        # For storage: decreases when exporting (discharge), increases when importing (charge)
        # Base implementation is for generators; storage overrides this method
        delta_plus = E_export - E_import + E_gain - E_loss
        self.E_plus = max(0.0, min(self.E_plus + delta_plus, self.capacity(t)))

        # Lower accessible energy: how much can still be imported (E_minus)
        # Inverse of E_plus
        delta_minus = E_import - E_export - E_gain + E_loss
        self.E_minus = max(0.0, min(self.E_minus + delta_minus, self.capacity(t)))

    # ------------------------------------------------------------------
    # 6. Initialisation / reset hooks
    # ------------------------------------------------------------------

    def reset_state(
        self,
        E_plus_init: float,
        E_minus_init: float,
    ) -> None:
        """
        Reset internal energy headroom to initial values.

        Args:
            E_plus_init:
                Initial 'draw headroom' [kWh].

            E_minus_init:
                Initial 'inject headroom' [kWh].

        Used at the start of a simulation or scenario.
        """
        self.E_plus = max(0.0, E_plus_init)
        self.E_minus = max(0.0, E_minus_init)
        self.P_grid_import = 0.0
        self.P_grid_export = 0.0
        self._time_in_active_state = 0