"""
Linear model representation for optimization.

This module defines the data structure for representing FlexAssets as linear
programs that can be aggregated and solved by an optimizer.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np


@dataclass
class LinearModel:
    """
    Linear representation of a FlexAsset for optimization.

    Contains all the information needed to represent an asset's behavior
    as a linear program: decision variables, costs, and constraints.

    Attributes:
        name:
            Human-readable identifier for this asset.

        n_timesteps:
            Number of timesteps in the optimization horizon.

        n_vars:
            Total number of decision variables.

        var_names:
            List of variable names (e.g., ['P_charge_0', 'P_charge_1', ...]).

        var_bounds:
            List of (lower, upper) bounds for each variable.

        cost_coefficients:
            Coefficients for the objective function (length n_vars).
            Objective: minimize c^T x

        A_eq:
            Coefficient matrix for equality constraints (shape: n_eq x n_vars).
            A_eq @ x == b_eq

        b_eq:
            Right-hand side for equality constraints (length n_eq).

        A_ub:
            Coefficient matrix for inequality constraints (shape: n_ub x n_vars).
            A_ub @ x <= b_ub

        b_ub:
            Right-hand side for inequality constraints (length n_ub).

        power_indices:
            Indices of variables that represent net power at each timestep.
            Used for energy balance constraint across assets.
            Dict mapping timestep -> list of (var_index, coefficient).

            Example: {0: [(0, 1.0), (5, -1.0)]} means at t=0:
                     net_power = 1.0 * x[0] - 1.0 * x[5]
    """

    name: str
    n_timesteps: int
    n_vars: int
    var_names: List[str]
    var_bounds: List[Tuple[float, Optional[float]]]
    cost_coefficients: np.ndarray
    A_eq: Optional[np.ndarray] = None
    b_eq: Optional[np.ndarray] = None
    A_ub: Optional[np.ndarray] = None
    b_ub: Optional[np.ndarray] = None
    power_indices: Dict[int, List[Tuple[int, float]]] = field(default_factory=dict)

    def __post_init__(self):
        """Validate dimensions after initialization."""
        if len(self.var_names) != self.n_vars:
            raise ValueError(f"var_names length {len(self.var_names)} != n_vars {self.n_vars}")

        if len(self.var_bounds) != self.n_vars:
            raise ValueError(f"var_bounds length {len(self.var_bounds)} != n_vars {self.n_vars}")

        if len(self.cost_coefficients) != self.n_vars:
            raise ValueError(f"cost_coefficients length {len(self.cost_coefficients)} != n_vars {self.n_vars}")

        if self.A_eq is not None:
            if self.A_eq.shape[1] != self.n_vars:
                raise ValueError(f"A_eq columns {self.A_eq.shape[1]} != n_vars {self.n_vars}")
            if self.b_eq is None or len(self.b_eq) != self.A_eq.shape[0]:
                raise ValueError(f"b_eq length must match A_eq rows")

        if self.A_ub is not None:
            if self.A_ub.shape[1] != self.n_vars:
                raise ValueError(f"A_ub columns {self.A_ub.shape[1]} != n_vars {self.n_vars}")
            if self.b_ub is None or len(self.b_ub) != self.A_ub.shape[0]:
                raise ValueError(f"b_ub length must match A_ub rows")

    def get_summary(self) -> str:
        """Return a human-readable summary of the linear model."""
        lines = [
            f"LinearModel: {self.name}",
            f"  Timesteps: {self.n_timesteps}",
            f"  Variables: {self.n_vars}",
            f"  Equality constraints: {0 if self.A_eq is None else self.A_eq.shape[0]}",
            f"  Inequality constraints: {0 if self.A_ub is None else self.A_ub.shape[0]}",
            f"  Power coupling: {len(self.power_indices)} timesteps",
        ]
        return "\n".join(lines)