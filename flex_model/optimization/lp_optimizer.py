"""
Linear Programming Optimizer for FlexAssets.

Aggregates multiple FlexAssets represented as LinearModels and solves
the combined optimization problem with global energy balance constraints.
"""

from typing import List, Dict, Optional
import numpy as np
from scipy.optimize import linprog

from .linear_model import LinearModel


class LPOptimizer:
    """
    Linear programming optimizer for aggregating FlexAssets.

    Combines multiple FlexAssets (represented as LinearModels) into a unified
    LP problem with global energy balance constraints.
    """

    def __init__(self, n_timesteps: int):
        """
        Initialize optimizer.

        Args:
            n_timesteps: Number of timesteps in optimization horizon.
        """
        self.n_timesteps = n_timesteps
        self.assets: List[LinearModel] = []
        self.imbalance: Optional[Dict[int, float]] = None

    def add_asset(self, linear_model: LinearModel) -> None:
        """
        Add a FlexAsset's linear model to the optimization problem.

        Args:
            linear_model: LinearModel representing the asset.

        Raises:
            ValueError: If asset has incompatible timestep count.
        """
        if linear_model.n_timesteps != self.n_timesteps:
            raise ValueError(
                f"Asset {linear_model.name} has {linear_model.n_timesteps} timesteps, "
                f"but optimizer expects {self.n_timesteps}"
            )

        self.assets.append(linear_model)

    def set_imbalance(self, imbalance: Dict[int, float]) -> None:
        """
        Set the energy imbalance profile that assets must collectively satisfy.

        Args:
            imbalance: Dict mapping timestep -> power imbalance [kW].
                      Positive = need to buy energy (deficit)
                      Negative = can sell energy (excess)

        Raises:
            ValueError: If imbalance has wrong length.
        """
        if len(imbalance) != self.n_timesteps:
            raise ValueError(
                f"Imbalance has {len(imbalance)} timesteps, "
                f"but optimizer expects {self.n_timesteps}"
            )

        self.imbalance = imbalance

    def solve(self) -> Dict[str, any]:
        """
        Solve the aggregated LP optimization problem.

        Returns:
            Dictionary containing:
                - 'success': bool, whether optimization succeeded
                - 'cost': float, optimal total cost
                - 'solution': Dict mapping asset name -> Dict of variable values
                - 'message': str, solver message

        Raises:
            RuntimeError: If no assets added or imbalance not set.
        """
        if not self.assets:
            raise RuntimeError("No assets added to optimizer")

        if self.imbalance is None:
            raise RuntimeError("Imbalance profile not set")

        # Build aggregated problem
        print(f"Building aggregated LP with {len(self.assets)} assets...")

        # Calculate total number of variables
        total_vars = sum(asset.n_vars for asset in self.assets)
        var_offset = 0
        asset_var_ranges = {}  # Track variable indices for each asset

        # Aggregate cost coefficients
        c = np.zeros(total_vars)
        for asset in self.assets:
            asset_var_ranges[asset.name] = (var_offset, var_offset + asset.n_vars)
            c[var_offset:var_offset + asset.n_vars] = asset.cost_coefficients
            var_offset += asset.n_vars

        # Aggregate variable bounds
        bounds = []
        for asset in self.assets:
            bounds.extend(asset.var_bounds)

        # Aggregate constraints
        constraints_eq = []
        bounds_eq = []
        constraints_ub = []
        bounds_ub = []

        var_offset = 0
        for asset in self.assets:
            # Add asset's equality constraints
            if asset.A_eq is not None:
                for i in range(asset.A_eq.shape[0]):
                    row = np.zeros(total_vars)
                    row[var_offset:var_offset + asset.n_vars] = asset.A_eq[i, :]
                    constraints_eq.append(row)
                    bounds_eq.append(asset.b_eq[i])

            # Add asset's inequality constraints
            if asset.A_ub is not None:
                for i in range(asset.A_ub.shape[0]):
                    row = np.zeros(total_vars)
                    row[var_offset:var_offset + asset.n_vars] = asset.A_ub[i, :]
                    constraints_ub.append(row)
                    bounds_ub.append(asset.b_ub[i])

            var_offset += asset.n_vars

        # Add global energy balance constraints (one per timestep)
        # Sum of all asset net powers = imbalance at each timestep
        # net_power_asset1 + net_power_asset2 + ... = imbalance[t]
        for t in range(self.n_timesteps):
            row = np.zeros(total_vars)

            # For each asset, add its contribution to net power at time t
            var_offset = 0
            for asset in self.assets:
                if t in asset.power_indices:
                    for var_idx, coeff in asset.power_indices[t]:
                        # var_idx is relative to asset's variables
                        # need to offset to global variable index
                        global_idx = var_offset + var_idx
                        row[global_idx] = coeff

                var_offset += asset.n_vars

            constraints_eq.append(row)
            bounds_eq.append(self.imbalance[t])

        # Convert to numpy arrays
        A_eq = np.array(constraints_eq) if constraints_eq else None
        b_eq = np.array(bounds_eq) if bounds_eq else None
        A_ub = np.array(constraints_ub) if constraints_ub else None
        b_ub = np.array(bounds_ub) if bounds_ub else None

        print(f"  Total variables: {total_vars}")
        print(f"  Equality constraints: {len(constraints_eq)}")
        print(f"  Inequality constraints: {len(constraints_ub)}")

        # Solve LP
        print("Solving LP...")
        result = linprog(
            c=c,
            A_eq=A_eq,
            b_eq=b_eq,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method='highs',
            options={'disp': False}
        )

        if not result.success:
            print(f"  WARNING: Optimization failed: {result.message}")
            return {
                'success': False,
                'cost': None,
                'solution': None,
                'message': result.message,
            }

        print(f"  Optimal cost: {result.fun:.2f}")

        # Extract solution for each asset
        solution = {}
        for asset in self.assets:
            var_start, var_end = asset_var_ranges[asset.name]
            asset_solution = result.x[var_start:var_end]

            # Map variable values back to named dict
            asset_vars = {}
            for i, var_name in enumerate(asset.var_names):
                asset_vars[var_name] = asset_solution[i]

            solution[asset.name] = asset_vars

        return {
            'success': True,
            'cost': result.fun,
            'solution': solution,
            'message': result.message,
        }

    def get_summary(self) -> str:
        """Return a human-readable summary of the optimization problem."""
        lines = [
            f"LPOptimizer: {self.n_timesteps} timesteps",
            f"  Assets: {len(self.assets)}",
        ]

        for asset in self.assets:
            lines.append(f"    - {asset.name}: {asset.n_vars} variables")

        if self.imbalance is not None:
            imb_values = list(self.imbalance.values())
            lines.append(f"  Imbalance: min={min(imb_values):.2f}, max={max(imb_values):.2f} kW")

        total_vars = sum(asset.n_vars for asset in self.assets)
        lines.append(f"  Total variables: {total_vars}")

        return "\n".join(lines)