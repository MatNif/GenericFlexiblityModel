"""
Linear Programming Optimizer for FlexAssets.

Aggregates multiple FlexAssets represented as LinearModels and solves
the combined optimization problem with global energy balance constraints.
"""

from typing import List, Dict, Optional
import numpy as np
from scipy.optimize import linprog
from scipy import sparse

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

        This optimizer uses the industry standard sign convention throughout:
            Positive imbalance = SURPLUS (excess energy available in balancing group)
            Negative imbalance = DEFICIT (shortage, energy needed by balancing group)

        The optimization minimizes total cost while satisfying the energy balance:
            Σ(asset net_power) = - imbalance[t]  for each timestep t

        Where each asset's net_power represents its contribution to handling the imbalance, e.g.:
            - Battery charging → negative contribution (storage compensates for surplus)
            - Battery discharging → positive contribution (provides for deficit)
            - Market export → negative contribution (sends away surplus)
            - Market import → positive contribution (brings in for deficit)

        Args:
            imbalance: Dict mapping timestep -> power imbalance [kW] in industry standard convention.

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

        # Aggregate constraints using sparse matrices
        # 1. Block-diagonal assembly of per-asset constraints
        # Need to include empty blocks for assets without constraints to maintain alignment
        asset_eq_blocks = []
        asset_eq_rhs = []
        asset_ub_blocks = []
        asset_ub_rhs = []

        for asset in self.assets:
            # Always add a block (empty if asset has no constraints)
            if asset.A_eq is not None:
                # Convert to sparse if it's dense (backward compatibility)
                if not sparse.issparse(asset.A_eq):
                    asset_eq_blocks.append(sparse.csr_matrix(asset.A_eq))
                else:
                    asset_eq_blocks.append(asset.A_eq)
                asset_eq_rhs.append(asset.b_eq)
            else:
                # Add empty block with shape (0, n_vars) to maintain block structure
                asset_eq_blocks.append(sparse.csr_matrix((0, asset.n_vars)))
                asset_eq_rhs.append(np.array([]))

            if asset.A_ub is not None:
                # Convert to sparse if it's dense (backward compatibility)
                if not sparse.issparse(asset.A_ub):
                    asset_ub_blocks.append(sparse.csr_matrix(asset.A_ub))
                else:
                    asset_ub_blocks.append(asset.A_ub)
                asset_ub_rhs.append(asset.b_ub)
            else:
                # Add empty block with shape (0, n_vars) to maintain block structure
                asset_ub_blocks.append(sparse.csr_matrix((0, asset.n_vars)))
                asset_ub_rhs.append(np.array([]))

        # 2. Build energy balance constraints as sparse matrix
        # One constraint per timestep: sum of all asset net powers = imbalance[t]
        balance_eq = sparse.lil_matrix((self.n_timesteps, total_vars))
        balance_rhs = np.zeros(self.n_timesteps)

        for t in range(self.n_timesteps):
            # For each asset, add its contribution to net power at time t
            var_offset = 0
            for asset in self.assets:
                if t in asset.power_indices:
                    for var_idx, coeff in asset.power_indices[t]:
                        # var_idx is relative to asset's variables
                        # need to offset to global variable index
                        global_idx = var_offset + var_idx
                        balance_eq[t, global_idx] = coeff

                var_offset += asset.n_vars

            balance_rhs[t] = -self.imbalance[t]

        # Convert balance to CSC format
        balance_eq = balance_eq.tocsc()

        # 3. Stack everything together
        # Combine per-asset constraints into block-diagonal matrix
        A_eq_assets = sparse.block_diag(asset_eq_blocks, format='csc')

        # Filter out empty RHS arrays before concatenating
        non_empty_eq_rhs = [rhs for rhs in asset_eq_rhs if len(rhs) > 0]
        if non_empty_eq_rhs:
            b_eq_assets = np.concatenate(non_empty_eq_rhs)
            # Stack with energy balance constraints
            A_eq = sparse.vstack([A_eq_assets, balance_eq], format='csc')
            b_eq = np.concatenate([b_eq_assets, balance_rhs])
        else:
            # No asset constraints, only energy balance
            A_eq = balance_eq
            b_eq = balance_rhs

        # Handle inequality constraints similarly
        A_ub_assets = sparse.block_diag(asset_ub_blocks, format='csc')
        non_empty_ub_rhs = [rhs for rhs in asset_ub_rhs if len(rhs) > 0]
        if non_empty_ub_rhs:
            A_ub = A_ub_assets
            b_ub = np.concatenate(non_empty_ub_rhs)
        else:
            A_ub = None
            b_ub = None

        print(f"  Total variables: {total_vars}")
        print(f"  Equality constraints: {A_eq.shape[0] if A_eq is not None else 0}")
        print(f"  Inequality constraints: {A_ub.shape[0] if A_ub is not None else 0}")
        if A_eq is not None and sparse.issparse(A_eq):
            sparsity = 100.0 * (1.0 - A_eq.nnz / (A_eq.shape[0] * A_eq.shape[1]))
            print(f"  A_eq sparsity: {sparsity:.3f}% zeros ({A_eq.nnz:,} non-zeros)")

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