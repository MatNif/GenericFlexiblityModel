"""
Abstract interface for optimization results.

This module defines the standard interface that all optimizer implementations
must provide, enabling visualization and analysis code to work with any
optimizer (LP, MILP, genetic algorithms, etc.) without modification.

Design Pattern: Abstract Base Class (Strategy Pattern)
    - Defines common interface for result extraction
    - Allows swapping optimizer implementations without breaking visualization code
    - Enforces API consistency across different optimization algorithms
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any


class OptimizationResult(ABC):
    """
    Abstract base class for optimization results.

    This interface standardizes access to optimization results regardless of
    the underlying algorithm (LP, MILP, GA, etc.). All optimizer-specific
    implementations must inherit from this class and implement its methods.

    Subclasses should provide:
        - LPOptimizationResult: For linear programming solvers
        - MILPOptimizationResult: For mixed-integer linear programming
        - GAOptimizationResult: For genetic algorithm-based optimization
        - etc.

    Attributes:
        assets: Dict mapping asset_name -> FlexAsset instance
        imbalance: Dict mapping timestep -> imbalance power [kW]
        dt_hours: Timestep duration [h]
        n_timesteps: Number of timesteps in optimization horizon
    """

    # These attributes must be set by concrete implementations
    assets: Dict[str, Any]
    imbalance: Dict[int, float]
    dt_hours: float
    n_timesteps: int

    @abstractmethod
    def get_power_profile(
        self,
        asset_name: str,
        return_format: str = 'dict'
    ) -> Dict[str, Any] | Any:
        """
        Extract power dispatch time-series for a specific asset.

        Args:
            asset_name: Name of asset to extract (must exist in assets dict)
            return_format: 'dict' or 'dataframe' (if pandas available)

        Returns:
            If return_format='dict':
                {
                    'timesteps': [0, 1, 2, ...],
                    'P_charge': [values...],      # For batteries (kWh per timestep)
                    'P_discharge': [values...],   # For batteries (kWh per timestep)
                    'P_import': [values...],      # For markets (kWh per timestep)
                    'P_export': [values...],      # For markets (kWh per timestep)
                    'P_net': [values...],         # Net power contribution (kW)
                }

            If return_format='dataframe':
                pandas.DataFrame with columns as above, index=timesteps

        Raises:
            ValueError: If asset_name not found
            ImportError: If return_format='dataframe' but pandas not available
        """
        pass

    @abstractmethod
    def get_soc_profile(
        self,
        battery_name: str,
        return_format: str = 'dict'
    ) -> Dict[str, Any] | Any:
        """
        Extract state of charge (SOC) evolution for a battery.

        Args:
            battery_name: Name of battery asset
            return_format: 'dict' or 'dataframe'

        Returns:
            If return_format='dict':
                {
                    'timesteps': [0, 1, 2, ..., n_timesteps],  # n_timesteps+1 values
                    'SOC': [values...],           # SOC [kWh]
                    'SOC_percent': [values...],   # SOC [%] of capacity
                }

            If return_format='dataframe':
                pandas.DataFrame with columns as above

        Raises:
            ValueError: If battery_name not found or is not a battery
            ImportError: If return_format='dataframe' but pandas not available
        """
        pass

    @abstractmethod
    def get_cost_breakdown(self) -> Dict[str, float]:
        """
        Calculate cost breakdown by asset and component.

        Returns:
            Dictionary with structure:
                {
                    'total_cost': float,
                    'by_asset': {
                        'asset_name': {
                            'total': float,
                            'investment': float,
                            'fixed_om': float,
                            'variable_om': float,
                            'market_net': float,
                        },
                        ...
                    }
                }

        Note:
            Specific cost components depend on optimizer capabilities.
            Some implementations may provide simplified breakdowns.
        """
        pass

    @abstractmethod
    def get_utilization_metrics(self, asset_name: str) -> Dict[str, float]:
        """
        Compute utilization metrics for an asset.

        Args:
            asset_name: Name of asset to analyze

        Returns:
            Dictionary with metrics:
                {
                    'capacity_factor': float,        # Actual / Maximum throughput
                    'utilization_hours': float,      # Hours with non-zero activation
                    'num_cycles': float,             # Full charge-discharge cycles (batteries)
                    'avg_cycle_depth': float,        # Average cycle depth [%] (batteries)
                    'max_power_used': float,         # Peak power [kW]
                }

        Raises:
            ValueError: If asset_name not found
        """
        pass

    @abstractmethod
    def get_imbalance_profile(self, return_format: str = 'dict') -> Dict[str, Any] | Any:
        """
        Get the imbalance profile that drove the optimization.

        Args:
            return_format: 'dict' or 'dataframe'

        Returns:
            If return_format='dict':
                {'timesteps': [...], 'imbalance': [...]}
            If return_format='dataframe':
                pandas.DataFrame with columns [timesteps, imbalance]

        Raises:
            ImportError: If return_format='dataframe' but pandas not available
        """
        pass

    @abstractmethod
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of optimization results.

        Returns:
            Dictionary with key metrics across all assets and optimization status.
            Structure depends on implementation but should include:
                {
                    'optimization': {
                        'success': bool,
                        'total_cost': float,
                        'message': str,
                        'n_timesteps': int,
                        'dt_hours': float,
                        'total_hours': float,
                    },
                    'assets': {
                        'asset_name': {...utilization_metrics...},
                        ...
                    },
                    'imbalance': {
                        'total_energy': float,
                        'peak_deficit': float,
                        'peak_surplus': float,
                    }
                }
        """
        pass

    @abstractmethod
    def get_total_cost(self) -> float:
        """
        Get the total cost (objective value) from optimization.

        This abstracts away optimizer-specific result structures, allowing
        metrics calculators to work with any optimizer implementation.

        Returns:
            Total cost/objective value [EUR or currency unit used].

        Example:
            >>> cost = result.get_total_cost()
            >>> print(f"Total cost: {cost:.2f} EUR")
        """
        pass

    @abstractmethod
    def is_successful(self) -> bool:
        """
        Check if optimization completed successfully.

        Different optimizers may define success differently (e.g., 'optimal',
        'converged', 'feasible'), but this method provides a unified interface.

        Returns:
            True if optimization succeeded, False otherwise.

        Example:
            >>> if result.is_successful():
            >>>     print("Optimization successful!")
            >>> else:
            >>>     print(f"Optimization failed: {result.get_status_message()}")
        """
        pass

    @abstractmethod
    def get_status_message(self) -> str:
        """
        Get human-readable status/error message from optimization.

        Provides details about optimization outcome, including error messages
        if optimization failed.

        Returns:
            Status message string (e.g., "Optimal solution found",
            "Infeasible problem", "Converged after 1000 iterations").

        Example:
            >>> print(result.get_status_message())
            "Optimal solution found"
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """
        String representation of optimization result.

        Should include key information like status, cost, and number of assets.
        """
        pass
