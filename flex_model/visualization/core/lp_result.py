"""
Linear Programming optimization result implementation.

This module provides LP-specific result processing, extracting data from
LP solvers that use variable naming conventions like '{asset_name}_P_charge_{t}'.
"""

from __future__ import annotations
from typing import Dict, Any

from flex_model.visualization.core.result_processor import OptimizationResult
from flex_model.settings import DT_HOURS

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class LPOptimizationResult(OptimizationResult):
    """
    Linear Programming optimizer result wrapper.

    Extracts and processes results from LP solvers, handling LP-specific
    variable naming conventions and data structures.

    Attributes:
        lp_result: Raw LP solver output dict
        assets: Dict of FlexAsset instances used in optimization
        imbalance: Dict mapping timestep -> imbalance power [kW]
        n_timesteps: Number of timesteps in optimization horizon
    """

    def __init__(
        self,
        lp_result: Dict[str, Any],
        assets: Dict[str, Any],
        imbalance: Dict[int, float],
    ) -> None:
        """
        Initialize LP result wrapper.

        Args:
            lp_result:
                Output from LP optimizer.solve(), must contain keys:
                    - 'success': bool
                    - 'cost': float (total cost)
                    - 'solution': dict {asset_name: {var_name: value}}
                    - 'message': str

            assets:
                Dict mapping asset_name -> FlexAsset instance.
                Names must match those used in optimization.

            imbalance:
                Dict mapping timestep -> imbalance power [kW].
                Positive = deficit (need import), Negative = excess (can export).
        """
        self.lp_result = lp_result
        self.assets = assets
        self.imbalance = imbalance
        self.n_timesteps = len(imbalance)

        # Validate result structure
        if not lp_result.get('success', False):
            print(f"Warning: Optimization failed - {lp_result.get('message', 'Unknown error')}")

    def get_power_profile(
        self,
        asset_name: str,
        return_format: str = 'dict'
    ) -> Dict[str, Any] | Any:
        """
        Extract power dispatch time-series for a specific asset.

        Parses LP variable names following convention: '{asset_name}_{var}_{t}'
        where var is one of: P_charge, P_discharge, P_import, P_export.

        Args:
            asset_name: Name of asset to extract (must exist in self.assets)
            return_format: 'dict' or 'dataframe' (if pandas available)

        Returns:
            If return_format='dict':
                {
                    'timesteps': [0, 1, 2, ...],
                    'P_charge': [values...],      # For batteries
                    'P_discharge': [values...],   # For batteries
                    'P_import': [values...],      # For markets
                    'P_export': [values...],      # For markets
                    'P_net': [values...],         # Net power contribution
                }

            If return_format='dataframe':
                pandas.DataFrame with columns as above, index=timesteps

        Raises:
            ValueError: If asset_name not found
            ImportError: If return_format='dataframe' but pandas not available
        """
        if asset_name not in self.assets:
            raise ValueError(f"Asset '{asset_name}' not found in result. Available: {list(self.assets.keys())}")

        if return_format == 'dataframe' and not PANDAS_AVAILABLE:
            raise ImportError("pandas required for return_format='dataframe'. Install with: pip install pandas")

        # Get solution for this asset
        solution = self.lp_result['solution'].get(asset_name, {})

        # Initialize data structure
        data = {
            'timesteps': list(range(self.n_timesteps)),
            'P_charge': [],
            'P_discharge': [],
            'P_import': [],
            'P_export': [],
            'P_net': [],
        }

        # Extract power values for each timestep
        for t in range(self.n_timesteps):
            # Battery variables
            P_charge = solution.get(f'{asset_name}_P_charge_{t}', 0.0)
            P_discharge = solution.get(f'{asset_name}_P_discharge_{t}', 0.0)
            data['P_charge'].append(P_charge)
            data['P_discharge'].append(P_discharge)

            # Market variables
            P_import = solution.get(f'{asset_name}_P_import_{t}', 0.0)
            P_export = solution.get(f'{asset_name}_P_export_{t}', 0.0)
            data['P_import'].append(P_import)
            data['P_export'].append(P_export)

            # Net power contribution
            # Battery: discharge is positive (injecting), charge is negative (drawing)
            # Market: import is positive (supplying), export is negative (absorbing)
            P_net = (P_discharge - P_charge) + (P_import - P_export)
            data['P_net'].append(P_net)

        if return_format == 'dataframe':
            return pd.DataFrame(data).set_index('timesteps')
        else:
            return data

    def get_soc_profile(
        self,
        battery_name: str,
        return_format: str = 'dict'
    ) -> Dict[str, Any] | Any:
        """
        Extract state of charge (SOC) evolution for a battery.

        Parses LP variable names: '{battery_name}_E_{t}' for stored energy.

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
        """
        if battery_name not in self.assets:
            raise ValueError(f"Battery '{battery_name}' not found. Available: {list(self.assets.keys())}")

        battery = self.assets[battery_name]
        if not hasattr(battery, 'unit') or not hasattr(battery.unit, 'C_spec'):
            raise ValueError(f"Asset '{battery_name}' is not a battery (missing C_spec)")

        if return_format == 'dataframe' and not PANDAS_AVAILABLE:
            raise ImportError("pandas required for return_format='dataframe'")

        solution = self.lp_result['solution'].get(battery_name, {})

        # Extract SOC values (including initial state at t=0)
        # Battery variables are: E_0, E_1, ..., E_(T-1) representing stored energy [kWh]
        data = {
            'timesteps': list(range(self.n_timesteps + 1)),
            'SOC': [],
            'SOC_percent': [],
        }

        capacity = battery.unit.C_spec
        efficiency = battery.unit.efficiency

        # Get E values from LP solution (stored energy at beginning of each timestep)
        for t in range(self.n_timesteps):
            e_kwh = solution.get(f'{battery_name}_E_{t}', 0.0)
            soc_percent = (e_kwh / capacity * 100) if capacity > 0 else 0.0

            data['SOC'].append(e_kwh)
            data['SOC_percent'].append(soc_percent)

        # Calculate final SOC after last timestep
        # E[T] = E[T-1] + (P_charge[T-1] * eff - P_discharge[T-1] / eff) * dt
        e_final = solution.get(f'{battery_name}_E_{self.n_timesteps-1}', 0.0)
        p_charge_final = solution.get(f'{battery_name}_P_charge_{self.n_timesteps-1}', 0.0)
        p_discharge_final = solution.get(f'{battery_name}_P_discharge_{self.n_timesteps-1}', 0.0)

        e_final += (p_charge_final * efficiency - p_discharge_final / efficiency) * DT_HOURS
        soc_final_percent = (e_final / capacity * 100) if capacity > 0 else 0.0

        data['SOC'].append(e_final)
        data['SOC_percent'].append(soc_final_percent)

        if return_format == 'dataframe':
            return pd.DataFrame(data).set_index('timesteps')
        else:
            return data

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
            For MVP, this returns simplified breakdown based on total cost.
            Future enhancement: Extract detailed cost components from solver.
        """
        breakdown = {
            'total_cost': self.get_total_cost(),
            'by_asset': {}
        }

        # For each asset, get metrics if available
        for asset_name, asset in self.assets.items():
            if hasattr(asset, 'get_metrics'):
                metrics = asset.get_metrics()
                asset_cost = metrics.get('total_cost_eur', 0.0)

                breakdown['by_asset'][asset_name] = {
                    'total': asset_cost,
                    'throughput_kwh': metrics.get('total_throughput_kwh', 0.0),
                    'activations': metrics.get('num_activations', 0),
                }

        return breakdown

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
        """
        if asset_name not in self.assets:
            raise ValueError(f"Asset '{asset_name}' not found")

        asset = self.assets[asset_name]
        solution = self.lp_result['solution'].get(asset_name, {})

        metrics = {}

        # Extract power profile
        power_data = self.get_power_profile(asset_name, return_format='dict')

        # Capacity factor (for batteries)
        if hasattr(asset, 'unit') and hasattr(asset.unit, 'power_kw'):
            max_power = asset.unit.power_kw
            total_hours = self.n_timesteps * DT_HOURS

            # Calculate actual throughput
            throughput = sum(power_data['P_charge']) + sum(power_data['P_discharge'])
            throughput += sum(power_data['P_import']) + sum(power_data['P_export'])
            throughput *= DT_HOURS  # Convert to energy

            max_throughput = max_power * total_hours
            metrics['capacity_factor'] = throughput / max_throughput if max_throughput > 0 else 0.0

            # Peak power used
            max_power_used = max(
                max(power_data['P_charge'] or [0]),
                max(power_data['P_discharge'] or [0]),
                max(power_data['P_import'] or [0]),
                max(power_data['P_export'] or [0])
            )
            metrics['max_power_used'] = max_power_used
        else:
            metrics['capacity_factor'] = 0.0
            metrics['max_power_used'] = 0.0

        # Utilization hours (timesteps with non-zero activation)
        active_timesteps = sum(
            1 for t in range(self.n_timesteps)
            if abs(power_data['P_net'][t]) > 1e-6
        )
        metrics['utilization_hours'] = active_timesteps * DT_HOURS

        # Cycle counting (for batteries)
        if hasattr(asset, 'unit') and hasattr(asset.unit, 'C_spec'):
            capacity = asset.unit.C_spec
            total_discharge = sum(power_data['P_discharge']) * DT_HOURS
            metrics['num_cycles'] = total_discharge / capacity if capacity > 0 else 0.0

            # Average cycle depth (simplified: total throughput / (2 * num_cycles * capacity))
            if metrics['num_cycles'] > 0:
                total_throughput = (sum(power_data['P_charge']) + sum(power_data['P_discharge'])) * DT_HOURS
                metrics['avg_cycle_depth'] = (total_throughput / (2 * metrics['num_cycles'] * capacity)) * 100
            else:
                metrics['avg_cycle_depth'] = 0.0
        else:
            metrics['num_cycles'] = 0.0
            metrics['avg_cycle_depth'] = 0.0

        return metrics

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
        """
        if return_format == 'dataframe' and not PANDAS_AVAILABLE:
            raise ImportError("pandas required for return_format='dataframe'")

        data = {
            'timesteps': list(range(self.n_timesteps)),
            'imbalance': [self.imbalance.get(t, 0.0) for t in range(self.n_timesteps)]
        }

        if return_format == 'dataframe':
            return pd.DataFrame(data).set_index('timesteps')
        else:
            return data

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of optimization results.

        Returns:
            Dictionary with key metrics across all assets.
        """
        summary = {
            'optimization': {
                'success': self.is_successful(),
                'total_cost': self.get_total_cost(),
                'message': self.get_status_message(),
                'n_timesteps': self.n_timesteps,
                'dt_hours': DT_HOURS,
                'total_hours': self.n_timesteps * DT_HOURS,
            },
            'assets': {},
            'imbalance': {
                'total_energy': sum(self.imbalance.values()) * DT_HOURS,
                'peak_deficit': max(self.imbalance.values()) if self.imbalance else 0.0,
                'peak_surplus': min(self.imbalance.values()) if self.imbalance else 0.0,
            }
        }

        # Add asset-specific summaries
        for asset_name in self.assets:
            try:
                summary['assets'][asset_name] = self.get_utilization_metrics(asset_name)
            except Exception as e:
                summary['assets'][asset_name] = {'error': str(e)}

        return summary

    def get_total_cost(self) -> float:
        """
        Get the total cost (objective value) from LP optimization.

        Returns:
            Total cost [EUR] from LP solver objective function.
        """
        return self.lp_result.get('cost', 0.0)

    def is_successful(self) -> bool:
        """
        Check if LP optimization completed successfully.

        Returns:
            True if LP solver found optimal/feasible solution, False otherwise.
        """
        return self.lp_result.get('success', False)

    def get_status_message(self) -> str:
        """
        Get human-readable status message from LP solver.

        Returns:
            Status message string from LP solver (e.g., "Optimal solution found").
        """
        return self.lp_result.get('message', 'No status message available')

    def __repr__(self) -> str:
        """String representation of optimization result."""
        status = "SUCCESS" if self.is_successful() else "FAILED"
        cost = self.get_total_cost()
        n_assets = len(self.assets)
        return (f"LPOptimizationResult(status={status}, cost={cost:.2f} EUR, "
                f"n_assets={n_assets}, n_timesteps={self.n_timesteps})")