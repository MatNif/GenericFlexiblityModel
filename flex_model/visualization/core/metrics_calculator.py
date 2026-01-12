"""
EconomicMetrics: Calculator for financial KPIs from optimization results.

This module provides methods for computing return on investment (ROI),
payback periods, net present value (NPV), and other economic indicators
to support business decision-making.
"""

from __future__ import annotations

from typing import Dict, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from flex_model.visualization.core.result_processor import OptimizationResult


class EconomicMetrics:
    """
    Calculator for economic and financial metrics.

    This class provides static methods for computing key performance indicators
    (KPIs) from optimization results, enabling comparison of different FlexAsset
    investment scenarios.

    All methods assume costs are in EUR and time periods in years unless
    otherwise specified.
    """

    @staticmethod
    def compute_roi(
        result: 'OptimizationResult',
        baseline_cost: float,
        lifetime_years: float = 10.0,
    ) -> float:
        """
        Calculate return on investment (ROI) as a percentage.

        ROI measures the profitability of the investment relative to the cost.

        Formula:
            ROI = (Total Savings / Total Investment) * 100

        Args:
            result:
                OptimizationResult instance with optimization outcome.

            baseline_cost:
                Annual cost without FlexAssets (pure market settlement) [EUR/year].
                This is the reference scenario for comparison.

            lifetime_years:
                Expected lifetime of assets [years]. Default: 10 years.

        Returns:
            ROI as percentage. Positive = profitable, Negative = loss.

        Example:
            >>> roi = EconomicMetrics.compute_roi(result, baseline_cost=50000, lifetime_years=10)
            >>> print(f"ROI: {roi:.1f}%")  # e.g., "ROI: 23.5%"
        """
        # Extract investment costs from assets
        total_investment = 0.0
        for asset_name, asset in result.assets.items():
            if hasattr(asset, 'cost_model') and hasattr(asset.cost_model, 'c_inv'):
                # Investment cost = c_inv [EUR/kWh] * capacity [kWh]
                if hasattr(asset, 'unit') and hasattr(asset.unit, 'C_spec'):
                    capacity = asset.unit.C_spec
                    c_inv = asset.cost_model.c_inv
                    total_investment += c_inv * capacity

        # Calculate annual operational cost with FlexAssets
        # Scale optimization cost to annual (optimization may be for shorter period)
        optimization_hours = result.n_timesteps * result.dt_hours
        hours_per_year = 8760
        annual_optimized_cost = result.get_total_cost() * (hours_per_year / optimization_hours)

        # Total savings over lifetime
        annual_savings = baseline_cost - annual_optimized_cost
        total_savings = annual_savings * lifetime_years

        # ROI = (Savings - Investment) / Investment * 100
        # Note: If investment is 0 (e.g., only market settlement), ROI is undefined
        if total_investment == 0:
            return float('inf') if total_savings > 0 else 0.0

        roi = ((total_savings - total_investment) / total_investment) * 100

        return roi

    @staticmethod
    def compute_payback_period(
        result: 'OptimizationResult',
        baseline_cost: float,
    ) -> float:
        """
        Calculate simple payback period in years.

        Payback period is the time required to recover the initial investment
        through annual savings.

        Formula:
            Payback Period = Total Investment / Annual Savings

        Args:
            result:
                OptimizationResult instance.

            baseline_cost:
                Annual baseline cost without FlexAssets [EUR/year].

        Returns:
            Payback period [years]. Returns inf if savings are zero or negative.

        Example:
            >>> payback = EconomicMetrics.compute_payback_period(result, baseline_cost=50000)
            >>> print(f"Payback: {payback:.1f} years")  # e.g., "Payback: 4.2 years"
        """
        # Extract total investment
        total_investment = 0.0
        for asset_name, asset in result.assets.items():
            if hasattr(asset, 'cost_model') and hasattr(asset.cost_model, 'c_inv'):
                if hasattr(asset, 'unit') and hasattr(asset.unit, 'C_spec'):
                    capacity = asset.unit.C_spec
                    c_inv = asset.cost_model.c_inv
                    total_investment += c_inv * capacity

        # Calculate annual savings
        optimization_hours = result.n_timesteps * result.dt_hours
        hours_per_year = 8760
        annual_optimized_cost = result.get_total_cost() * (hours_per_year / optimization_hours)
        annual_savings = baseline_cost - annual_optimized_cost

        # Payback period
        if annual_savings <= 0:
            return float('inf')  # Never pays back

        payback_years = total_investment / annual_savings

        return payback_years

    @staticmethod
    def compute_npv(
        result: 'OptimizationResult',
        baseline_cost: float,
        lifetime_years: float = 10.0,
        discount_rate: float = 0.05,
    ) -> float:
        """
        Calculate net present value (NPV) of the investment.

        NPV accounts for the time value of money by discounting future cash flows
        to present value. A positive NPV indicates a profitable investment.

        Formula:
            NPV = -Investment + Σ(Annual_Savings / (1 + r)^t) for t=1 to lifetime

        Args:
            result:
                OptimizationResult instance.

            baseline_cost:
                Annual baseline cost [EUR/year].

            lifetime_years:
                Asset lifetime [years]. Default: 10 years.

            discount_rate:
                Annual discount rate (e.g., 0.05 = 5%). Default: 0.05.

        Returns:
            NPV [EUR]. Positive = profitable investment.

        Example:
            >>> npv = EconomicMetrics.compute_npv(result, baseline_cost=50000, discount_rate=0.05)
            >>> if npv > 0:
            >>>     print(f"Profitable! NPV = {npv:,.0f} EUR")
        """
        # Extract total investment (upfront cost at t=0)
        total_investment = 0.0
        for asset_name, asset in result.assets.items():
            if hasattr(asset, 'cost_model') and hasattr(asset.cost_model, 'c_inv'):
                if hasattr(asset, 'unit') and hasattr(asset.unit, 'C_spec'):
                    capacity = asset.unit.C_spec
                    c_inv = asset.cost_model.c_inv
                    total_investment += c_inv * capacity

        # Calculate annual savings
        optimization_hours = result.n_timesteps * result.dt_hours
        hours_per_year = 8760
        annual_optimized_cost = result.get_total_cost() * (hours_per_year / optimization_hours)
        annual_savings = baseline_cost - annual_optimized_cost

        # Calculate NPV: -Investment + PV(savings stream)
        pv_savings = 0.0
        for year in range(1, int(lifetime_years) + 1):
            discount_factor = (1 + discount_rate) ** year
            pv_savings += annual_savings / discount_factor

        npv = -total_investment + pv_savings

        return npv

    @staticmethod
    def compute_lcoe(
        result: 'OptimizationResult',
        lifetime_years: float = 10.0,
        discount_rate: float = 0.05,
    ) -> float:
        """
        Calculate levelized cost of energy (LCOE).

        LCOE represents the average cost per kWh of energy throughput over the
        asset lifetime, accounting for investment and operational costs.

        Formula:
            LCOE = (Investment + PV(Annual Costs)) / PV(Annual Throughput)

        Args:
            result:
                OptimizationResult instance.

            lifetime_years:
                Asset lifetime [years].

            discount_rate:
                Annual discount rate.

        Returns:
            LCOE [EUR/kWh].

        Note:
            This is useful for comparing different technologies or configurations
            on a normalized cost-per-energy basis.
        """
        # Extract total investment
        total_investment = 0.0
        for asset_name, asset in result.assets.items():
            if hasattr(asset, 'cost_model') and hasattr(asset.cost_model, 'c_inv'):
                if hasattr(asset, 'unit') and hasattr(asset.unit, 'C_spec'):
                    capacity = asset.unit.C_spec
                    c_inv = asset.cost_model.c_inv
                    total_investment += c_inv * capacity

        # Calculate annual operational cost and throughput
        optimization_hours = result.n_timesteps * result.dt_hours
        hours_per_year = 8760
        annual_cost = result.get_total_cost() * (hours_per_year / optimization_hours)

        # Total throughput from all assets
        total_throughput = 0.0
        for asset_name, asset in result.assets.items():
            if hasattr(asset, 'get_metrics'):
                metrics = asset.get_metrics()
                throughput = metrics.get('total_throughput_kwh', 0.0)
                # Scale to annual
                total_throughput += throughput * (hours_per_year / optimization_hours)

        # Present value of costs
        pv_costs = total_investment
        for year in range(1, int(lifetime_years) + 1):
            discount_factor = (1 + discount_rate) ** year
            pv_costs += annual_cost / discount_factor

        # Present value of throughput
        pv_throughput = 0.0
        for year in range(1, int(lifetime_years) + 1):
            discount_factor = (1 + discount_rate) ** year
            pv_throughput += total_throughput / discount_factor

        # LCOE
        if pv_throughput == 0:
            return float('inf')

        lcoe = pv_costs / pv_throughput

        return lcoe

    @staticmethod
    def compute_savings_breakdown(
        result: 'OptimizationResult',
        baseline_cost: float,
    ) -> Dict[str, float]:
        """
        Calculate detailed savings breakdown vs baseline.

        Args:
            result:
                OptimizationResult instance.

            baseline_cost:
                Annual baseline cost [EUR/year].

        Returns:
            Dictionary with breakdown:
                {
                    'baseline_cost': float,           # Annual baseline
                    'optimized_cost': float,          # Annual with FlexAssets
                    'absolute_savings': float,        # EUR/year saved
                    'relative_savings': float,        # % saved
                    'investment_required': float,     # Upfront CAPEX
                }
        """
        # Extract investment
        total_investment = 0.0
        for asset_name, asset in result.assets.items():
            if hasattr(asset, 'cost_model') and hasattr(asset.cost_model, 'c_inv'):
                if hasattr(asset, 'unit') and hasattr(asset.unit, 'C_spec'):
                    capacity = asset.unit.C_spec
                    c_inv = asset.cost_model.c_inv
                    total_investment += c_inv * capacity

        # Calculate annual optimized cost
        optimization_hours = result.n_timesteps * result.dt_hours
        hours_per_year = 8760
        annual_optimized_cost = result.get_total_cost() * (hours_per_year / optimization_hours)

        # Savings
        absolute_savings = baseline_cost - annual_optimized_cost
        relative_savings = (absolute_savings / baseline_cost * 100) if baseline_cost > 0 else 0.0

        return {
            'baseline_cost': baseline_cost,
            'optimized_cost': annual_optimized_cost,
            'absolute_savings': absolute_savings,
            'relative_savings': relative_savings,
            'investment_required': total_investment,
        }

    @staticmethod
    def compute_capacity_factor(
        result: 'OptimizationResult',
        asset_name: str,
    ) -> float:
        """
        Calculate capacity factor for a specific asset.

        Capacity factor measures how much of the asset's potential capacity
        is actually utilized.

        Formula:
            Capacity Factor = Actual Throughput / Maximum Possible Throughput

        Args:
            result:
                OptimizationResult instance.

            asset_name:
                Name of asset to analyze.

        Returns:
            Capacity factor [0-1]. 1.0 = 100% utilization.

        Example:
            >>> cf = EconomicMetrics.compute_capacity_factor(result, 'battery')
            >>> print(f"Capacity factor: {cf:.1%}")  # e.g., "Capacity factor: 42.3%"
        """
        metrics = result.get_utilization_metrics(asset_name)
        return metrics.get('capacity_factor', 0.0)

    @staticmethod
    def compute_irr(
        result: 'OptimizationResult',
        baseline_cost: float,
        lifetime_years: float = 10.0,
    ) -> float:
        """
        Calculate Internal Rate of Return (IRR).

        IRR is the discount rate that makes NPV = 0. It represents the annual
        return rate of the investment, making it easy to compare with other
        investment opportunities (stocks, bonds, etc.).

        Args:
            result:
                OptimizationResult instance.

            baseline_cost:
                Annual baseline cost [EUR/year].

            lifetime_years:
                Asset lifetime [years]. Default: 10 years.

        Returns:
            IRR as percentage (e.g., 18.2 means 18.2% annual return).
            Returns 0.0 if no positive cash flows or calculation fails.

        Example:
            >>> irr = EconomicMetrics.compute_irr(result, baseline_cost=50000, lifetime_years=10)
            >>> print(f"IRR: {irr:.1f}%")  # e.g., "IRR: 18.2%"

        Note:
            IRR accounts for the time value of money, unlike ROI which is a
            simple total return percentage. IRR is typically lower than ROI
            because it considers when savings are received.
        """
        # Extract total investment
        total_investment = 0.0
        for asset_name, asset in result.assets.items():
            if hasattr(asset, 'cost_model') and hasattr(asset.cost_model, 'c_inv'):
                if hasattr(asset, 'unit') and hasattr(asset.unit, 'C_spec'):
                    capacity = asset.unit.C_spec
                    c_inv = asset.cost_model.c_inv
                    total_investment += c_inv * capacity

        # Calculate annual savings
        optimization_hours = result.n_timesteps * result.dt_hours
        hours_per_year = 8760
        annual_optimized_cost = result.get_total_cost() * (hours_per_year / optimization_hours)
        annual_savings = baseline_cost - annual_optimized_cost

        # Edge cases
        if total_investment <= 0 or annual_savings <= 0:
            return 0.0

        # Create cash flow array: t=0 is -investment, t=1..lifetime is +savings
        cash_flows = np.array([-total_investment] + [annual_savings] * int(lifetime_years))

        # Try to use numpy_financial if available, otherwise use scipy
        try:
            import numpy_financial as npf
            irr = npf.irr(cash_flows) * 100  # Convert to percentage
            return irr if np.isfinite(irr) else 0.0
        except ImportError:
            # Fallback: Use Newton's method to find IRR
            try:
                from scipy.optimize import newton

                def npv_func(rate):
                    """NPV as function of discount rate."""
                    return np.sum(cash_flows / (1 + rate) ** np.arange(len(cash_flows)))

                # Initial guess: simple approximation
                initial_guess = annual_savings / total_investment

                # Solve for rate where NPV = 0
                irr_rate = newton(npv_func, x0=initial_guess, maxiter=100, tol=1e-6)
                return irr_rate * 100 if np.isfinite(irr_rate) else 0.0

            except Exception:
                # If all else fails, return a simple approximation
                # IRR ≈ annual savings / average investment (rough estimate)
                return (annual_savings / (total_investment / 2)) * 100

    @staticmethod
    def compute_cost_revenue_breakdown(
        result: 'OptimizationResult',
        baseline_cost: float,
        lifetime_years: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Calculate detailed cost and revenue breakdown.

        Separates total costs into components (CAPEX depreciation, O&M, market costs)
        and revenues (market exports, avoided baseline costs).

        Args:
            result:
                OptimizationResult instance.

            baseline_cost:
                Annual baseline cost [EUR/year].

            lifetime_years:
                Asset lifetime [years] for depreciation calculation.

        Returns:
            Dictionary with structure:
                {
                    'costs': {
                        'battery_depreciation': float,  # Annual CAPEX amortization
                        'battery_om': float,            # Variable O&M from throughput
                        'market_import_cost': float,    # Cost of importing from market
                    },
                    'revenues': {
                        'market_export_revenue': float, # Revenue from exporting to market
                        'avoided_baseline': float,      # Savings vs baseline
                    },
                    'net': {
                        'total_cost': float,            # Sum of all costs
                        'total_revenue': float,         # Sum of all revenues
                        'net_annual_cost': float,       # Total cost - total revenue
                    }
                }

        Example:
            >>> breakdown = EconomicMetrics.compute_cost_revenue_breakdown(result, baseline_cost=50000)
            >>> print(f"Battery depreciation: {breakdown['costs']['battery_depreciation']:,.0f} EUR/year")
        """
        # Initialize breakdown structure
        breakdown = {
            'costs': {
                'battery_depreciation': 0.0,
                'battery_om': 0.0,
                'market_import_cost': 0.0,
            },
            'revenues': {
                'market_export_revenue': 0.0,
                'avoided_baseline': 0.0,
            },
            'net': {
                'total_cost': 0.0,
                'total_revenue': 0.0,
                'net_annual_cost': 0.0,
            }
        }

        # Calculate annual scaling factor
        optimization_hours = result.n_timesteps * result.dt_hours
        hours_per_year = 8760
        annual_scale = hours_per_year / optimization_hours

        # Process each asset
        for asset_name, asset in result.assets.items():
            # Battery costs
            if hasattr(asset, 'unit') and hasattr(asset.unit, 'C_spec'):
                # Battery depreciation (annualized CAPEX)
                if hasattr(asset, 'cost_model') and hasattr(asset.cost_model, 'c_inv'):
                    capacity = asset.unit.C_spec
                    c_inv = asset.cost_model.c_inv
                    annual_depreciation = (c_inv * capacity) / lifetime_years
                    breakdown['costs']['battery_depreciation'] += annual_depreciation

                # Battery O&M (variable cost based on throughput)
                if hasattr(asset, 'cost_model') and hasattr(asset.cost_model, 'p_int'):
                    p_int = asset.cost_model.p_int(0)
                    power_profile = result.get_power_profile(asset_name)
                    # Throughput = total charge + total discharge (in kWh)
                    throughput_kwh = sum(power_profile['P_charge']) + sum(power_profile['P_discharge'])
                    annual_throughput = throughput_kwh * annual_scale
                    annual_om = p_int * annual_throughput
                    breakdown['costs']['battery_om'] += annual_om

            # Market costs and revenues
            if hasattr(asset, 'cost_model'):
                cost_model = asset.cost_model
                # Check if this is a market asset
                if hasattr(cost_model, 'p_E_buy') and hasattr(cost_model, 'p_E_sell'):
                    power_profile = result.get_power_profile(asset_name)

                    # Market import costs
                    for t in range(result.n_timesteps):
                        p_import_kwh = power_profile['P_import'][t]
                        if p_import_kwh > 0:
                            price_buy = cost_model.p_E_buy(t)
                            breakdown['costs']['market_import_cost'] += p_import_kwh * price_buy

                    # Market export revenues
                    for t in range(result.n_timesteps):
                        p_export_kwh = power_profile['P_export'][t]
                        if p_export_kwh > 0:
                            price_sell = cost_model.p_E_sell(t)
                            breakdown['revenues']['market_export_revenue'] += p_export_kwh * price_sell

                    # Scale to annual
                    breakdown['costs']['market_import_cost'] *= annual_scale
                    breakdown['revenues']['market_export_revenue'] *= annual_scale

        # Calculate avoided baseline cost
        annual_optimized_cost = result.get_total_cost() * annual_scale
        breakdown['revenues']['avoided_baseline'] = max(0, baseline_cost - annual_optimized_cost)

        # Calculate net totals
        breakdown['net']['total_cost'] = sum(breakdown['costs'].values())
        breakdown['net']['total_revenue'] = sum(breakdown['revenues'].values())
        breakdown['net']['net_annual_cost'] = breakdown['net']['total_cost'] - breakdown['net']['total_revenue']

        return breakdown

    @staticmethod
    def compute_daily_cost_profile(
        result: 'OptimizationResult',
        market_name: str = 'market',
    ) -> Dict[str, Any]:
        """
        Calculate daily cost/revenue time series with statistics.

        Aggregates timesteps into daily buckets and computes cost/revenue
        for each day, plus summary statistics for variability analysis.

        Args:
            result:
                OptimizationResult instance.

            market_name:
                Name of market asset (for price data). Default: 'market'.

        Returns:
            Dictionary with structure:
                {
                    'daily_costs': List[float],      # Cost per day [EUR/day]
                    'daily_revenues': List[float],   # Revenue per day [EUR/day]
                    'daily_net': List[float],        # Net cost per day [EUR/day]
                    'statistics': {
                        'mean': float,               # Average daily net cost
                        'std': float,                # Standard deviation
                        'min': float,                # Minimum daily cost
                        'max': float,                # Maximum daily cost
                        'median': float,             # Median daily cost
                        'p25': float,                # 25th percentile
                        'p75': float,                # 75th percentile
                    }
                }

        Example:
            >>> daily = EconomicMetrics.compute_daily_cost_profile(result, 'market')
            >>> print(f"Average daily cost: {daily['statistics']['mean']:.2f} EUR/day")
            >>> print(f"Cost varies between {daily['statistics']['min']:.0f} and {daily['statistics']['max']:.0f} EUR/day")
        """
        # Calculate timesteps per day
        timesteps_per_day = int(24.0 / result.dt_hours)
        num_days = result.n_timesteps // timesteps_per_day

        if num_days == 0:
            # Less than a day of data - return single-day stats
            num_days = 1
            timesteps_per_day = result.n_timesteps

        # Initialize daily arrays
        daily_costs = []
        daily_revenues = []

        # Get market cost model for prices
        market = result.assets.get(market_name)
        if market is None:
            # No market found - return zeros
            return {
                'daily_costs': [0.0] * num_days,
                'daily_revenues': [0.0] * num_days,
                'daily_net': [0.0] * num_days,
                'statistics': {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'median': 0.0,
                    'p25': 0.0,
                    'p75': 0.0,
                }
            }

        # Process each day
        for day in range(num_days):
            day_cost = 0.0
            day_revenue = 0.0

            start_t = day * timesteps_per_day
            end_t = min((day + 1) * timesteps_per_day, result.n_timesteps)

            # Accumulate costs and revenues for this day
            for asset_name, asset in result.assets.items():
                power_profile = result.get_power_profile(asset_name)

                # Battery O&M costs
                if hasattr(asset, 'cost_model') and hasattr(asset.cost_model, 'p_int'):
                    p_int = asset.cost_model.p_int(0)
                    for t in range(start_t, end_t):
                        # Cost of throughput
                        throughput_t = power_profile['P_charge'][t] + power_profile['P_discharge'][t]
                        day_cost += p_int * throughput_t

                # Market import costs and export revenues
                if hasattr(asset, 'cost_model'):
                    cost_model = asset.cost_model
                    if hasattr(cost_model, 'p_E_buy') and hasattr(cost_model, 'p_E_sell'):
                        for t in range(start_t, end_t):
                            # Import costs
                            p_import = power_profile['P_import'][t]
                            if p_import > 0:
                                price_buy = cost_model.p_E_buy(t)
                                day_cost += p_import * price_buy

                            # Export revenues
                            p_export = power_profile['P_export'][t]
                            if p_export > 0:
                                price_sell = cost_model.p_E_sell(t)
                                day_revenue += p_export * price_sell

            daily_costs.append(day_cost)
            daily_revenues.append(day_revenue)

        # Calculate net daily cost (cost - revenue)
        daily_net = [cost - revenue for cost, revenue in zip(daily_costs, daily_revenues)]

        # Calculate statistics
        daily_net_array = np.array(daily_net)
        statistics = {
            'mean': float(np.mean(daily_net_array)),
            'std': float(np.std(daily_net_array)),
            'min': float(np.min(daily_net_array)),
            'max': float(np.max(daily_net_array)),
            'median': float(np.median(daily_net_array)),
            'p25': float(np.percentile(daily_net_array, 25)),
            'p75': float(np.percentile(daily_net_array, 75)),
        }

        return {
            'daily_costs': daily_costs,
            'daily_revenues': daily_revenues,
            'daily_net': daily_net,
            'statistics': statistics,
        }

    @staticmethod
    def compute_investment_sensitivity(
        result: 'OptimizationResult',
        baseline_cost: float,
        lifetime_years: float = 10.0,
        discount_rate: float = 0.05,
        sensitivity_range: list = None,
    ) -> Dict[str, Any]:
        """
        Calculate how metrics change with investment cost.

        Tests different investment cost scenarios (e.g., ±50%) to understand
        how sensitive the investment decision is to capital cost uncertainty.

        Args:
            result:
                OptimizationResult instance.

            baseline_cost:
                Annual baseline cost [EUR/year].

            lifetime_years:
                Asset lifetime [years]. Default: 10 years.

            discount_rate:
                Annual discount rate. Default: 0.05.

            sensitivity_range:
                List of multipliers to test (e.g., [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]).
                Default: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0]

        Returns:
            Dictionary with structure:
                {
                    'multipliers': [0.5, 0.75, 1.0, ...],
                    'roi': [45.2, 32.1, 23.5, ...],        # ROI for each multiplier [%]
                    'irr': [28.5, 22.1, 18.2, ...],        # IRR for each multiplier [%]
                    'payback': [2.1, 3.1, 4.2, ...],       # Payback for each multiplier [years]
                    'npv': [30000, 20000, 10000, ...],     # NPV for each multiplier [EUR]
                    'breakeven_multiplier': 1.8,           # Where NPV = 0 or ROI = 0
                }

        Example:
            >>> sensitivity = EconomicMetrics.compute_investment_sensitivity(result, baseline_cost=50000)
            >>> print(f"At 50% lower cost, ROI would be {sensitivity['roi'][0]:.1f}%")
            >>> print(f"Break-even at {sensitivity['breakeven_multiplier']:.2f}x current cost")
        """
        if sensitivity_range is None:
            sensitivity_range = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0]

        # Extract current investment
        current_investment = 0.0
        for asset_name, asset in result.assets.items():
            if hasattr(asset, 'cost_model') and hasattr(asset.cost_model, 'c_inv'):
                if hasattr(asset, 'unit') and hasattr(asset.unit, 'C_spec'):
                    capacity = asset.unit.C_spec
                    c_inv = asset.cost_model.c_inv
                    current_investment += c_inv * capacity

        # Calculate annual savings (constant across all scenarios)
        optimization_hours = result.n_timesteps * result.dt_hours
        hours_per_year = 8760
        annual_optimized_cost = result.get_total_cost() * (hours_per_year / optimization_hours)
        annual_savings = baseline_cost - annual_optimized_cost

        # Initialize results arrays
        multipliers = []
        roi_values = []
        irr_values = []
        payback_values = []
        npv_values = []

        # Calculate metrics for each investment multiplier
        for multiplier in sensitivity_range:
            adjusted_investment = current_investment * multiplier

            # ROI = ((Total Savings - Investment) / Investment) * 100
            total_savings = annual_savings * lifetime_years
            if adjusted_investment > 0:
                roi = ((total_savings - adjusted_investment) / adjusted_investment) * 100
            else:
                roi = float('inf') if total_savings > 0 else 0.0

            # Payback Period = Investment / Annual Savings
            if annual_savings > 0:
                payback = adjusted_investment / annual_savings
            else:
                payback = float('inf')

            # NPV = -Investment + PV(Savings Stream)
            pv_savings = 0.0
            for year in range(1, int(lifetime_years) + 1):
                discount_factor = (1 + discount_rate) ** year
                pv_savings += annual_savings / discount_factor
            npv = -adjusted_investment + pv_savings

            # IRR: Solve for discount rate where NPV = 0
            if adjusted_investment > 0 and annual_savings > 0:
                cash_flows = np.array([-adjusted_investment] + [annual_savings] * int(lifetime_years))
                try:
                    import numpy_financial as npf
                    irr = npf.irr(cash_flows) * 100
                    irr = irr if np.isfinite(irr) else 0.0
                except ImportError:
                    try:
                        from scipy.optimize import newton
                        def npv_func(rate):
                            return np.sum(cash_flows / (1 + rate) ** np.arange(len(cash_flows)))
                        irr_rate = newton(npv_func, x0=annual_savings/adjusted_investment, maxiter=100, tol=1e-6)
                        irr = irr_rate * 100 if np.isfinite(irr_rate) else 0.0
                    except Exception:
                        irr = (annual_savings / (adjusted_investment / 2)) * 100
            else:
                irr = 0.0

            # Store results
            multipliers.append(multiplier)
            roi_values.append(roi)
            irr_values.append(irr)
            payback_values.append(payback)
            npv_values.append(npv)

        # Find break-even multiplier (where NPV = 0 or ROI = 0)
        breakeven_multiplier = None
        for i in range(len(multipliers) - 1):
            if npv_values[i] > 0 and npv_values[i+1] <= 0:
                # Linear interpolation to find exact break-even
                m1, m2 = multipliers[i], multipliers[i+1]
                npv1, npv2 = npv_values[i], npv_values[i+1]
                if npv1 != npv2:
                    breakeven_multiplier = m1 + (m2 - m1) * (0 - npv1) / (npv2 - npv1)
                else:
                    breakeven_multiplier = m1
                break

        # If not found in range, check endpoints
        if breakeven_multiplier is None:
            if npv_values[-1] > 0:
                # All scenarios are profitable, estimate break-even beyond range
                # Simple linear extrapolation
                if len(npv_values) >= 2:
                    slope = (npv_values[-1] - npv_values[-2]) / (multipliers[-1] - multipliers[-2])
                    if slope < 0:
                        breakeven_multiplier = multipliers[-1] + (0 - npv_values[-1]) / slope
                    else:
                        breakeven_multiplier = float('inf')
                else:
                    breakeven_multiplier = float('inf')
            else:
                # All scenarios are unprofitable
                breakeven_multiplier = 0.0

        return {
            'multipliers': multipliers,
            'roi': roi_values,
            'irr': irr_values,
            'payback': payback_values,
            'npv': npv_values,
            'breakeven_multiplier': breakeven_multiplier,
        }

    @staticmethod
    def compute_financial_summary(
        result: 'OptimizationResult',
        baseline_cost: float,
        lifetime_years: float = 10.0,
        discount_rate: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive financial summary for executive reporting.

        This is a convenience method that computes all key metrics at once.

        Args:
            result: OptimizationResult instance
            baseline_cost: Annual baseline cost [EUR/year]
            lifetime_years: Asset lifetime [years]
            discount_rate: Annual discount rate

        Returns:
            Dictionary with all key financial metrics:
                {
                    'roi': float,                  # Return on investment [%]
                    'irr': float,                  # Internal rate of return [%]
                    'payback_period': float,       # Years to break even
                    'npv': float,                  # Net present value [EUR]
                    'lcoe': float,                 # Levelized cost [EUR/kWh]
                    'savings': {...},              # Savings breakdown dict
                    'investment': float,           # Total upfront cost [EUR]
                }
        """
        return {
            'roi': EconomicMetrics.compute_roi(result, baseline_cost, lifetime_years),
            'irr': EconomicMetrics.compute_irr(result, baseline_cost, lifetime_years),
            'payback_period': EconomicMetrics.compute_payback_period(result, baseline_cost),
            'npv': EconomicMetrics.compute_npv(result, baseline_cost, lifetime_years, discount_rate),
            'lcoe': EconomicMetrics.compute_lcoe(result, lifetime_years, discount_rate),
            'savings': EconomicMetrics.compute_savings_breakdown(result, baseline_cost),
            'assumptions': {
                'lifetime_years': lifetime_years,
                'discount_rate': discount_rate,
                'baseline_cost': baseline_cost,
            }
        }
