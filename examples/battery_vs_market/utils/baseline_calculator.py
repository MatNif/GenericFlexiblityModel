"""
Baseline cost calculation utilities.

This module provides functions to calculate baseline costs for optimization
scenarios, representing the "do nothing" or "pure market settlement" case
without any FlexAssets.
"""

from typing import Dict, Tuple
from pathlib import Path
import sys
from flex_model.settings import DT_HOURS

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from flex_model.assets import BalancingMarketCost, BalancingMarketFlex


def calculate_baseline_cost(
    imbalance: Dict[int, float],
    p_buy: Dict[int, float],
    p_sell: Dict[int, float],
) -> Tuple[float, float]:
    """
    Calculate pure market settlement baseline cost without FlexAssets.

    This represents the "do nothing" scenario where all imbalances are settled
    directly through the market without any optimization or battery usage.

    Args:
        imbalance:
            Dict mapping timestep -> imbalance power [kW].
            Positive = deficit (need to import from market).
            Negative = surplus (can export to market).

        p_buy:
            Dict mapping timestep -> buy price [CHF/kWh].
            Price for importing energy from the market.

        p_sell:
            Dict mapping timestep -> sell price [CHF/kWh].
            Price for exporting energy to the market.

    Returns:
        Tuple of (baseline_cost, baseline_cost_annual):
            - baseline_cost: Total cost for the simulation period [CHF]
            - baseline_cost_annual: Annualized cost assuming 365 days/year [CHF/year]

    Example:
        >>> p_buy = {0: 0.10, 1: 0.12, 2: 0.11}
        >>> p_sell = {0: 0.08, 1: 0.09, 2: 0.07}
        >>> imbalance = {0: 100.0, 1: -50.0, 2: 75.0}  # kW
        >>> baseline, annual = calculate_baseline_cost(imbalance, p_buy, p_sell)
        >>> print(f"Baseline: {baseline:.2f} CHF")
        >>> print(f"Annual: {annual:.2f} CHF/year")
    """
    # Create market asset for baseline calculation
    market_cost = BalancingMarketCost(
        name="baseline_market",
        p_E_buy=p_buy,
        p_E_sell=p_sell,
    )

    market_baseline = BalancingMarketFlex(cost_model=market_cost)
    market_baseline.reset()

    # Calculate baseline cost by settling all imbalances through market
    baseline_cost = 0.0

    for t in range(len(imbalance)):
        imb_kw = imbalance[t]

        if imb_kw > 0:
            # Deficit: Need to import from market
            result = market_baseline.evaluate_operation(
                t=t,
                P_grid_import=imb_kw,
                P_grid_export=0
            )
            market_baseline.execute_operation(
                t=t,
                P_grid_import=imb_kw,
                P_grid_export=0
            )
            baseline_cost += result['cost']
        else:
            # Surplus: Can export to market
            result = market_baseline.evaluate_operation(
                t=t,
                P_grid_import=0,
                P_grid_export=abs(imb_kw)
            )
            market_baseline.execute_operation(
                t=t,
                P_grid_import=0,
                P_grid_export=abs(imb_kw)
            )
            baseline_cost += result['cost']

    # Annualize baseline cost
    # Assumes timestep resolution (DT_HOURS) and 96 timesteps/day (15-min resolution)
    timesteps_per_day = 24 / DT_HOURS  # e.g., 96 for 15-min timesteps
    days_in_scenario = len(imbalance) / timesteps_per_day
    baseline_cost_annual = baseline_cost * (365 / days_in_scenario)

    return baseline_cost, baseline_cost_annual