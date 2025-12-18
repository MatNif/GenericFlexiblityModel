"""
Battery vs. Imbalance Market - Linear Programming Optimization.

Compares two strategies for handling energy imbalances:
    A. Battery + Market: Use LP to find optimal battery scheduling
    B. Pure Market: Settle all imbalances directly with TSO

Uses the modular LP optimization framework where FlexAssets are automatically
converted to linear models and aggregated.
"""

from pathlib import Path
from typing import Dict
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from flex_model.assets import BatteryUnit, BatteryCostModel, BatteryFlex
from flex_model.assets import BalancingMarketCost, BalancingMarketFlex
from flex_model.optimization import LPOptimizer
from flex_model.settings import DT_HOURS
from utils.data_loader import load_imbalance_prices, load_imbalance_profile, get_data_path


def run_scenario():
    """Run battery vs. market comparison with LP optimization."""

    print("=" * 70)
    print("BATTERY VS. IMBALANCE MARKET - LP OPTIMIZATION")
    print("=" * 70)

    # ========================================================================
    # 1. Load Data
    # ========================================================================
    print("\n[1/5] Loading data...")
    data_path = get_data_path()

    # Load imbalance prices
    prices_file = data_path / 'imbalance_prices.csv'
    if not prices_file.exists():
        print(f"ERROR: Price data not found at {prices_file}")
        print("Please add Swissgrid data to examples/battery_vs_market/data/")
        return

    p_buy, p_sell = load_imbalance_prices(str(prices_file))
    print(f"  > Loaded {len(p_buy)} timesteps of imbalance prices")

    # Load imbalance profile
    profile_file = data_path / 'imbalance_profile.csv'
    if not profile_file.exists():
        print(f"ERROR: Imbalance profile not found at {profile_file}")
        print("Run generate_dummy_profile.py first")
        return

    imbalance = load_imbalance_profile(str(profile_file))
    print(f"  > Loaded {len(imbalance)} timesteps of imbalance profile")
    print(f"  > Time horizon: {len(imbalance) / 96:.1f} days (15-min resolution)")

    # ========================================================================
    # 2. Configure Assets
    # ========================================================================
    print("\n[2/5] Configuring assets...")

    # Battery configuration
    BATTERY_CAPACITY_KWH = 100.0
    BATTERY_POWER_KW = 50.0
    BATTERY_EFFICIENCY = 0.95
    BATTERY_INV_COST_CHF_PER_KWH = 500.0
    BATTERY_LIFETIME_YEARS = 10.0
    BATTERY_DEGRADATION_CHF_PER_KWH = 0.05
    INITIAL_SOC = 0.5

    print(f"\n  Battery:")
    print(f"    Capacity:    {BATTERY_CAPACITY_KWH} kWh")
    print(f"    Power:       {BATTERY_POWER_KW} kW")
    print(f"    Efficiency:  {BATTERY_EFFICIENCY:.1%}")
    print(f"    Investment:  {BATTERY_INV_COST_CHF_PER_KWH} CHF/kWh")
    print(f"    Lifetime:    {BATTERY_LIFETIME_YEARS} years")
    print(f"    Initial SOC: {INITIAL_SOC:.0%}")

    # Create battery physical unit
    battery_unit = BatteryUnit(
        name="BESS_100kWh",
        capacity_kwh=BATTERY_CAPACITY_KWH,
        power_kw=BATTERY_POWER_KW,
        efficiency=BATTERY_EFFICIENCY,
        soc_min=0.1,
        soc_max=0.9,
    )

    # Create battery cost model
    battery_cost = BatteryCostModel(
        name="battery_economics",
        c_inv=BATTERY_INV_COST_CHF_PER_KWH,
        n_lifetime=BATTERY_LIFETIME_YEARS,
        p_int=BATTERY_DEGRADATION_CHF_PER_KWH,
    )

    # Create battery flex asset
    battery = BatteryFlex(unit=battery_unit, cost_model=battery_cost)

    # Create market cost model
    market_cost = BalancingMarketCost(
        name="imbalance_market",
        p_E_buy=p_buy,
        p_E_sell=p_sell,
    )

    # Create market flex asset
    market = BalancingMarketFlex(cost_model=market_cost)

    print(f"\n  Market:")
    print(f"    No investment cost")
    print(f"    No capacity limits")
    print(f"    Same imbalance prices as battery scenario")

    # ========================================================================
    # 3. Run Scenario A: LP-Optimized Battery + Market
    # ========================================================================
    print("\n[3/5] Running Scenario A: LP-Optimized Battery + Market...")

    # Convert assets to linear models
    print("  > Converting assets to linear models...")
    n_timesteps = len(imbalance)
    battery_lm = battery.get_linear_model(
        n_timesteps=n_timesteps,
        dt_hours=DT_HOURS,
        initial_soc=INITIAL_SOC
    )
    market_lm = market.get_linear_model(
        n_timesteps=n_timesteps,
        dt_hours=DT_HOURS
    )

    # Create optimizer and add assets
    print("  > Building LP optimization problem...")
    optimizer = LPOptimizer(n_timesteps=n_timesteps)
    optimizer.add_asset(battery_lm)
    optimizer.add_asset(market_lm)
    optimizer.set_imbalance(imbalance)

    # Solve
    print("  > Solving LP...")
    result = optimizer.solve()

    if not result['success']:
        print(f"  ERROR: Optimization failed: {result['message']}")
        return

    operational_cost = result['cost']

    # Calculate battery activations from solution
    battery_solution = result['solution']['BESS_100kWh']
    battery_activations = 0
    for t in range(n_timesteps):
        P_charge = battery_solution[f'BESS_100kWh_P_charge_{t}']
        P_discharge = battery_solution[f'BESS_100kWh_P_discharge_{t}']
        if P_charge > 1e-6 or P_discharge > 1e-6:
            battery_activations += 1

    # Calculate final SOC
    final_E = battery_solution[f'BESS_100kWh_E_{n_timesteps-1}']
    final_soc = final_E / BATTERY_CAPACITY_KWH

    print(f"  > LP solved successfully")
    print(f"  > Battery activations: {battery_activations}")
    print(f"  > Final SOC: {final_soc:.1%}")

    # Add annualized investment cost
    investment_annual = battery_cost.annualized_investment(
        capacity=BATTERY_CAPACITY_KWH,
        discount_rate=0.05
    )

    days_in_scenario = len(imbalance) / 96
    investment_scenario = investment_annual * (days_in_scenario / 365)
    cost_battery_scenario = operational_cost + investment_scenario

    # ========================================================================
    # 4. Run Scenario B: Pure Market
    # ========================================================================
    print("\n[4/5] Running Scenario B: Pure Market...")

    market.reset()
    cost_market_scenario = 0.0

    for t in range(len(imbalance)):
        imb_kw = imbalance[t]

        if imb_kw > 0:
            result = market.evaluate_operation(t=t, dt_hours=DT_HOURS, P_grid_import=imb_kw, P_grid_export=0)
            market.execute_operation(t=t, dt_hours=DT_HOURS, P_grid_import=imb_kw, P_grid_export=0)
            cost_market_scenario += result['cost']
        else:
            result = market.evaluate_operation(t=t, dt_hours=DT_HOURS, P_grid_import=0, P_grid_export=abs(imb_kw))
            market.execute_operation(t=t, dt_hours=DT_HOURS, P_grid_import=0, P_grid_export=abs(imb_kw))
            cost_market_scenario += result['cost']

    print(f"  > All imbalances settled with TSO")

    # ========================================================================
    # 5. Compare Results
    # ========================================================================
    print("\n[5/5] Results:")
    print("\n" + "=" * 70)
    print("COST COMPARISON")
    print("=" * 70)

    print(f"\nScenario A - LP-Optimized Battery + Market:")
    print(f"  Investment (annualized):      {investment_scenario:10.2f} CHF")
    print(f"  Operational cost:             {operational_cost:10.2f} CHF")
    print(f"  TOTAL:                        {cost_battery_scenario:10.2f} CHF")

    print(f"\nScenario B - Pure Market:")
    print(f"  Investment:                   {0.0:10.2f} CHF")
    print(f"  Imbalance settlement:         {cost_market_scenario:10.2f} CHF")
    print(f"  TOTAL:                        {cost_market_scenario:10.2f} CHF")

    savings = cost_market_scenario - cost_battery_scenario
    savings_pct = 100 * savings / cost_market_scenario if cost_market_scenario != 0 else 0

    print(f"\n{'=' * 70}")
    if savings > 0:
        print(f"SAVINGS WITH BATTERY: {savings:.2f} CHF ({savings_pct:.1f}%)")

        # Calculate break-even investment
        operational_savings = cost_market_scenario - operational_cost
        breakeven_investment_total = operational_savings * BATTERY_LIFETIME_YEARS * (365 / days_in_scenario)
        breakeven_per_kwh = breakeven_investment_total / BATTERY_CAPACITY_KWH

        print(f"\nBreak-even analysis (over {days_in_scenario:.0f} days):")
        print(f"  Operational savings:          {operational_savings:.2f} CHF")
        print(f"  Annualized:                   {operational_savings * (365/days_in_scenario):.2f} CHF/year")
        print(f"  Break-even investment:        {breakeven_investment_total:.2f} CHF ({breakeven_per_kwh:.2f} CHF/kWh)")
        print(f"  Actual investment:            {BATTERY_INV_COST_CHF_PER_KWH * BATTERY_CAPACITY_KWH:.2f} CHF ({BATTERY_INV_COST_CHF_PER_KWH:.2f} CHF/kWh)")

        if breakeven_per_kwh > BATTERY_INV_COST_CHF_PER_KWH:
            print(f"\n>> RECOMMENDATION: Battery investment is economical")
        else:
            print(f"\n>> RECOMMENDATION: Battery too expensive (needs < {breakeven_per_kwh:.2f} CHF/kWh)")
    else:
        print(f"EXTRA COST WITH BATTERY: {-savings:.2f} CHF ({-savings_pct:.1f}%)")
        print(f"\n>> RECOMMENDATION: Battery not economical for this scenario")

    print("=" * 70)


if __name__ == "__main__":
    run_scenario()