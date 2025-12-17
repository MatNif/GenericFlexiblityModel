"""
Battery vs. Imbalance Market Optimization Scenario.

Compares two strategies for handling energy imbalances:
    A. Battery + Market: Use battery to minimize imbalance costs, market as backup
    B. Pure Market: Settle all imbalances directly with TSO

Uses a greedy heuristic for battery operation:
    - Charge when imbalance prices are low or negative
    - Discharge when imbalance prices are high
    - Respect physical constraints (SOC, power limits)
"""

from pathlib import Path
from typing import Dict, Tuple
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from flex_model.assets import BatteryUnit, BatteryCostModel, BatteryFlex
from flex_model.assets import BalancingMarketCost, BalancingMarketFlex
from flex_model.settings import DT_HOURS
from utils.data_loader import load_imbalance_prices, load_imbalance_profile, get_data_path


def run_scenario():
    """Run battery vs. market comparison scenario."""

    print("=" * 70)
    print("BATTERY VS. IMBALANCE MARKET - COST COMPARISON")
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

    print(f"\n  Battery:")
    print(f"    Capacity:    {BATTERY_CAPACITY_KWH} kWh")
    print(f"    Power:       {BATTERY_POWER_KW} kW")
    print(f"    Efficiency:  {BATTERY_EFFICIENCY:.1%}")
    print(f"    Investment:  {BATTERY_INV_COST_CHF_PER_KWH} CHF/kWh")
    print(f"    Lifetime:    {BATTERY_LIFETIME_YEARS} years")

    # Create battery physical unit
    battery_unit = BatteryUnit(
        name="BESS_100kWh",
        capacity_kwh=BATTERY_CAPACITY_KWH,
        power_kw=BATTERY_POWER_KW,
        efficiency=BATTERY_EFFICIENCY,
        soc_min=0.1,  # Reserve 10% minimum
        soc_max=0.9,  # Reserve 10% maximum
    )

    # Create battery cost model
    battery_cost = BatteryCostModel(
        name="battery_economics",
        c_inv=BATTERY_INV_COST_CHF_PER_KWH,
        n_lifetime=BATTERY_LIFETIME_YEARS,
        p_int=BATTERY_DEGRADATION_CHF_PER_KWH,  # Degradation cost
    )

    # Create battery flex asset
    battery = BatteryFlex(unit=battery_unit, cost_model=battery_cost)

    # Create market cost model (same prices, no investment)
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
    # 3. Run Scenario A: Battery + Market
    # ========================================================================
    print("\n[3/5] Running Scenario A: Battery + Market...")

    # Initialize battery at 50% SOC
    battery.reset(E_plus_init=BATTERY_CAPACITY_KWH * 0.4, E_minus_init=BATTERY_CAPACITY_KWH * 0.4)
    market.reset()

    # Calculate charging threshold based on round-trip efficiency
    # Charge battery if opportunity cost (sell price) is less than future buy cost
    mean_p_buy = sum(p_buy.values()) / len(p_buy)
    round_trip_efficiency = BATTERY_EFFICIENCY ** 2
    charge_threshold = mean_p_buy * round_trip_efficiency

    print(f"\n  Charging strategy:")
    print(f"    Mean buy price:           {mean_p_buy:.3f} CHF/kWh")
    print(f"    Round-trip efficiency:    {round_trip_efficiency:.1%}")
    print(f"    Charge when p_sell <      {charge_threshold:.3f} CHF/kWh")

    cost_battery_scenario = 0.0
    battery_activations = 0
    market_backup_count = 0

    for t in range(len(imbalance)):
        imb_kw = imbalance[t]

        # Greedy strategy: Try to use battery optimally
        if imb_kw > 0:
            # Positive imbalance: need to BUY energy
            # Try to discharge battery to avoid buying at high price

            # Determine how much the battery can actually discharge
            P_discharge = min(imb_kw, battery.max_discharge_power(t))
            result = battery.evaluate_operation(t=t, P_grid_import=0, P_grid_export=P_discharge)

            if result['feasible'] and P_discharge > 1e-6:
                # Battery can provide some power
                battery.execute_operation(t=t, P_grid_import=0, P_grid_export=P_discharge)
                cost_battery_scenario += result['cost']
                battery_activations += 1

                # Remaining imbalance goes to market
                imb_remaining = imb_kw - P_discharge
            else:
                # Battery can't discharge (empty or at SOC limit)
                imb_remaining = imb_kw

            if abs(imb_remaining) > 0.01:  # Small tolerance
                # Settle remaining with market
                market_result = market.evaluate_operation(t=t, dt_hours=DT_HOURS,
                                                         P_grid_import=imb_remaining, P_grid_export=0)
                market.execute_operation(t=t, dt_hours=DT_HOURS, P_grid_import=imb_remaining, P_grid_export=0)
                cost_battery_scenario += market_result['cost']
                market_backup_count += 1

        else:
            # Negative imbalance: can SELL energy
            # Store in battery if sell price is low (cheaper than buying later after losses)

            imb_abs = abs(imb_kw)

            # Charge if opportunity cost (current sell price) < future buy cost (adjusted for losses)
            if p_sell[t] < charge_threshold:
                # Determine how much the battery can actually charge
                P_charge = min(imb_abs, battery.max_charge_power(t))
                result = battery.evaluate_operation(t=t, P_grid_import=P_charge, P_grid_export=0)

                if result['feasible'] and P_charge > 1e-6:
                    # Battery can absorb some power
                    battery.execute_operation(t=t, P_grid_import=P_charge, P_grid_export=0)
                    cost_battery_scenario += result['cost']
                    battery_activations += 1

                    # Remaining excess goes to market
                    imb_remaining = -(imb_abs - P_charge)
                else:
                    # Battery can't charge (full or at SOC limit)
                    imb_remaining = imb_kw
            else:
                # Sell price too high, better to sell now than store
                imb_remaining = imb_kw

            if abs(imb_remaining) > 0.01:
                # Sell remaining to market
                market_result = market.evaluate_operation(t=t, dt_hours=DT_HOURS,
                                                         P_grid_import=0, P_grid_export=abs(imb_remaining))
                market.execute_operation(t=t, dt_hours=DT_HOURS, P_grid_import=0, P_grid_export=abs(imb_remaining))
                cost_battery_scenario += market_result['cost']
                market_backup_count += 1

    # Add annualized investment cost
    investment_annual = battery_cost.annualized_investment(
        capacity=BATTERY_CAPACITY_KWH,
        discount_rate=0.05
    )

    # Scale to scenario horizon
    days_in_scenario = len(imbalance) / 96
    investment_scenario = investment_annual * (days_in_scenario / 365)
    cost_battery_scenario += investment_scenario

    print(f"  > Battery activations: {battery_activations}")
    print(f"  > Market backup used: {market_backup_count} times")
    print(f"  > Final SOC: {battery.unit.soc():.1%}")

    # ========================================================================
    # 4. Run Scenario B: Pure Market
    # ========================================================================
    print("\n[4/5] Running Scenario B: Pure Market...")

    market.reset()
    cost_market_scenario = 0.0

    for t in range(len(imbalance)):
        imb_kw = imbalance[t]

        if imb_kw > 0:
            # Buy from market
            result = market.evaluate_operation(t=t, dt_hours=DT_HOURS, P_grid_import=imb_kw, P_grid_export=0)
            market.execute_operation(t=t, dt_hours=DT_HOURS, P_grid_import=imb_kw, P_grid_export=0)
            cost_market_scenario += result['cost']
        else:
            # Sell to market
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

    print(f"\nScenario A - Battery + Market:")
    print(f"  Investment (annualized):      {investment_scenario:10.2f} CHF")
    print(f"  Operational cost:             {cost_battery_scenario - investment_scenario:10.2f} CHF")
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
        operational_savings = (cost_market_scenario - (cost_battery_scenario - investment_scenario))
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
