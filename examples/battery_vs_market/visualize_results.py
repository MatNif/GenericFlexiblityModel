"""
Visualization Example: Battery vs. Market Optimization Results.

This script demonstrates the visualization framework for optimization results:
    - Power dispatch profiles
    - State of charge evolution
    - Cost breakdown analysis
    - Economic metrics (ROI, payback period, NPV)
    - Price signal correlation

Generates interactive HTML visualizations that can be opened in a web browser.
"""

from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from flex_model.assets import BatteryUnit, BatteryCostModel, BatteryFlex
from flex_model.assets import BalancingMarketCost, BalancingMarketFlex
from flex_model.optimization import LPOptimizer
from flex_model.settings import DT_HOURS
from examples.battery_vs_market.utils.data.data_loader import load_imbalance_prices, load_imbalance_profile, get_data_path
from utils.baseline_calculator import calculate_baseline_cost

# Import visualization framework
try:
    from flex_model.visualization import LPOptimizationResult, EconomicMetrics
    from flex_model.visualization.plots import OperationalPlots, EconomicPlots
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Visualization framework not available: {e}")
    print("Install with: pip install -e .[visualization]")
    VISUALIZATION_AVAILABLE = False


def run_optimization_with_visualization():
    """Run battery optimization and generate visualizations."""

    if not VISUALIZATION_AVAILABLE:
        print("ERROR: Visualization framework not installed")
        print("Install dependencies: pip install plotly pandas")
        return

    print("=" * 80)
    print("BATTERY OPTIMIZATION WITH VISUALIZATION")
    print("=" * 80)

    # ========================================================================
    # 1. Load Data
    # ========================================================================
    print("\n[1/6] Loading data...")
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
        print("Run utils/generate_dummy_profile.py first")
        return

    imbalance = load_imbalance_profile(str(profile_file))
    print(f"  > Loaded {len(imbalance)} timesteps of imbalance profile")
    print(f"  > Time horizon: {len(imbalance) / 96:.1f} days (15-min resolution)")

    # ========================================================================
    # 2. Configure Assets
    # ========================================================================
    print("\n[2/6] Configuring assets...")

    # Battery configuration
    BATTERY_CAPACITY_KWH = 100.0
    BATTERY_POWER_KW = 50.0
    BATTERY_EFFICIENCY = 0.95
    BATTERY_INV_COST_CHF_PER_KWH = 500.0
    BATTERY_LIFETIME_YEARS = 10.0
    BATTERY_DEGRADATION_CHF_PER_KWH = 0.05
    INITIAL_SOC = 0.5

    print(f"  Battery: {BATTERY_CAPACITY_KWH} kWh @ {BATTERY_POWER_KW} kW")
    print(f"  Investment: {BATTERY_INV_COST_CHF_PER_KWH} CHF/kWh")
    print(f"  Lifetime: {BATTERY_LIFETIME_YEARS} years")

    # Create battery
    battery_unit = BatteryUnit(
        name="BESS_100kWh",
        capacity_kwh=BATTERY_CAPACITY_KWH,
        power_kw=BATTERY_POWER_KW,
        efficiency=BATTERY_EFFICIENCY,
        soc_min=0.1,
        soc_max=0.9,
    )

    battery_cost = BatteryCostModel(
        name="battery_economics",
        c_inv=BATTERY_INV_COST_CHF_PER_KWH,
        n_lifetime=BATTERY_LIFETIME_YEARS,
        p_int=BATTERY_DEGRADATION_CHF_PER_KWH,
    )

    battery = BatteryFlex(unit=battery_unit, cost_model=battery_cost)

    # Create market
    market_cost = BalancingMarketCost(
        name="imbalance_market",
        p_E_buy=p_buy,
        p_E_sell=p_sell,
    )

    market = BalancingMarketFlex(cost_model=market_cost)

    # ========================================================================
    # 3. Run LP Optimization
    # ========================================================================
    print("\n[3/6] Running LP optimization...")

    n_timesteps = len(imbalance)

    # Convert to linear models
    battery_lm = battery.get_linear_model(
        n_timesteps=n_timesteps,
        initial_soc=INITIAL_SOC
    )
    market_lm = market.get_linear_model(
        n_timesteps=n_timesteps
    )

    # Create and solve optimizer
    optimizer = LPOptimizer(n_timesteps=n_timesteps)
    optimizer.add_asset(battery_lm)
    optimizer.add_asset(market_lm)
    optimizer.set_imbalance(imbalance)

    lp_result = optimizer.solve()

    if not lp_result['success']:
        print(f"  ERROR: Optimization failed: {lp_result['message']}")
        return

    print(f"  > Optimization successful")
    print(f"  > Total cost: {lp_result['cost']:.2f} CHF")

    # ========================================================================
    # 4. Calculate Baseline (Pure Market)
    # ========================================================================
    print("\n[4/6] Calculating baseline (pure market settlement)...")

    baseline_cost, baseline_cost_annual = calculate_baseline_cost(
        imbalance=imbalance,
        p_buy=p_buy,
        p_sell=p_sell
    )

    days_in_scenario = len(imbalance) / (24 / DT_HOURS)
    print(f"  > Baseline cost: {baseline_cost:.2f} CHF ({days_in_scenario:.1f} days)")
    print(f"  > Annualized: {baseline_cost_annual:.2f} CHF/year")

    # ========================================================================
    # 5. Wrap Results and Calculate Metrics
    # ========================================================================
    print("\n[5/6] Processing results...")

    # Wrap optimization result
    result = LPOptimizationResult(
        lp_result=lp_result,
        assets={'BESS_100kWh': battery, 'imbalance_market': market},
        imbalance=imbalance
    )

    print(f"  > {result}")

    # Calculate economic metrics
    financial_summary = EconomicMetrics.compute_financial_summary(
        result=result,
        baseline_cost=baseline_cost_annual,
        lifetime_years=BATTERY_LIFETIME_YEARS,
        discount_rate=0.05,
    )

    print(f"\n  Financial Summary:")
    print(f"    ROI:             {financial_summary['roi']:.1f}%")
    print(f"    Payback Period:  {financial_summary['payback_period']:.1f} years")
    print(f"    NPV:             {financial_summary['npv']:,.0f} EUR")
    print(f"    LCOE:            {financial_summary['lcoe']:.4f} EUR/kWh")

    # ========================================================================
    # 6. Generate Visualizations
    # ========================================================================
    print("\n[6/6] Generating visualizations...")

    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)

    # 1. Power Dispatch Profile
    print("  > Creating power dispatch profile...")
    fig_dispatch = OperationalPlots.create_dispatch_profile(result, view_mode='system')
    dispatch_file = output_dir / 'power_dispatch.html'
    fig_dispatch.write_html(str(dispatch_file))
    print(f"    Saved to: {dispatch_file}")

    # 2. SOC Evolution
    print("  > Creating SOC evolution...")
    fig_soc = OperationalPlots.create_soc_evolution(result, battery_name='BESS_100kWh')
    soc_file = output_dir / 'soc_evolution.html'
    fig_soc.write_html(str(soc_file))
    print(f"    Saved to: {soc_file}")

    # 3. Price Overlay
    print("  > Creating price overlay...")
    fig_prices = OperationalPlots.create_price_overlay(result, market_name='imbalance_market')
    prices_file = output_dir / 'price_signals.html'
    fig_prices.write_html(str(prices_file))
    print(f"    Saved to: {prices_file}")

    # 4. Cost Breakdown
    print("  > Creating cost breakdown...")
    fig_cost = EconomicPlots.create_cost_breakdown(result, breakdown_type='by_asset')
    cost_file = output_dir / 'cost_breakdown.html'
    fig_cost.write_html(str(cost_file))
    print(f"    Saved to: {cost_file}")

    # 5. Savings Comparison
    print("  > Creating savings comparison...")
    fig_savings = EconomicPlots.create_savings_comparison(
        baseline_cost=baseline_cost_annual,
        optimized_cost=financial_summary['savings']['optimized_cost'],
        investment_cost=financial_summary['savings']['investment_required'],
    )
    savings_file = output_dir / 'savings_comparison.html'
    fig_savings.write_html(str(savings_file))
    print(f"    Saved to: {savings_file}")

    # 6. ROI Gauge
    print("  > Creating ROI gauge...")
    fig_roi = EconomicPlots.create_roi_gauge(roi=financial_summary['roi'], target_roi=15.0)
    roi_file = output_dir / 'roi_gauge.html'
    fig_roi.write_html(str(roi_file))
    print(f"    Saved to: {roi_file}")

    # 7. Payback Timeline
    print("  > Creating payback timeline...")
    fig_payback = EconomicPlots.create_payback_timeline(
        payback_period=financial_summary['payback_period'],
        lifetime_years=BATTERY_LIFETIME_YEARS,
    )
    payback_file = output_dir / 'payback_timeline.html'
    fig_payback.write_html(str(payback_file))
    print(f"    Saved to: {payback_file}")

    # 8. Financial Dashboard
    print("  > Creating financial dashboard...")
    fig_dashboard = EconomicPlots.create_financial_dashboard(
        metrics=financial_summary,
        baseline_cost=baseline_cost_annual,
    )
    dashboard_file = output_dir / 'financial_dashboard.html'
    fig_dashboard.write_html(str(dashboard_file))
    print(f"    Saved to: {dashboard_file}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nAll visualizations saved to: {output_dir}")
    print("\nGenerated files:")
    print("  1. power_dispatch.html       - Power flow over time")
    print("  2. soc_evolution.html        - Battery state of charge")
    print("  3. price_signals.html        - Market prices and operations")
    print("  4. cost_breakdown.html       - Cost distribution")
    print("  5. savings_comparison.html   - Baseline vs optimized")
    print("  6. roi_gauge.html            - Return on investment")
    print("  7. payback_timeline.html     - Break-even analysis")
    print("  8. financial_dashboard.html  - Complete financial overview")
    print("\nOpen any HTML file in a web browser to view interactive visualizations.")
    print("=" * 80)


if __name__ == "__main__":
    run_optimization_with_visualization()
