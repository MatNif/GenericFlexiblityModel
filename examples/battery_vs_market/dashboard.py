"""
Interactive Dashboard for Battery Optimization Results.

Displays optimization results in a local web browser using Streamlit.

Run with:
    streamlit run dashboard.py

Then open your browser to http://localhost:8501
"""

from pathlib import Path
import sys

from flex_model.settings import DT_HOURS

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    print("ERROR: Streamlit not installed")
    print("Install with: pip install streamlit")
    STREAMLIT_AVAILABLE = False
    sys.exit(1)

from flex_model.assets import BatteryUnit, BatteryCostModel, BatteryFlex
from flex_model.assets import BalancingMarketCost, BalancingMarketFlex
from flex_model.optimization import LPOptimizer
from flex_model.visualization import LPOptimizationResult, EconomicMetrics
from flex_model.visualization.plots import OperationalPlots, EconomicPlots
from examples.battery_vs_market.utils.data.data_loader import load_imbalance_prices, load_imbalance_profile, get_data_path, ensure_converted
from utils.baseline_calculator import calculate_baseline_cost


# Page configuration
st.set_page_config(
    page_title="Battery Optimization Dashboard",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)


def get_plotly_template():
    """
    Detect browser theme and return appropriate Plotly template.

    Returns:
        str: 'plotly_dark' for dark mode, 'plotly_white' for light mode
    """
    # Try to detect Streamlit theme
    try:
        theme_base = st.get_option("theme.base")
        if theme_base == "dark":
            return "plotly_dark"
        elif theme_base == "light":
            return "plotly_white"
    except:
        pass

    # Fallback: Use JavaScript to detect browser preference
    try:
        # Inject JavaScript to detect color scheme
        dark_mode_js = """
        <script>
        const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        window.parent.postMessage({type: 'streamlit:setComponentValue', value: isDark}, '*');
        </script>
        """
        # For now, default to plotly (auto-adapting) as fallback
        return "plotly"
    except:
        return "plotly"  # Auto-adapting template


@st.cache_data
def load_data():
    """Load imbalance prices and profile (cached), with template fallback."""
    # Auto-convert raw data if outputs are missing or stale
    ensure_converted()

    data_path = get_data_path()

    def pick_file(primary: str, template: str) -> Path:
        primary_path = data_path / primary
        template_path = data_path / template

        if primary_path.exists():
            return primary_path

        if template_path.exists():
            st.warning(
                f"Using template file {template_path.name} because {primary_path.name} is missing."
            )
            return template_path

        # Neither exists -> hard stop with helpful message
        st.error(f"Data not found: {primary_path} (and no template {template_path})")
        st.stop()

    prices_file = pick_file("imbalance_prices.csv", "imbalance_prices_template.csv")
    p_buy, p_sell, start_date = load_imbalance_prices(str(prices_file))

    profile_file = pick_file("imbalance_profile.csv", "imbalance_profile_template.csv")
    imbalance = load_imbalance_profile(str(profile_file))

    return p_buy, p_sell, imbalance, start_date


def run_optimization(
    capacity_kwh,
    power_kw,
    efficiency,
    inv_cost_per_kwh,
    lifetime_years,
    degradation_per_kwh,
    initial_soc,
    p_buy,
    p_sell,
    imbalance
):
    """Run battery optimization (no caching due to lambda functions in FlexAssets)."""

    # Create battery
    battery_unit = BatteryUnit(
        name="BESS",
        capacity_kwh=capacity_kwh,
        power_kw=power_kw,
        efficiency=efficiency,
        soc_min=0.1,
        soc_max=0.9,
    )

    battery_cost = BatteryCostModel(
        name="battery_economics",
        c_inv=inv_cost_per_kwh,
        n_lifetime=lifetime_years,
        p_int=degradation_per_kwh,
    )

    battery = BatteryFlex(unit=battery_unit, cost_model=battery_cost)

    # Create market
    market_cost = BalancingMarketCost(
        name="market",
        p_E_buy=p_buy,
        p_E_sell=p_sell,
    )

    market = BalancingMarketFlex(cost_model=market_cost)

    # Run LP optimization
    n_timesteps = len(imbalance)

    battery_lm = battery.get_linear_model(
        n_timesteps=n_timesteps,
        initial_soc=initial_soc
    )
    market_lm = market.get_linear_model(
        n_timesteps=n_timesteps
    )

    optimizer = LPOptimizer(n_timesteps=n_timesteps)
    optimizer.add_asset(battery_lm)
    optimizer.add_asset(market_lm)
    optimizer.set_imbalance(imbalance)

    lp_result = optimizer.solve()

    if not lp_result['success']:
        st.error(f"Optimization failed: {lp_result['message']}")
        st.stop()

    # Calculate baseline (pure market)
    baseline_cost, baseline_cost_annual = calculate_baseline_cost(
        imbalance=imbalance,
        p_buy=p_buy,
        p_sell=p_sell
    )

    # Wrap result
    result = LPOptimizationResult(
        lp_result=lp_result,
        assets={'BESS': battery, 'market': market},
        imbalance=imbalance
    )

    # Calculate metrics
    metrics = EconomicMetrics.compute_financial_summary(
        result=result,
        baseline_cost=baseline_cost_annual,
        lifetime_years=lifetime_years,
        discount_rate=0.05,
    )

    return result, metrics, baseline_cost_annual


def main():
    """Main dashboard application."""

    # Get theme template for all plots
    plot_template = get_plotly_template()

    # Title
    st.title("🔋 Battery Optimization Dashboard")
    st.markdown("---")

    # Sidebar - Configuration
    st.sidebar.header("⚙️ Configuration")

    st.sidebar.subheader("Battery Parameters")
    capacity_kwh = st.sidebar.number_input(
        "Capacity [kWh]",
        min_value=10.0,
        max_value=100000.0,
        value=20000.0,
        step=10.0
    )

    power_kw = st.sidebar.number_input(
        "Power [kW]",
        min_value=5.0,
        max_value=5000.0,
        value=1000.0,
        step=5.0
    )

    efficiency = st.sidebar.slider(
        "Efficiency",
        min_value=0.80,
        max_value=1.00,
        value=0.95,
        step=0.01
    )

    st.sidebar.subheader("Economic Parameters")
    inv_cost_per_kwh = st.sidebar.number_input(
        "Investment Cost [CHF/kWh]",
        min_value=100.0,
        max_value=2000.0,
        value=500.0,
        step=50.0
    )

    lifetime_years = st.sidebar.number_input(
        "Lifetime [years]",
        min_value=5.0,
        max_value=25.0,
        value=10.0,
        step=1.0
    )

    degradation_per_kwh = st.sidebar.number_input(
        "Degradation Cost [CHF/kWh]",
        min_value=0.00,
        max_value=0.20,
        value=0.05,
        step=0.01
    )

    st.sidebar.subheader("Operational Parameters")
    initial_soc = st.sidebar.slider(
        "Initial SOC",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.1
    )

    # Load data
    with st.spinner("Loading data..."):
        p_buy, p_sell, imbalance, start_date_str = load_data()

    st.sidebar.success(f"✓ Loaded {len(imbalance)} timesteps")

    # Run optimization with session state caching
    # Only re-run when optimization parameters change (not when time-range slider changes)
    opt_params = (capacity_kwh, power_kw, efficiency, inv_cost_per_kwh,
                  lifetime_years, degradation_per_kwh, initial_soc)

    if ('opt_params' not in st.session_state or
        st.session_state.opt_params != opt_params):
        with st.spinner("Running optimization..."):
            result, metrics, baseline_cost_annual = run_optimization(
                capacity_kwh=capacity_kwh,
                power_kw=power_kw,
                efficiency=efficiency,
                inv_cost_per_kwh=inv_cost_per_kwh,
                lifetime_years=lifetime_years,
                degradation_per_kwh=degradation_per_kwh,
                initial_soc=initial_soc,
                p_buy=p_buy,
                p_sell=p_sell,
                imbalance=imbalance
            )
        st.session_state.opt_params = opt_params
        st.session_state.result = result
        st.session_state.metrics = metrics
        st.session_state.baseline_cost_annual = baseline_cost_annual
    else:
        result = st.session_state.result
        metrics = st.session_state.metrics
        baseline_cost_annual = st.session_state.baseline_cost_annual

    # Key Metrics Header
    st.header("📊 Key Performance Indicators")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        roi = metrics['roi']
        roi_color = "green" if roi > 15 else "orange" if roi > 5 else "red"
        st.metric(
            label="ROI",
            value=f"{roi:.1f}%",
            delta="Good" if roi > 15 else "Marginal" if roi > 5 else "Poor"
        )

    with col2:
        payback = metrics['payback_period']
        st.metric(
            label="Payback Period",
            value=f"{payback:.1f} years",
            delta="Within lifetime" if payback <= lifetime_years else "Exceeds lifetime",
            delta_color="normal" if payback <= lifetime_years else "inverse"
        )

    with col3:
        npv = metrics['npv']
        st.metric(
            label="NPV",
            value=f"{npv:,.0f} CHF",
            delta="Profitable" if npv > 0 else "Loss",
            delta_color="normal" if npv > 0 else "inverse"
        )

    with col4:
        savings = metrics['savings']['absolute_savings']
        st.metric(
            label="Annual Savings",
            value=f"{savings:,.0f} CHF/year",
            delta=f"{metrics['savings']['relative_savings']:.1f}%"
        )

    st.markdown("---")

    # Main visualizations
    operational_tab, economic_tab, metrics_tab = st.tabs([
        "📈 Operational Analysis",
        "💰 Economic Analysis",
        "📋 Detailed Metrics"
    ])

    with operational_tab:
        st.header("Operational Analysis")

        # Use actual start date from data file
        import pandas as pd
        start = pd.to_datetime(start_date_str, format='%d.%m.%Y %H:%M:%S')
        end = start + pd.Timedelta(hours=result.n_timesteps * DT_HOURS)

        # Format for plot functions (they expect "YYYY-MM-DD HH:MM" format)
        start_date_for_plots = start.strftime('%Y-%m-%d %H:%M')

        # Add date range controls
        st.subheader("Time Range Selection")

        # Single range slider with two handles (using timestep indices)
        time_range_idx = st.slider(
            "Select Time Range",
            min_value=0,
            max_value=result.n_timesteps,
            value=(0, result.n_timesteps),  # Tuple creates two handles
            step=1,
        )

        range_start_idx, range_end_idx = time_range_idx

        # Calculate actual datetime range from timestep indices
        display_start = start + pd.Timedelta(hours=range_start_idx * DT_HOURS)
        display_end = start + pd.Timedelta(hours=range_end_idx * DT_HOURS)

        # Display selected range prominently
        st.markdown(f"""
        <div style='background-color: #d4edda; border: 2px solid #28a745; border-radius: 5px; padding: 15px; margin: 10px 0;'>
            <h4 style='color: #155724; margin: 0; text-align: center;'>
                📅 Displaying: <strong>{display_start.strftime('%a %d/%m/%Y %H:%M')}</strong> to <strong>{display_end.strftime('%a %d/%m/%Y %H:%M')}</strong>
            </h4>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Create time range tuple for filtering
        time_range_filter = (range_start_idx, range_end_idx)

        # Power dispatch
        st.subheader("Power Dispatch Profile")
        fig_dispatch = OperationalPlots.create_dispatch_profile(
            result,
            view_mode='system',
            start_date=start_date_for_plots,
            template=plot_template,
            time_range_idx=time_range_filter
        )
        st.plotly_chart(fig_dispatch, use_container_width=True)

        # SOC evolution
        st.subheader("State of Charge Evolution")
        fig_soc = OperationalPlots.create_soc_evolution(
            result,
            battery_name='BESS',
            start_date=start_date_for_plots,
            template=plot_template,
            time_range_idx=time_range_filter
        )
        st.plotly_chart(fig_soc, use_container_width=True)

        # Price overlay
        st.subheader("Price Signals and Market Operations")
        fig_prices = OperationalPlots.create_price_overlay(
            result,
            market_name='market',
            start_date=start_date_for_plots,
            template=plot_template,
            time_range_idx=time_range_filter
        )
        st.plotly_chart(fig_prices, use_container_width=True)

    with economic_tab:
        st.header("Economic Analysis")

        # =====================================================
        # SECTION 1: INVESTMENT DECISION (UPFRONT)
        # =====================================================
        st.subheader("1️⃣ Investment Decision")

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            # Investment summary card
            fig_investment = EconomicPlots.create_investment_summary(
                investment_cost=metrics['savings']['investment_required'],
                capacity=capacity_kwh,
                unit_cost=inv_cost_per_kwh,
                lifetime_years=lifetime_years,
                template=plot_template,
            )
            st.plotly_chart(fig_investment, use_container_width=True)

        with col2:
            # ROI gauge
            fig_roi = EconomicPlots.create_roi_gauge(
                roi=metrics['roi'],
                target_roi=15.0,
                template=plot_template,
            )
            st.plotly_chart(fig_roi, use_container_width=True)

        with col3:
            # IRR gauge
            fig_irr = EconomicPlots.create_irr_gauge(
                irr=metrics['irr'],
                target_irr=12.0,
                template=plot_template,
            )
            st.plotly_chart(fig_irr, use_container_width=True)

        # Educational explanation
        with st.expander("💡 What's the difference between ROI and IRR?"):
            st.markdown("""
            **ROI (Return on Investment)** - Total return over lifetime
            - Simple calculation: (Savings - Investment) / Investment × 100%
            - Example: 23.5% ROI means you get 23.5% more than you invested
            - Best for: Quick yes/no profitability check

            **IRR (Internal Rate of Return)** - Annual return rate
            - Like an interest rate on your investment
            - Example: 18.2% IRR means 18.2% annual return
            - Best for: Comparing to other investments (stocks return ~7-10%, bonds ~3-5%)

            **Why is IRR lower than ROI here?**
            Because IRR accounts for *when* you receive savings. Money later is worth
            less than money today (time value of money).

            **Which should I use?**
            - Compare this investment to alternatives? → Use **IRR**
            - Quick profitability check? → Use **ROI**
            """)

        # Payback timeline
        st.subheader("Payback Analysis")
        fig_payback = EconomicPlots.create_payback_timeline(
            payback_period=metrics['payback_period'],
            lifetime_years=lifetime_years,
            template=plot_template,
        )
        st.plotly_chart(fig_payback, use_container_width=True)

        st.markdown("---")

        # =====================================================
        # SECTION 2: OPERATIONAL ECONOMICS (ONGOING)
        # =====================================================
        st.subheader("2️⃣ Operational Economics")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Annual Cost Comparison")
            fig_savings = EconomicPlots.create_savings_comparison(
                baseline_cost=baseline_cost_annual,
                optimized_cost=metrics['savings']['optimized_cost'],
                investment_cost=0,  # Don't show investment here (shown in section 1)
                template=plot_template,
            )
            st.plotly_chart(fig_savings, use_container_width=True)

        with col2:
            st.subheader("Cost & Revenue Breakdown")
            breakdown = EconomicMetrics.compute_cost_revenue_breakdown(
                result=result,
                baseline_cost=baseline_cost_annual,
                lifetime_years=lifetime_years,
            )
            fig_breakdown = EconomicPlots.create_cost_revenue_waterfall(breakdown, template=plot_template)
            st.plotly_chart(fig_breakdown, use_container_width=True)

        # Daily operations
        st.subheader("Daily Operations")
        daily_profile = EconomicMetrics.compute_daily_cost_profile(result, 'market')

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Daily Cost Timeline")
            fig_daily = EconomicPlots.create_daily_cost_timeseries(daily_profile, template=plot_template)
            st.plotly_chart(fig_daily, use_container_width=True)

        with col2:
            st.subheader("Cost Variability")
            fig_variability = EconomicPlots.create_cost_variability_analysis(daily_profile, template=plot_template)
            st.plotly_chart(fig_variability, use_container_width=True)

        st.markdown("---")

        # =====================================================
        # SECTION 3: RISK & SENSITIVITY
        # =====================================================
        st.subheader("3️⃣ Risk & Sensitivity Analysis")

        # Investment cost sensitivity (AVAILABLE)
        st.subheader("Investment Cost Sensitivity")
        st.markdown("*How do returns change if the battery costs more/less than expected?*")

        sensitivity = EconomicMetrics.compute_investment_sensitivity(
            result=result,
            baseline_cost=baseline_cost_annual,
            lifetime_years=lifetime_years,
        )
        fig_sensitivity = EconomicPlots.create_investment_sensitivity_chart(
            sensitivity=sensitivity,
            current_multiplier=1.0,
            template=plot_template,
        )
        st.plotly_chart(fig_sensitivity, use_container_width=True)

        breakeven_cost = sensitivity['breakeven_multiplier'] * inv_cost_per_kwh
        if breakeven_cost != float('inf') and breakeven_cost > 0:
            st.info(f"💡 **Break-even**: Investment would still be profitable up to {breakeven_cost:.0f} CHF/kWh")
        else:
            st.info("💡 **Break-even**: Investment is profitable at all tested cost levels")

    with metrics_tab:
        st.header("Detailed Metrics")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Battery Configuration")
            st.write({
                "Capacity": f"{capacity_kwh} kWh",
                "Power": f"{power_kw} kW",
                "Efficiency": f"{efficiency:.1%}",
                "SOC Range": "10% - 90%",
                "Initial SOC": f"{initial_soc:.0%}",
            })

            st.subheader("Economic Assumptions")
            st.write({
                "Investment Cost": f"{inv_cost_per_kwh} CHF/kWh",
                "Total Investment": f"{inv_cost_per_kwh * capacity_kwh:,.0f} CHF",
                "Lifetime": f"{lifetime_years} years",
                "Degradation Cost": f"{degradation_per_kwh} CHF/kWh",
                "Discount Rate": "5%",
            })

        with col2:
            st.subheader("Optimization Results")
            st.write({
                "Optimization Status": "✓ Success" if result.is_successful() else "✗ Failed",
                "Total Cost": f"{result.get_total_cost():.2f} CHF",
                "Timesteps": result.n_timesteps,
                "Time Horizon": f"{result.n_timesteps * DT_HOURS:.1f} hours",
            })

            st.subheader("Utilization Metrics")
            util = result.get_utilization_metrics('BESS')
            st.write({
                "Capacity Factor": f"{util['capacity_factor']:.1%}",
                "Utilization Hours": f"{util['utilization_hours']:.1f} h",
                "Number of Cycles": f"{util['num_cycles']:.2f}",
                "Avg Cycle Depth": f"{util['avg_cycle_depth']:.1f}%",
                "Max Power Used": f"{util['max_power_used']:.1f} kW",
            })

    # Footer
    st.markdown("---")
    st.caption("🔋 Battery Optimization Dashboard | Generic Flexibility Model Framework")


if __name__ == "__main__":
    if not STREAMLIT_AVAILABLE:
        print("Please install streamlit: pip install streamlit")
    else:
        main()
