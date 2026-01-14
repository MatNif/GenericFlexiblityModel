# Battery Optimization Dashboard

Interactive web dashboard for visualizing battery optimization results.

## Quick Start

### 1. Install Dependencies

```bash
# From project root
pip install -e .[dashboard]

# Or manually
pip install streamlit plotly pandas
```

### 2. Generate Test Data (if needed)

```bash
cd examples/battery_vs_market/utils
python generate_dummy_profile.py
```

### 3. Launch Dashboard

```bash
cd examples/battery_vs_market
streamlit run dashboard.py
```

The dashboard will automatically open in your browser at **http://localhost:8501**

## Features

### Interactive Configuration

Use the **sidebar** to adjust parameters in real-time:

**Battery Parameters:**
- Capacity [kWh]
- Power [kW]
- Efficiency [%]

**Economic Parameters:**
- Investment cost [CHF/kWh]
- Lifetime [years]
- Degradation cost [CHF/kWh]

**Operational Parameters:**
- Initial state of charge

Click **"Run Optimization"** to re-run with new parameters (results are cached for speed).

### Dashboard Tabs

**1. Operational Analysis**
- Power dispatch profile (how battery/market balance imbalances)
- State of charge evolution (SOC over time with limits)
- Price signals and market operations (correlation analysis)

**2. Economic Analysis**
- Savings comparison (baseline vs optimized)
- Cost breakdown by asset
- ROI gauge with target benchmarks
- Payback period timeline

**3. Executive Summary**
- Comprehensive financial dashboard
- Investment recommendation (✅/⚠️/❌)
- All KPIs in one view

**4. Detailed Metrics**
- Battery configuration summary
- Economic assumptions
- Optimization results
- Utilization metrics (capacity factor, cycles, etc.)

## Key Performance Indicators (Header)

The dashboard displays 4 critical KPIs at the top:

1. **ROI** - Return on investment [%]
2. **Payback Period** - Years to break even
3. **NPV** - Net present value [CHF]
4. **Annual Savings** - Cost reduction vs baseline [CHF/year]

## Tips

- **Caching**: Results are cached based on parameters. If you modify data files, click "Run Optimization" to refresh.
- **Interactivity**: All Plotly charts support hover, zoom, pan, and export to PNG.
- **Parameter Exploration**: Adjust battery capacity to find optimal sizing.
- **Sensitivity Analysis**: Change investment costs to see break-even points.

## Example Use Cases

### 1. Optimal Battery Sizing
- Start with 100 kWh capacity
- Gradually increase/decrease using sidebar slider
- Watch ROI and payback period change
- Find sweet spot where ROI is maximized

### 2. Investment Sensitivity
- Fix battery size (e.g., 100 kWh @ 50 kW)
- Adjust investment cost from 300-700 CHF/kWh
- Observe break-even investment cost
- Use for vendor negotiations

### 3. Operational Analysis
- Run optimization with default settings
- Navigate to "Operational Analysis" tab
- Check if battery charges during low prices
- Check if battery discharges during high prices
- Validate optimizer logic

## Troubleshooting

**Dashboard won't start:**
```bash
# Check streamlit installation
streamlit --version

# If missing:
pip install streamlit
```

**No data found:**
```bash
# Generate dummy data
cd utils
python generate_dummy_profile.py
```

**Optimization fails:**
- Check that imbalance profile has values
- Ensure price data matches number of timesteps
- Review console for error messages

## Architecture

The dashboard:
1. Loads imbalance prices and profile (cached)
2. Runs LP optimization with configured parameters (cached)
3. Wraps results in `OptimizationResult`
4. Calculates metrics with `EconomicMetrics`
5. Generates visualizations with `OperationalPlots` and `EconomicPlots`
6. Displays everything in Streamlit UI

All optimization calls are **cached** - changing a parameter invalidates cache and re-runs optimization automatically.

## Performance

- **First run**: ~1-5 seconds (depends on problem size)
- **Cached runs**: Instant (parameter changes without re-optimization)
- **Cache clear**: Click "Run Optimization" button

For large problems (1000+ timesteps), consider:
- Reducing time horizon for dashboard
- Pre-computing results offline
- Using result files instead of live optimization
