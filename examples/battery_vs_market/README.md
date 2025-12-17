# Battery vs. Imbalance Market: Investment Decision Example

This example demonstrates how to use the Generic Flexibility Model to answer a real-world investment question:

**Should we invest in a battery or simply accept imbalance settlement costs?**

## Scenario Description

A facility has energy imbalances that need to be settled with the TSO (Swissgrid). Two options:

1. **Option A: Battery Investment**
   - Install a 100 kWh / 50 kW battery energy storage system
   - Use battery to minimize imbalance settlement costs
   - Fall back to market when battery is depleted/full
   - Costs: Investment (CAPEX) + degradation + residual imbalance settlement

2. **Option B: Pure Market Settlement**
   - No investment
   - Settle all imbalances directly with TSO at real-time prices
   - Costs: Only imbalance settlement

## Data Requirements

### Imbalance Prices (`data/imbalance_prices.csv`)
Real-time imbalance prices from Swissgrid (Preise für Ausgleichsenergie):
- `timestep`: Integer time index (0, 1, 2, ...)
- `price_positive`: Price when you consumed MORE than scheduled [CHF/kWh]
- `price_negative`: Price when you consumed LESS than scheduled [CHF/kWh]

**Source:** [Swissgrid Transparency Platform](https://www.swissgrid.ch/en/home/operation/regulation/ancillary-services.html)

### Imbalance Profile (`data/imbalance_profile.csv`)
Time series of actual energy imbalances:
- `timestep`: Integer time index
- `imbalance_kw`: Power imbalance at this timestep [kW]
  - Positive = Need to buy energy (consumed more than scheduled)
  - Negative = Can sell energy (consumed less than scheduled)

**Source:** Provided by utility company or generated as dummy data for testing

## File Structure

```
battery_vs_market/
├── README.md                    # This file
├── __init__.py                  # Package description
├── data_loader.py               # Utilities for loading/generating data
├── scenario.py                  # Main optimization script
├── data/
│   ├── imbalance_prices.csv    # Real prices from Swissgrid (you provide)
│   └── imbalance_profile.csv   # Your imbalance data (you provide)
└── results/                     # Generated results (created by scenario.py)
    ├── battery_schedule.csv
    ├── cost_comparison.csv
    └── visualization.png
```

## Usage

### 1. Add Your Data

Place your data files in the `data/` directory:
- `imbalance_prices.csv`: Download from Swissgrid
- `imbalance_profile.csv`: Obtain from your utility company

If you don't have real data yet, the script will generate dummy data automatically.

### 2. Run the Scenario

```bash
cd examples/battery_vs_market
python scenario.py
```

### 3. Review Results

The script will output:
- Total cost for each scenario
- Break-even battery investment cost
- Optimal battery operation schedule
- Visualization of results

## Expected Output

```
=== Battery vs. Market Comparison ===

Scenario A (Battery + Market):
  - Battery investment (annualized): 5,000 CHF/year
  - Battery degradation: 450 CHF
  - Residual imbalance costs: 2,300 CHF
  - TOTAL: 7,750 CHF/year

Scenario B (Pure Market):
  - Imbalance costs: 12,500 CHF/year
  - TOTAL: 12,500 CHF/year

Savings with battery: 4,750 CHF/year (38%)
Break-even investment: 47,500 CHF (for 10-year lifetime)

Recommendation: Battery investment is economical
```

## Customization

Edit `scenario.py` to adjust:
- Battery size (capacity, power)
- Investment cost assumptions
- Optimization horizon
- SOC limits and efficiency
- Discount rate for annualization