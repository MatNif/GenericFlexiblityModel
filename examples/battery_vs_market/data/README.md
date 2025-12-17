# Data Directory

Place your input data files here:

## Required Files

### `imbalance_prices.csv`
Real-time imbalance prices from Swissgrid.

**Format:**
```csv
timestep,price_positive,price_negative
0,0.245,0.152
1,0.268,0.165
...
```

**Columns:**
- `timestep`: Integer time index (0, 1, 2, ...)
- `price_positive`: Price for positive imbalance (consumed more) [CHF/kWh]
- `price_negative`: Price for negative imbalance (consumed less) [CHF/kWh]

**Source:** Download from [Swissgrid Transparency Platform](https://www.swissgrid.ch/en/home/operation/regulation/ancillary-services.html)

### `imbalance_profile.csv`
Your facility's energy imbalances.

**Format:**
```csv
timestep,imbalance_kw
0,15.5
1,-8.2
...
```

**Columns:**
- `timestep`: Integer time index (0, 1, 2, ...)
- `imbalance_kw`: Power imbalance [kW]
  - Positive = need to buy (consumed more than scheduled)
  - Negative = can sell (consumed less than scheduled)

**Source:** Obtain from your utility company or energy management system

## Templates

See `*_template.csv` files for format examples. The scenario script will generate dummy data if real data is not available.

## Notes

- Time resolution: Typically 15 minutes (96 timesteps per day)
- Horizon: Recommend at least 1 week (672 timesteps) for meaningful analysis
- The `timestep` column should be sequential integers starting from 0
