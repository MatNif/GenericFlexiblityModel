# Generic Flexibility Model

A Python framework for modeling and optimizing flexibility assets in energy systems from a utility company perspective.

## Overview

This framework provides a **clean, extensible architecture** for evaluating the technical and economic potential of flexibility resources such as battery storage, photovoltaic systems, domestic hot water heaters, and other distributed energy resources (DERs).

The framework is designed to:
- **Separate concerns**: Physical behavior, economic evaluation, and operational decisions are cleanly separated into distinct layers
- **Enable optimization**: Provide interfaces that optimization algorithms can use to evaluate and execute operations
- **Support multiple assets**: Generic base classes allow easy implementation of new flexibility assets
- **Facilitate learning**: Clear architecture and comprehensive documentation make it suitable for teaching energy systems modeling

### Key Use Cases

- Evaluating flexibility potential for utility companies
- Optimizing distributed energy resources (DERs) operation
- Assessing techno-economic feasibility of flexibility services
- Teaching energy systems modeling and optimization
- Research on flexibility aggregation and market participation

---

## Architecture

The framework uses a **three-layer architecture** that separates physical modeling from economic evaluation, joined by operational decision-making:

```
┌─────────────────────────────────────────────────────┐
│                   FlexAsset                         │
│            (Operational Composition)                │
│  • evaluate_operation() - Check feasibility & cost  │
│  • execute_operation() - Update state & track       │
└───────────────┬─────────────────┬───────────────────┘
                │                 │
       ┌────────▼────────┐   ┌────▼──────────┐
       │   FlexUnit      │   │  CostModel    │
       │   (Physics)     │   │  (Economics)  │
       │                 │   │               │
       │ • Power limits  │   │ • Investment  │
       │ • Efficiency    │   │ • Degradation │
       │ • State (SOC)   │   │ • Energy cost │
       │ • Constraints   │   │ • Revenues    │
       └─────────────────┘   └───────────────┘
```

### Layer 1: FlexUnit (Physical/Technical Model)

Models the **physical behavior** of a flexibility asset:
- Power limits and ramping constraints
- Energy capacity and state of charge (SOC)
- Efficiency characteristics (charging, discharging, conversion)
- Self-discharge, standby losses, degradation
- Availability and maintenance schedules

**Key concept**: Energy headroom representation
- `E_plus`: Energy available to inject into the grid [kWh]
- `E_minus`: Energy capacity available to draw from the grid [kWh]

### Layer 2: CostModel (Economic Model)

Evaluates the **economic implications** of operations:
- Investment costs (CAPEX)
- Fixed and variable O&M costs
- Degradation and cycling costs
- Energy purchase and sales prices (time-varying)
- Capacity reservation payments
- Emissions costs (CO₂)

### Layer 3: FlexAsset (Operational Interface)

Composes physics and economics to provide an **operational interface**:
- `evaluate_operation()`: Check if an operation is feasible and calculate its cost
- `execute_operation()`: Execute the operation and update system state
- Tracks operational metrics (throughput, costs, activations)

This interface allows optimization algorithms to explore the solution space without needing to understand the internal physics or economics.

---

## Current Implementation

### ✅ Battery Energy Storage System (BESS)

A complete implementation demonstrating all three layers:

- **BatteryUnit**: Physical model with efficiency, SOC limits, power constraints, self-discharge
- **BatteryCostModel**: Economic model with degradation costs and energy arbitrage
- **BatteryFlex**: Operational composition for feasibility checking and execution

**Example**:
```python
from flex_model.assets import BatteryUnit, BatteryCostModel, BatteryFlex

# 1. Create physical model
battery_unit = BatteryUnit(
    name="BESS_100kWh",
    capacity_kwh=100.0,      # 100 kWh storage
    power_kw=50.0,           # 50 kW charge/discharge
    efficiency=0.95,         # 95% one-way efficiency
    soc_min=0.1,             # Don't discharge below 10%
    soc_max=0.9,             # Don't charge above 90%
)

# 2. Create economic model
battery_cost = BatteryCostModel(
    name="battery_economics",
    c_inv=500.0,                          # 500 CHF/kWh investment
    n_lifetime=10.0,                      # 10 year lifetime
    p_int=0.05,                           # 0.05 CHF/kWh degradation
    p_E_buy={0: 0.20, 1: 0.25, ...},      # Time-varying buy prices
    p_E_sell={0: 0.18, 1: 0.23, ...},     # Time-varying sell prices
)

# 3. Compose operational interface
battery = BatteryFlex(unit=battery_unit, cost_model=battery_cost)

# 4. Initialize state (50% SOC)
battery.reset(E_plus_init=50.0, E_minus_init=50.0)

# 5. Evaluate operation (discharge 30 kW for 15 min)
result = battery.evaluate_operation(
    t=10,
    dt_hours=0.25,
    P_grid_import=0.0,
    P_grid_export=30.0
)

if result['feasible']:
    print(f"Operation cost: {result['cost']:.2f} CHF")
    print(f"SOC after: {result['soc']:.1%}")

    # 6. Execute if optimal
    battery.execute_operation(t=10, dt_hours=0.25, P_grid_import=0.0, P_grid_export=30.0)
```

### 🚧 Planned Implementations

- **Photovoltaic (PV)**: Solar generation with curtailment options
- **Domestic Hot Water (DHW)**: Thermal storage with temperature layers
- **Heat Pumps**: Heating/cooling with COP modeling
- **Electric Vehicles (EV)**: Mobile storage with availability patterns

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/GenericFlexiblityModel.git
cd GenericFlexiblityModel

# Install in development mode
pip install -e .

# Run tests
pytest tests/ -v
```

### Requirements

- Python 3.10+
- NumPy (for numerical operations)
- pytest (for testing)

---

## Project Structure

```
GenericFlexiblityModel/
├── flex_model/
│   ├── core/
│   │   ├── flex_unit.py      # Base class for physical models
│   │   ├── cost_model.py     # Base class for economic models
│   │   ├── flex_asset.py     # Base class for operational interface
│   │   └── access_state.py   # State management utilities
│   └── assets/
│       ├── battery.py        # Battery implementation (BatteryUnit, BatteryCostModel, BatteryFlex)
│       └── ...               # Future: PV, DHW, heat pumps, etc.
├── tests/
│   ├── test_battery.py       # Battery unit tests
│   └── ...
├── README.md
└── setup.py
```

---

## Optimization Integration

The framework is designed to integrate with optimization algorithms for finding optimal operation schedules. The `FlexAsset` interface provides:

- **evaluate_operation()**: For optimization algorithms to query feasibility and costs
- **execute_operation()**: To apply the optimal solution and update system state

Example optimization workflow:
1. Optimizer proposes an operation (e.g., "charge battery at 30 kW")
2. FlexAsset evaluates feasibility and calculates cost
3. Optimizer explores solution space (linear programming, MILP, heuristics, etc.)
4. Optimal operations are executed to update system state

The framework is **optimization-algorithm agnostic** - it can work with:
- Linear programming (LP)
- Mixed-integer linear programming (MILP)
- Dynamic programming
- Heuristic methods
- Reinforcement learning

Future work will include reference implementations of common optimization approaches.

---

## Design Principles

1. **Separation of Concerns**: Physics, economics, and operations are cleanly separated
2. **Composition over Inheritance**: FlexAsset composes FlexUnit + CostModel
3. **Interface Segregation**: Optimization algorithms only see evaluate/execute interface
4. **Time-Dependent Flexibility**: All parameters can vary with time
5. **Educational Value**: Clear documentation and examples for learning

---

## Contributing

This is a research/educational project. Contributions are welcome:
- New flexibility asset implementations
- Optimization algorithm examples
- Documentation improvements
- Bug fixes and test coverage

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this framework in your research, please cite:
```
[Add citation information when published]
```

---

## Contact

Mathias Niffeler
Urban Energy Systems Laboratory
Empa, Dübendorf
mathias.niffeler@empa.ch
