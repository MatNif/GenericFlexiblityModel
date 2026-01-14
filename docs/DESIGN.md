# Design Documentation

This document captures architectural decisions, design patterns, and implementation guidelines for the Generic Flexibility Model framework.

---

## Table of Contents

1. [Multiple Optimization Algorithm Support](#multiple-optimization-algorithm-support)
2. [Model Consistency Testing](#model-consistency-testing)

---

## Multiple Optimization Algorithm Support

### Problem Statement

As we add more optimization algorithms (GA, DP, MILP, RL), each `FlexAsset` needs multiple representations:

- **Operational interface** (`evaluate_operation`, `execute_operation`) - Sequential, timestep-by-timestep
- **Linear model** (`get_linear_model`) - Matrix form for LP solvers
- **GA encoding** (future: `get_ga_encoding`) - Chromosome representation
- **DP state space** (future: `get_dp_model`) - State transitions and value functions
- **Differentiable model** (future: `get_torch_model`) - For gradient-based methods

Without a clear pattern, `FlexAsset` classes will become cluttered with representation-specific methods, making the code harder to understand and maintain.

### Solution: Representation Builder Pattern

**Pattern:** Keep convenience methods in `FlexAsset` classes but delegate implementation to specialized builder classes.

**When to implement:** When we add the **second optimization algorithm** that requires a new representation (likely MILP, GA, or DP).

### Architecture

#### Current Structure (LP only)
```python
# flex_model/assets/battery.py
class BatteryFlex(FlexAsset):
    def evaluate_operation(self, ...): ...
    def execute_operation(self, ...): ...

    def get_linear_model(self, n_timesteps, dt_hours):
        """Get LP representation - all logic inline."""
        # 100+ lines of matrix construction...
        ...
```

#### Future Structure (Multiple representations)

**1. Create builders module:**
```
flex_model/representations/
├── __init__.py
├── linear_builder.py      # LinearModelBuilder
├── ga_builder.py          # GAModelBuilder
└── dp_builder.py          # DPModelBuilder
```

**2. Extract representation logic:**
```python
# flex_model/representations/linear_builder.py
class LinearModelBuilder:
    """Builds LinearModel representation from FlexAssets."""

    @staticmethod
    def build_battery(battery: BatteryFlex, n_timesteps: int, dt_hours: float) -> LinearModel:
        """
        Convert battery to LP matrix representation.

        Decision variables:
            - P_charge[t] for t in 0..n_timesteps
            - P_discharge[t] for t in 0..n_timesteps
            - SOC[t] for t in 0..n_timesteps+1

        Constraints:
            - Power limits: 0 <= P_charge[t] <= P_max
            - SOC limits: SOC_min <= SOC[t] <= SOC_max
            - Energy balance: SOC[t+1] = SOC[t] + (P_charge[t]*eta - P_discharge[t]/eta)*dt
            - Initial condition: SOC[0] = SOC_init

        Objective:
            - Minimize: sum_t [degradation_cost * throughput + energy_cost]
        """
        import numpy as np

        # All the detailed matrix construction logic here...
        # This keeps the FlexAsset class clean while preserving all implementation details

        n_vars = 3 * n_timesteps + 1  # P_charge, P_discharge, SOC

        # Variable bounds
        var_bounds = []
        for t in range(n_timesteps):
            var_bounds.append((0.0, battery.unit.power_kw))  # P_charge
        for t in range(n_timesteps):
            var_bounds.append((0.0, battery.unit.power_kw))  # P_discharge
        for t in range(n_timesteps + 1):
            var_bounds.append((battery.unit.soc_min * battery.unit.capacity_kwh,
                             battery.unit.soc_max * battery.unit.capacity_kwh))  # SOC

        # Cost coefficients
        cost_coefficients = np.zeros(n_vars)
        for t in range(n_timesteps):
            # Degradation cost on throughput
            cost_coefficients[t] = battery.cost_model.p_int * dt_hours  # P_charge
            cost_coefficients[n_timesteps + t] = battery.cost_model.p_int * dt_hours  # P_discharge

        # Constraints: Energy balance, initial SOC, etc.
        # ... (detailed matrix construction)

        return LinearModel(
            name=battery.name,
            n_timesteps=n_timesteps,
            n_vars=n_vars,
            var_names=var_names,
            var_bounds=var_bounds,
            cost_coefficients=cost_coefficients,
            A_eq=A_eq,
            b_eq=b_eq,
            A_ub=A_ub,
            b_ub=b_ub,
            power_indices=power_indices,
        )

    @staticmethod
    def build_market(market: BalancingMarketFlex, n_timesteps: int, dt_hours: float) -> LinearModel:
        """Convert market settlement to LP representation."""
        # Market-specific LP construction...
        ...
```

**3. Refactor FlexAsset to use builders:**
```python
# flex_model/assets/battery.py
class BatteryFlex(FlexAsset):
    # ============================================================================
    # OPERATIONAL INTERFACE (Core - always required)
    # ============================================================================

    def evaluate_operation(self, t, dt_hours, P_grid_import, P_grid_export):
        """Evaluate operation feasibility and cost without modifying state."""
        ...

    def execute_operation(self, t, dt_hours, P_grid_import, P_grid_export):
        """Execute operation and update internal state."""
        ...

    def power_limits(self, t):
        """Return instantaneous power limits at timestep t."""
        ...

    def reset(self, E_plus_init, E_minus_init):
        """Reset to initial state."""
        ...

    def get_metrics(self):
        """Return operational metrics."""
        ...

    # ============================================================================
    # OPTIMIZATION REPRESENTATIONS (For different solver types)
    # ============================================================================

    def get_linear_model(self, n_timesteps: int, dt_hours: float) -> LinearModel:
        """
        Convert to linear programming representation.

        Used by: LP optimizer, MILP optimizer

        Returns:
            LinearModel with decision variables, constraints, and costs in matrix form.
        """
        from flex_model.representations import LinearModelBuilder
        return LinearModelBuilder.build_battery(self, n_timesteps, dt_hours)

    def get_ga_encoding(self, n_timesteps: int, dt_hours: float):
        """
        Convert to genetic algorithm representation.

        Used by: GA optimizer

        Returns:
            GAModel with chromosome encoding, fitness function, and constraint checkers.
        """
        from flex_model.representations import GAModelBuilder
        return GAModelBuilder.build_battery(self, n_timesteps, dt_hours)

    def get_dp_model(self, n_timesteps: int, dt_hours: float):
        """
        Convert to dynamic programming representation.

        Used by: DP optimizer

        Returns:
            DPModel with state space, action space, transition function, and reward function.
        """
        from flex_model.representations import DPModelBuilder
        return DPModelBuilder.build_battery(self, n_timesteps, dt_hours)
```

### Benefits

1. **Clarity**: Clear visual separation between operational logic and optimizer representations
2. **Maintainability**: Representation logic isolated in dedicated modules
3. **Testability**: Can test builders independently from FlexAssets
4. **Discoverability**: `get_<algorithm>_model()` naming makes it obvious what's supported
5. **Backward compatibility**: Existing `get_linear_model()` API unchanged
6. **Documentation**: Each representation method documents which optimizers use it

### Implementation Checklist

When adding a new optimization algorithm representation:

- [ ] Create builder class in `flex_model/representations/`
- [ ] Implement `build_<asset>()` methods for each FlexAsset type
- [ ] Add convenience method `get_<algorithm>_model()` to each FlexAsset
- [ ] Add model consistency tests (see next section)
- [ ] Update README optimizer table
- [ ] Document builder in this DESIGN.md file

---

## Model Consistency Testing

### Problem Statement

Each `FlexAsset` can have multiple model representations (operational, LP, GA, DP, etc.). These representations must encode **identical underlying behavior**:

- Same costs for same operations
- Same feasibility constraints
- Same state transitions
- Same optimization outcomes (when constraints are satisfied)

Without systematic testing, representations can drift and produce inconsistent results.

### Solution: Model Consistency Test Framework

**Goal:** Verify that different representations of the same asset produce equivalent results.

**Location:** `tests/test_model_consistency.py`

**NOT testing:** Optimizer performance or solution quality (that's integration testing)

**Testing:** Model representation equivalence

### Test Strategy

#### 1. Operational vs. Linear Model Consistency

For assets that support both operational and LP representations:

```python
# tests/test_model_consistency.py
class TestBatteryModelConsistency:
    """Verify BatteryFlex representations are internally consistent."""

    def test_fixed_operation_sequence(self):
        """
        Test that a fixed sequence of operations produces identical costs
        through operational interface vs. LP representation.
        """
        # Setup
        battery = create_test_battery()
        n_timesteps = 10
        dt_hours = 0.25

        # Define fixed operation sequence
        operations = [
            (0.0, 30.0),  # t=0: discharge 30 kW
            (20.0, 0.0),  # t=1: charge 20 kW
            (0.0, 0.0),   # t=2: idle
            # ... etc
        ]

        # Method 1: Apply through operational interface
        battery.reset(E_plus_init=50.0, E_minus_init=50.0)
        total_cost_operational = 0.0
        for t, (P_import, P_export) in enumerate(operations):
            result = battery.evaluate_operation(t, dt_hours, P_import, P_export)
            assert result['feasible'], f"Operation at t={t} should be feasible"
            total_cost_operational += result['cost']
            battery.execute_operation(t, dt_hours, P_import, P_export)

        # Method 2: Apply through LP (with fixed decisions)
        linear_model = battery.get_linear_model(n_timesteps, dt_hours)

        # Construct LP problem with fixed decisions
        # (Set bounds to force the same operation sequence)
        total_cost_lp = compute_lp_cost_for_fixed_operations(
            linear_model, operations, dt_hours
        )

        # Assert equivalence
        assert total_cost_operational == pytest.approx(total_cost_lp, rel=1e-6), \
            "Operational and LP representations must produce identical costs"

    def test_soc_evolution_consistency(self):
        """Verify SOC state evolution is identical across representations."""
        # Test that SOC trajectory matches between operational execution
        # and LP solution for the same operation sequence
        ...

    def test_constraint_enforcement_consistency(self):
        """Verify power limits and SOC limits are enforced identically."""
        # Test boundary conditions: max charge, max discharge, SOC limits
        ...
```

#### 2. Test Organization

```
tests/
├── test_battery.py                    # Battery unit tests (existing)
├── test_balancing_market.py           # Market unit tests (existing)
└── test_model_consistency.py          # NEW: Cross-representation tests
    ├── TestBalancingMarketConsistency
    │   ├── test_single_timestep_consistency
    │   ├── test_multi_timestep_consistency
    │   └── test_price_variation_consistency
    └── TestBatteryConsistency
        ├── test_fixed_operation_sequence
        ├── test_soc_evolution_consistency
        ├── test_constraint_enforcement_consistency
        └── test_efficiency_losses_consistency
```

#### 3. Future Extensions

As new representations are added:

```python
class TestBatteryConsistency:
    def test_operational_vs_lp(self): ...
    def test_operational_vs_ga(self): ...      # When GA is added
    def test_operational_vs_dp(self): ...      # When DP is added
    def test_lp_vs_ga(self): ...               # Cross-check representations
```

### Test Principles

1. **Fixed operations:** Tests use predetermined operation sequences (not optimization)
2. **Deterministic:** No randomness in test scenarios
3. **Comprehensive:** Test normal operations, boundary conditions, and edge cases
4. **Isolated:** Each test is independent and can run in any order
5. **Fast:** Use small problem sizes (10-20 timesteps) for quick feedback

### Example: Simple Consistency Test

```python
def test_balancing_market_single_timestep():
    """Simplest possible test: one timestep, one operation."""
    # Setup
    market = BalancingMarketFlex(
        cost_model=BalancingMarketCost(
            name="test_market",
            p_E_buy=0.25,  # Buy at 25 ct/kWh
            p_E_sell=0.15, # Sell at 15 ct/kWh
        )
    )

    # Operation: Import 40 kW for 15 minutes
    P_import = 40.0
    P_export = 0.0
    dt_hours = 0.25

    # Method 1: Operational interface
    result = market.evaluate_operation(0, dt_hours, P_import, P_export)
    cost_operational = result['cost']

    # Method 2: Linear model
    linear_model = market.get_linear_model(n_timesteps=1, dt_hours=dt_hours)

    # LP cost = p_buy * P_import * dt
    cost_lp = linear_model.cost_coefficients[0] * P_import

    # Assert equivalence
    assert cost_operational == pytest.approx(cost_lp), \
        f"Operational cost {cost_operational} != LP cost {cost_lp}"
```

---

## Future Design Topics

As the framework evolves, this document will expand to cover:

- State management patterns for stateful assets
- Time-dependent parameter handling
- Multi-asset aggregation strategies
- Uncertainty modeling approaches
- Performance optimization techniques

---

**Document version:** 1.0
**Last updated:** 2026-01-05
**Maintainer:** Mathias Niffeler
