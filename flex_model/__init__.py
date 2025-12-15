"""
Generic Flexibility Model for Energy Systems.

This package provides a framework for modeling flexibility assets in energy systems
with a clean separation between physical behavior, economic evaluation, and operational
decision-making.

Architecture:
    - core.flex_unit: Physical/technical models (FlexUnit)
    - core.cost_model: Economic models (CostModel)
    - core.flex_asset: Operational composition (FlexAsset)
    - assets: Concrete implementations (Battery, PV, DHW, etc.)

Quick start:
    from flex_model.assets import BatteryUnit, BatteryCostModel, BatteryFlex

    # Create battery system
    battery_unit = BatteryUnit(name="BESS", capacity_kwh=100, power_kw=50)
    battery_cost = BatteryCostModel(name="cost", c_inv=500, n_lifetime=10, p_int=0.05)
    battery = BatteryFlex(unit=battery_unit, cost_model=battery_cost)

    # Initialize and operate
    battery.reset(E_plus_init=50.0, E_minus_init=50.0)
    result = battery.evaluate_operation(t=0, dt_hours=0.25, P_draw=30.0, P_inject=0.0)
"""

__version__ = "0.1.0"