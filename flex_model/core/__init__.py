"""
Core base classes for the flexibility modeling framework.

This module provides the abstract base classes that define the three-layer architecture:
    - FlexUnit: Physical/technical behavior
    - CostModel: Economic evaluation
    - FlexAsset: Operational composition
    - AccessState: Discrete operating states

Usage:
    from flex_model.core import FlexUnit, CostModel, FlexAsset, AccessState
"""

from flex_model.core.flex_unit import FlexUnit
from flex_model.core.cost_model import CostModel
from flex_model.core.flex_asset import FlexAsset
from flex_model.core.access_state import AccessState

__all__ = [
    'FlexUnit',
    'CostModel',
    'FlexAsset',
    'AccessState',
]