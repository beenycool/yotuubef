"""
Optimization package for smart A/B testing and data-driven decisions
"""

from .smart_optimization_engine import (
    SmartOptimizationEngine,
    ABTest,
    TestVariant,
    TestResult,
    TestType,
    TestStatus
)

__all__ = [
    'SmartOptimizationEngine',
    'ABTest',
    'TestVariant', 
    'TestResult',
    'TestType',
    'TestStatus'
]