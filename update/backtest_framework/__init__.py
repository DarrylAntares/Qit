"""
五层架构回测框架
"""

from .backtest_framework import BacktestFramework
from .config.config import Config

__version__ = "1.0.0"
__author__ = "Backtest Framework Team"

__all__ = [
    'BacktestFramework',
    'Config'
]
