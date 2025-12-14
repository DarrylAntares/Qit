"""
绩效分析基类
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd

try:
    from ..backtest_engine.base_backtest_engine import BacktestResult
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backtest_engine.base_backtest_engine import BacktestResult

class BasePerformanceAnalyzer(ABC):
    """绩效分析基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = None
        
    def set_logger(self, logger):
        """设置日志记录器"""
        self.logger = logger
        
    @abstractmethod
    def analyze(self, backtest_result: BacktestResult) -> Dict[str, Any]:
        """
        分析回测结果
        
        参数:
            backtest_result: 回测结果
            
        返回:
            Dict[str, Any]: 分析结果
        """
        pass
        
    @abstractmethod
    def generate_report(self, analysis_result: Dict[str, Any], 
                       backtest_result: BacktestResult) -> str:
        """
        生成分析报告
        
        参数:
            analysis_result: 分析结果
            backtest_result: 回测结果
            
        返回:
            str: 报告内容
        """
        pass
        
    @abstractmethod
    def save_results(self, analysis_result: Dict[str, Any], 
                    backtest_result: BacktestResult, output_dir: str):
        """
        保存分析结果
        
        参数:
            analysis_result: 分析结果
            backtest_result: 回测结果
            output_dir: 输出目录
        """
        pass
