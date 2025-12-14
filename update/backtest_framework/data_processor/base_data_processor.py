"""
数据处理基类
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime

class BaseDataProcessor(ABC):
    """数据处理基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = None
    
    def set_logger(self, logger):
        """设置日志记录器"""
        self.logger = logger
    
    @abstractmethod
    def clean_data(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        数据清洗
        
        参数:
            stock_data: 原始股票数据
            
        返回:
            Dict[str, pd.DataFrame]: 清洗后的股票数据
        """
        pass
    
    @abstractmethod
    def calculate_factors(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        计算因子
        
        参数:
            stock_data: 股票数据
            
        返回:
            Dict[str, pd.DataFrame]: 包含因子的股票数据
        """
        pass
    
    @abstractmethod
    def align_data_with_calendar(self, stock_data: Dict[str, pd.DataFrame], 
                                trading_calendar: List[datetime]) -> Dict[str, pd.DataFrame]:
        """
        将数据与交易日历对齐
        
        参数:
            stock_data: 股票数据
            trading_calendar: 交易日历
            
        返回:
            Dict[str, pd.DataFrame]: 对齐后的股票数据
        """
        pass
