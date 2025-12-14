"""
数据源基类
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime

class BaseDataSource(ABC):
    """数据源基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = None
    
    def set_logger(self, logger):
        """设置日志记录器"""
        self.logger = logger
    
    @abstractmethod
    def load_stock_data(self, start_date: datetime, end_date: datetime, 
                       stock_codes: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        加载股票数据
        
        参数:
            start_date: 开始日期
            end_date: 结束日期
            stock_codes: 股票代码列表，如果为None则加载所有股票
            
        返回:
            Dict[str, pd.DataFrame]: 股票代码到数据DataFrame的映射
        """
        pass
    
    @abstractmethod
    def load_index_constituents(self, index_code: str, start_date: datetime, 
                               end_date: datetime) -> Dict[str, List[str]]:
        """
        加载指数成分股数据
        
        参数:
            index_code: 指数代码
            start_date: 开始日期
            end_date: 结束日期
            
        返回:
            Dict[str, List[str]]: 日期到成分股列表的映射
        """
        pass
    
    @abstractmethod
    def load_trading_calendar(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        """
        加载交易日历
        
        参数:
            start_date: 开始日期
            end_date: 结束日期
            
        返回:
            List[datetime]: 交易日列表
        """
        pass
