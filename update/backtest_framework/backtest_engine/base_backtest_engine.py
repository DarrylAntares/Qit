"""
回测引擎基类
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime

try:
    from ..strategy.base_strategy import OrderBook
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from strategy.base_strategy import OrderBook

class BacktestResult:
    """回测结果类"""
    
    def __init__(self):
        self.portfolio_values = []  # 组合净值序列
        self.dates = []  # 日期序列
        self.positions = {}  # 持仓记录
        self.trades = []  # 交易记录
        self.cash_flows = []  # 现金流记录
        self.metrics = {}  # 绩效指标
        
    def add_daily_record(self, date: datetime, portfolio_value: float, 
                        cash: float, positions: Dict, stock_prices: Dict = None):
        """添加每日记录"""
        self.dates.append(date)
        self.portfolio_values.append(portfolio_value)
        self.cash_flows.append(cash)
        
        # 深拷贝持仓记录，并更新当日价格
        daily_positions = {}
        for stock_code, pos_info in positions.items():
            daily_pos = pos_info.copy()
            
            # 更新当日价格和市值
            if stock_prices and stock_code in stock_prices:
                current_price = stock_prices[stock_code]
                daily_pos['current_price'] = current_price
                daily_pos['market_value'] = daily_pos['quantity'] * current_price
            
            daily_positions[stock_code] = daily_pos
        
        self.positions[date] = daily_positions
        
    def add_trade_record(self, trade_record: Dict):
        """添加交易记录"""
        self.trades.append(trade_record)
        
    def add_cash_flow_record(self, cash_flow_record: Dict):
        """添加现金流记录"""
        self.cash_flows.append(cash_flow_record)
        
    def get_portfolio_series(self) -> pd.Series:
        """获取组合净值序列"""
        return pd.Series(self.portfolio_values, index=self.dates)
        
    def get_trades_dataframe(self) -> pd.DataFrame:
        """获取交易记录DataFrame"""
        return pd.DataFrame(self.trades)
        
    def get_positions_dataframe(self) -> pd.DataFrame:
        """获取持仓记录DataFrame"""
        positions_list = []
        for date, positions in self.positions.items():
            for stock_code, position_info in positions.items():
                record = {
                    'date': date,
                    'stock_code': stock_code,
                    **position_info
                }
                positions_list.append(record)
        return pd.DataFrame(positions_list)

class BaseBacktestEngine(ABC):
    """回测引擎基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = None
        self.initial_cash = config.get('INITIAL_CASH', 1000000)
        self.commission = config.get('COMMISSION', 0.001)
        
    def set_logger(self, logger):
        """设置日志记录器"""
        self.logger = logger
        
    @abstractmethod
    def run_backtest(self, order_book: OrderBook, stock_data: Dict[str, pd.DataFrame],
                    trading_calendar: List[datetime]) -> BacktestResult:
        """
        运行回测
        
        参数:
            order_book: 订单簿
            stock_data: 股票数据
            trading_calendar: 交易日历
            
        返回:
            BacktestResult: 回测结果
        """
        pass
