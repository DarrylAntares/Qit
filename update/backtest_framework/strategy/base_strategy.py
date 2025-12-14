"""
策略基类
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime

class OrderBook:
    """订单簿类"""
    
    def __init__(self):
        self.orders = []
    
    def add_order(self, stock_code: str, signal_date: datetime, trade_date: datetime, 
                  action: str, weight: float = None, quantity: int = None, price: float = None):
        """
        添加订单
        
        参数:
            stock_code: 股票代码
            signal_date: 信号生成日期
            trade_date: 交易日期
            action: 交易动作 ('buy', 'sell', 'hold')
            weight: 买卖比例 (0-1)
            quantity: 买卖数量
            price: 交易价格
        """
        order = {
            'stock_code': stock_code,
            'signal_date': signal_date,
            'trade_date': trade_date,
            'action': action,
            'weight': weight,
            'quantity': quantity,
            'price': price
        }
        self.orders.append(order)
    
    def get_orders_by_date(self, trade_date: datetime) -> List[Dict]:
        """获取指定日期的订单"""
        return [order for order in self.orders if order['trade_date'] == trade_date]
    
    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame"""
        return pd.DataFrame(self.orders)
    
    def clear(self):
        """清空订单簿"""
        self.orders.clear()

class BaseStrategy(ABC):
    """策略基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = None
        self.order_book = OrderBook()
        self.current_positions = {}  # 当前持仓
        self.trading_calendar = []
        self.constituents = {}  # 成分股数据
    
    def set_logger(self, logger):
        """设置日志记录器"""
        self.logger = logger
    
    def set_trading_calendar(self, trading_calendar: List[datetime]):
        """设置交易日历"""
        self.trading_calendar = trading_calendar
    
    def set_constituents(self, constituents: Dict[str, List[str]]):
        """设置成分股数据"""
        self.constituents = constituents
    
    @abstractmethod
    def generate_signals(self, stock_data: Dict[str, pd.DataFrame], 
                        current_date: datetime) -> Dict[str, str]:
        """
        生成交易信号
        
        参数:
            stock_data: 股票数据
            current_date: 当前日期
            
        返回:
            Dict[str, str]: 股票代码到信号的映射 ('buy', 'sell', 'hold')
        """
        pass
    
    @abstractmethod
    def calculate_position_weights(self, signals: Dict[str, str], 
                                  stock_data: Dict[str, pd.DataFrame],
                                  current_date: datetime) -> Dict[str, float]:
        """
        计算持仓权重
        
        参数:
            signals: 交易信号
            stock_data: 股票数据
            current_date: 当前日期
            
        返回:
            Dict[str, float]: 股票代码到权重的映射
        """
        pass
    
    def need_rebalance(self, current_date: datetime) -> bool:
        """
        判断是否需要调仓
        
        参数:
            current_date: 当前日期
            
        返回:
            bool: 是否需要调仓
        """
        rebalance_freq = self.config.get('REBALANCE_FREQ', 'M')
        
        if not hasattr(self, 'last_rebalance_date') or self.last_rebalance_date is None:
            return True
        
        if rebalance_freq == 'D':  # 每日调仓
            return True
        elif rebalance_freq == 'W':  # 每周调仓
            return current_date.weekday() == 0  # 每周一调仓
        elif rebalance_freq == 'M':  # 每月调仓
            try:
                current_idx = self.trading_calendar.index(current_date)
                if current_idx > 0:
                    prev_trading_day = self.trading_calendar[current_idx - 1]
                    return current_date.month != prev_trading_day.month
                else:
                    return False
            except ValueError:
                return False
        else:
            return False
    
    def run_strategy(self, stock_data: Dict[str, pd.DataFrame]) -> OrderBook:
        """
        运行策略
        
        参数:
            stock_data: 股票数据
            
        返回:
            OrderBook: 订单簿
        """
        if self.logger:
            self.logger.info("开始运行策略")
        
        self.order_book.clear()
        self.last_rebalance_date = None
        
        for current_date in self.trading_calendar:
            # 检查是否需要调仓
            if self.need_rebalance(current_date):
                # 生成交易信号
                signals = self.generate_signals(stock_data, current_date)
                
                # 计算持仓权重
                weights = self.calculate_position_weights(signals, stock_data, current_date)
                
                # 生成订单
                self._generate_orders(signals, weights, current_date)
                
                # 更新最后调仓日期
                self.last_rebalance_date = current_date
        
        if self.logger:
            self.logger.info(f"策略运行完成，生成 {len(self.order_book.orders)} 个订单")
        
        return self.order_book
    
    def _generate_orders(self, signals: Dict[str, str], weights: Dict[str, float], 
                        current_date: datetime):
        """
        生成订单
        
        参数:
            signals: 交易信号
            weights: 持仓权重
            current_date: 当前日期
        """
        # 当日完成交易
        trade_date = current_date
        
        # 处理卖出信号（先卖后买）
        for stock_code, signal in signals.items():
            if signal == 'sell' or (signal == 'hold' and stock_code in self.current_positions):
                if stock_code in self.current_positions:
                    self.order_book.add_order(
                        stock_code=stock_code,
                        signal_date=current_date,
                        trade_date=trade_date,
                        action='sell',
                        weight=0.0
                    )
                    del self.current_positions[stock_code]
        
        # 处理买入信号
        for stock_code, signal in signals.items():
            if signal == 'buy' and stock_code in weights:
                weight = weights[stock_code]
                if weight > 0:
                    self.order_book.add_order(
                        stock_code=stock_code,
                        signal_date=current_date,
                        trade_date=trade_date,
                        action='buy',
                        weight=weight
                    )
                    self.current_positions[stock_code] = weight
