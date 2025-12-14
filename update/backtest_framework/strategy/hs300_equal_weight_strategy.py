"""
沪深300等权重策略实现
"""
import pandas as pd
from datetime import datetime
from typing import Dict, List

try:
    from .base_strategy import BaseStrategy
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from strategy.base_strategy import BaseStrategy

class HS300EqualWeightStrategy(BaseStrategy):
    """沪深300等权重策略"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.index_code = 'hs300'
    
    def generate_signals(self, stock_data: Dict[str, pd.DataFrame], 
                        current_date: datetime) -> Dict[str, str]:
        """
        生成交易信号
        
        对于等权重策略，信号生成逻辑：
        1. 获取当前日期的指数成分股
        2. 对成分股生成买入信号
        3. 对非成分股生成卖出信号
        
        参数:
            stock_data: 股票数据
            current_date: 当前日期
            
        返回:
            Dict[str, str]: 股票代码到信号的映射
        """
        signals = {}
        
        # 获取当前日期的成分股
        date_str = current_date.strftime('%Y-%m-%d')
        current_constituents = set(self.constituents.get(date_str, []))
        
        if not current_constituents:
            if self.logger:
                self.logger.warning(f"未找到 {date_str} 的成分股数据")
            return signals
        
        # 过滤出在股票数据中存在的成分股
        available_constituents = current_constituents.intersection(set(stock_data.keys()))
        
        # 进一步过滤：确保股票在当前日期有数据
        valid_constituents = []
        for stock_code in available_constituents:
            if stock_code in stock_data:
                stock_df = stock_data[stock_code]
                if current_date in stock_df.index and not pd.isna(stock_df.loc[current_date, 'close']):
                    valid_constituents.append(stock_code)
        
        if self.logger:
            self.logger.debug(f"{date_str}: 成分股总数={len(current_constituents)}, "
                           f"可用成分股={len(available_constituents)}, "
                           f"有效成分股={len(valid_constituents)}")
        
        # 为有效成分股生成买入信号
        for stock_code in valid_constituents:
            signals[stock_code] = 'buy'
        
        # 为当前持仓中不在成分股中的股票生成卖出信号
        for stock_code in self.current_positions:
            if stock_code not in valid_constituents:
                signals[stock_code] = 'sell'
        
        return signals
    
    def calculate_position_weights(self, signals: Dict[str, str], 
                                  stock_data: Dict[str, pd.DataFrame],
                                  current_date: datetime) -> Dict[str, float]:
        """
        计算持仓权重
        
        对于等权重策略，每只成分股的权重相等
        
        参数:
            signals: 交易信号
            stock_data: 股票数据
            current_date: 当前日期
            
        返回:
            Dict[str, float]: 股票代码到权重的映射
        """
        weights = {}
        
        # 统计买入信号的股票数量
        buy_stocks = [stock_code for stock_code, signal in signals.items() if signal == 'buy']
        
        if len(buy_stocks) == 0:
            return weights
        
        # 等权重分配
        equal_weight = 1.0 / len(buy_stocks)
        
        for stock_code in buy_stocks:
            weights[stock_code] = equal_weight
        
        if self.logger:
            self.logger.debug(f"{current_date.strftime('%Y-%m-%d')}: "
                           f"等权重分配给 {len(buy_stocks)} 只股票，每只权重 {equal_weight:.4f}")
        
        return weights
    
    def get_strategy_info(self) -> Dict:
        """获取策略信息"""
        return {
            'strategy_name': 'HS300等权重策略',
            'strategy_type': '指数增强',
            'rebalance_freq': self.config.get('REBALANCE_FREQ', 'M'),
            'index_code': self.index_code,
            'description': '基于沪深300指数成分股的等权重配置策略'
        }
