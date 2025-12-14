"""
简单回测引擎实现
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
from tqdm import tqdm

try:
    from .base_backtest_engine import BaseBacktestEngine, BacktestResult
    from ..strategy.base_strategy import OrderBook
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backtest_engine.base_backtest_engine import BaseBacktestEngine, BacktestResult
    from strategy.base_strategy import OrderBook

class SimpleBacktestEngine(BaseBacktestEngine):
    """简单回测引擎"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.min_trade_unit = 100  # 最小交易单位（手）
        
        # 价格配置
        self.trade_price_type = config.get('TRADE_PRICE_TYPE', 'close')  # 交易价格类型
        self.position_value_price_type = config.get('POSITION_VALUE_PRICE_TYPE', 'close')  # 持仓估值价格类型
        
        # 验证价格类型配置
        valid_price_types = ['open', 'close', 'high', 'low']
        if self.trade_price_type not in valid_price_types:
            if self.logger:
                self.logger.warning(f"无效的交易价格类型: {self.trade_price_type}, 使用默认值 'close'")
            self.trade_price_type = 'close'
        
        if self.position_value_price_type not in valid_price_types:
            if self.logger:
                self.logger.warning(f"无效的持仓估值价格类型: {self.position_value_price_type}, 使用默认值 'close'")
            self.position_value_price_type = 'close'
        
        if self.logger:
            self.logger.info(f"交易价格类型: {self.trade_price_type}")
            self.logger.info(f"持仓估值价格类型: {self.position_value_price_type}")
    
    def _get_price(self, stock_df: pd.DataFrame, date: datetime, price_type: str) -> float:
        """
        获取指定类型的价格
        
        参数:
            stock_df: 股票数据DataFrame
            date: 日期
            price_type: 价格类型 ('open', 'close', 'high', 'low')
            
        返回:
            float: 价格，如果获取失败返回None
        """
        try:
            if date not in stock_df.index:
                return None
            
            price = stock_df.loc[date, price_type]
            if pd.isna(price):
                return None
            
            return float(price)
        except (KeyError, IndexError, ValueError):
            return None
    
    def _get_trade_price(self, stock_df: pd.DataFrame, date: datetime) -> float:
        """获取交易价格"""
        return self._get_price(stock_df, date, self.trade_price_type)
    
    def _get_position_value_price(self, stock_df: pd.DataFrame, date: datetime) -> float:
        """获取持仓估值价格"""
        return self._get_price(stock_df, date, self.position_value_price_type)
    
    def _calculate_holding_days(self, stock_code: str, sell_date: datetime, 
                               trades_history: List[Dict], trading_calendar: List[datetime]) -> int:
        """计算持有天数
        
        参数:
            stock_code: 股票代码
            sell_date: 卖出日期
            trades_history: 历史交易记录列表
            trading_calendar: 交易日历
            
        返回:
            int: 持有交易日天数
        """
        # 找到最近一笔买入记录
        buy_date = None
        for trade in reversed(trades_history):
            if trade['stock_code'] == stock_code and trade['action'] == 'buy':
                buy_date = trade['date']
                break
        
        if buy_date is None:
            return 0
        
        # 计算交易日天数
        try:
            buy_idx = trading_calendar.index(buy_date)
            sell_idx = trading_calendar.index(sell_date)
            holding_days = sell_idx - buy_idx
            return max(0, holding_days)
        except (ValueError, IndexError):
            return 0
    
    def _calculate_return_rate(self, sell_price: float, avg_cost: float) -> float:
        """计算收益率
        
        参数:
            sell_price: 卖出价格
            avg_cost: 平均持仓成本
            
        返回:
            float: 收益率（百分比）
        """
        if avg_cost <= 0:
            return 0.0
        return ((sell_price - avg_cost) / avg_cost) * 100
        
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
        if self.logger:
            self.logger.info("开始运行回测")
            
        # 初始化回测状态
        result = BacktestResult()
        cash = self.initial_cash
        positions = {}  # {stock_code: {'quantity': int, 'avg_price': float, 'market_value': float}}
        self.trading_calendar = trading_calendar  # 保存交易日历用于计算持有天数
        
        # 将订单按日期分组
        orders_df = order_book.to_dataframe()
        if orders_df.empty:
            if self.logger:
                self.logger.warning("订单簿为空，无法进行回测")
            return result
            
        orders_by_date = orders_df.groupby('trade_date')
        
        # 逐日进行回测
        for current_date in tqdm(trading_calendar, desc="回测进度"):
            # 更新持仓市值
            self._update_positions_market_value(positions, stock_data, current_date)
            
            # 计算当前组合总市值（用于计算仓位占比）
            portfolio_value = cash + sum(pos['market_value'] for pos in positions.values())
            
            # 执行当日订单
            if current_date in orders_by_date.groups:
                daily_orders = orders_by_date.get_group(current_date)
                cash, positions = self._execute_orders(daily_orders, cash, positions, 
                                                     stock_data, current_date, result, portfolio_value)
            
            # 计算组合总价值
            portfolio_value = cash + sum(pos['market_value'] for pos in positions.values())
            
            # 获取当日股价用于记录
            current_prices = {}
            for stock_code in positions.keys():
                if stock_code in stock_data:
                    stock_df = stock_data[stock_code]
                    price = self._get_position_value_price(stock_df, current_date)
                    if price is not None:
                        current_prices[stock_code] = price
            
            # 记录每日数据
            result.add_daily_record(current_date, portfolio_value, cash, positions, current_prices)
            
            # 记录现金流
            result.add_cash_flow_record({
                'date': current_date,
                'cash': cash,
                'market_value': portfolio_value - cash,
                'total_value': portfolio_value
            })
        
        if self.logger:
            self.logger.info(f"回测完成，共执行 {len(result.trades)} 笔交易")
            
        return result
    
    def _update_positions_market_value(self, positions: Dict, stock_data: Dict[str, pd.DataFrame], 
                                     current_date: datetime):
        # 更新持仓市值
        for stock_code in list(positions.keys()):
            if stock_code in stock_data:
                stock_df = stock_data[stock_code]
                current_price = self._get_trade_price(stock_df, current_date)
                if current_price is not None:
                    positions[stock_code]['current_price'] = current_price
                    positions[stock_code]['market_value'] = positions[stock_code]['quantity'] * current_price
                else:
                    # 如果没有当日价格，使用前一日价格
                    positions[stock_code]['market_value'] = positions[stock_code]['quantity'] * positions[stock_code].get('current_price', positions[stock_code]['avg_price'])
            else:
                # 如果股票数据不存在，市值为0
                positions[stock_code]['market_value'] = 0
    
    def _execute_orders(self, daily_orders: pd.DataFrame, cash: float, positions: Dict,
                       stock_data: Dict[str, pd.DataFrame], current_date: datetime,
                       result: BacktestResult, portfolio_value: float) -> tuple:
        """执行当日订单"""
        
        # 先执行卖出订单
        sell_orders = daily_orders[daily_orders['action'] == 'sell']
        for _, order in sell_orders.iterrows():
            cash, positions = self._execute_sell_order(order, cash, positions, 
                                                     stock_data, current_date, result, portfolio_value)
        
        # 再执行买入订单
        buy_orders = daily_orders[daily_orders['action'] == 'buy']
        if not buy_orders.empty:
            cash, positions = self._execute_buy_orders(buy_orders, cash, positions,
                                                     stock_data, current_date, result, portfolio_value)
        
        return cash, positions
    
    def _execute_sell_order(self, order: pd.Series, cash: float, positions: Dict,
                           stock_data: Dict[str, pd.DataFrame], current_date: datetime,
                           result: BacktestResult, portfolio_value: float) -> tuple:
        """执行卖出订单"""
        stock_code = order['stock_code']
        
        if stock_code not in positions:
            return cash, positions
            
        if stock_code not in stock_data:
            if self.logger:
                self.logger.warning(f"股票 {stock_code} 数据不存在，无法卖出")
            return cash, positions
            
        stock_df = stock_data[stock_code]
        
        # 获取卖出价格
        sell_price = self._get_trade_price(stock_df, current_date)
        if sell_price is None:
            if self.logger:
                self.logger.warning(f"无法获取 {stock_code} 在 {current_date.strftime('%Y-%m-%d')} 的{self.trade_price_type}价格，跳过卖出")
            return cash, positions
        quantity = positions[stock_code]['quantity']
        avg_cost = positions[stock_code]['avg_price']
        
        # 计算交易金额和手续费
        trade_amount = quantity * sell_price
        commission = trade_amount * self.commission
        net_amount = trade_amount - commission
        
        # 计算收益率
        return_rate = self._calculate_return_rate(sell_price, avg_cost)
        
        # 计算持有天数
        holding_days = self._calculate_holding_days(stock_code, current_date, result.trades, self.trading_calendar)
        
        # 计算仓位占比
        position_ratio = (trade_amount / portfolio_value * 100) if portfolio_value > 0 else 0
        
        # 更新现金和持仓
        cash += net_amount
        
        # 记录交易
        trade_record = {
            'date': current_date,
            'stock_code': stock_code,
            'action': 'sell',
            'quantity': quantity,
            'price': sell_price,
            'amount': trade_amount,
            'commission': commission,
            'net_amount': net_amount,
            'position_ratio': position_ratio,
            'avg_cost': avg_cost,
            'return_rate': return_rate,
            'holding_days': holding_days
        }
        result.add_trade_record(trade_record)
        
        # 删除持仓
        del positions[stock_code]
        
        if self.logger:
            self.logger.debug(f"卖出 {stock_code}: 数量={quantity}, 价格={sell_price:.2f}, "
                            f"金额={trade_amount:.2f}, 手续费={commission:.2f}")
        
        return cash, positions
    
    def _execute_buy_orders(self, buy_orders: pd.DataFrame, cash: float, positions: Dict,
                           stock_data: Dict[str, pd.DataFrame], current_date: datetime,
                           result: BacktestResult, portfolio_value: float) -> tuple:
        """执行买入订单（包含再平衡逻辑）"""
        
        # 计算总资产价值（现金 + 持仓市值）
        total_portfolio_value = cash + sum(pos['market_value'] for pos in positions.values())
        
        # 计算可用资金（扣除预留现金）
        available_cash = total_portfolio_value * 0.99  # 预留1%现金
        
        # 按权重分配资金
        total_weight = buy_orders['weight'].sum()
        if total_weight <= 0:
            return cash, positions
        
        # 准备再平衡操作列表
        rebalance_sells = []  # 需要减仓的操作
        rebalance_buys = []   # 需要加仓的操作
        new_positions = []    # 需要新建的仓位
        
        # 第一步：分类所有操作
        for _, order in buy_orders.iterrows():
            stock_code = order['stock_code']
            weight = order['weight']
            
            if stock_code not in stock_data:
                if self.logger:
                    self.logger.warning(f"股票 {stock_code} 数据不存在，无法买入")
                continue
                
            stock_df = stock_data[stock_code]
            
            # 获取交易价格
            price = self._get_trade_price(stock_df, current_date)
            if price is None:
                if self.logger:
                    self.logger.warning(f"无法获取 {stock_code} 在 {current_date.strftime('%Y-%m-%d')} 的{self.trade_price_type}价格，跳过该股票")
                continue
            
            # 计算目标投资金额
            target_amount = available_cash * (weight / total_weight)
            
            # 检查是否已持仓该股票
            if stock_code in positions:
                # 已持仓，进行再平衡
                current_value = positions[stock_code]['market_value']
                adjustment_amount = target_amount - current_value
                
                if adjustment_amount > 0:
                    # 需要加仓
                    rebalance_buys.append({
                        'stock_code': stock_code,
                        'adjustment_amount': adjustment_amount,
                        'price': price
                    })
                elif adjustment_amount < 0:
                    # 需要减仓
                    rebalance_sells.append({
                        'stock_code': stock_code,
                        'adjustment_amount': abs(adjustment_amount),
                        'price': price
                    })
                # adjustment_amount == 0 时不需要调整
            else:
                # 新建仓位
                new_positions.append({
                    'stock_code': stock_code,
                    'target_amount': target_amount,
                    'price': price
                })
        
        # 第二步：先执行所有减仓操作，释放资金
        if self.logger and rebalance_sells:
            self.logger.debug(f"{current_date.strftime('%Y-%m-%d')} 开始执行 {len(rebalance_sells)} 个减仓操作")
        
        for sell_op in rebalance_sells:
            cash, positions = self._execute_rebalance_sell(
                sell_op['stock_code'], sell_op['adjustment_amount'], sell_op['price'], 
                cash, positions, current_date, result, portfolio_value)
        
        # 第三步：再执行所有加仓操作
        if self.logger and rebalance_buys:
            self.logger.debug(f"{current_date.strftime('%Y-%m-%d')} 开始执行 {len(rebalance_buys)} 个加仓操作")
        
        for buy_op in rebalance_buys:
            cash, positions = self._execute_rebalance_buy(
                buy_op['stock_code'], buy_op['adjustment_amount'], buy_op['price'], 
                cash, positions, current_date, result, portfolio_value)
        
        # 第四步：最后执行新建仓位操作
        if self.logger and new_positions:
            self.logger.debug(f"{current_date.strftime('%Y-%m-%d')} 开始执行 {len(new_positions)} 个新建仓位操作")
        
        for new_op in new_positions:
            cash, positions = self._execute_new_position(
                new_op['stock_code'], new_op['target_amount'], new_op['price'], 
                cash, positions, current_date, result, portfolio_value)
        
        return cash, positions
    
    def _execute_rebalance_buy(self, stock_code: str, adjustment_amount: float, 
                              buy_price: float, cash: float, positions: Dict,
                              current_date: datetime, result: BacktestResult, portfolio_value: float) -> tuple:
        """执行再平衡买入"""
        # 计算买入数量（向下取整到最小交易单位）
        target_quantity = int(adjustment_amount / buy_price)
        quantity = (target_quantity // self.min_trade_unit) * self.min_trade_unit
        
        if quantity <= 0:
            return cash, positions
            
        # 计算实际交易金额和手续费
        trade_amount = quantity * buy_price
        commission = trade_amount * self.commission
        total_cost = trade_amount + commission
        
        # 检查现金是否足够
        if total_cost > cash:
            quantity = (int(cash / (buy_price * (1 + self.commission))) // self.min_trade_unit) * self.min_trade_unit
            trade_amount = quantity * buy_price
            commission = trade_amount * self.commission
            total_cost = trade_amount + trade_amount * self.commission
            if total_cost > cash:
                if self.logger:
                    self.logger.warning(f"现金不足，无法加仓 {stock_code}")
                return cash, positions
        
        # 计算仓位占比
        position_ratio = (trade_amount / portfolio_value * 100) if portfolio_value > 0 else 0
        
        # 更新现金
        cash -= total_cost
        
        # 更新现有持仓
        old_quantity = positions[stock_code]['quantity']
        old_avg_price = positions[stock_code]['avg_price']
        new_quantity = old_quantity + quantity
        new_avg_price = (old_quantity * old_avg_price + trade_amount) / new_quantity
        
        positions[stock_code]['quantity'] = new_quantity
        positions[stock_code]['avg_price'] = new_avg_price
        positions[stock_code]['current_price'] = buy_price
        positions[stock_code]['market_value'] = new_quantity * buy_price
        
        # 记录交易
        trade_record = {
            'date': current_date,
            'stock_code': stock_code,
            'action': 'rebalance_buy',
            'quantity': quantity,
            'price': buy_price,
            'amount': trade_amount,
            'commission': commission,
            'total_cost': total_cost,
            'position_ratio': position_ratio
        }
        result.add_trade_record(trade_record)
        
        if self.logger:
            self.logger.debug(f"{current_date.strftime('%Y-%m-%d')} 再平衡加仓 {stock_code}: 数量={quantity}, 价格={buy_price:.2f}")
        
        return cash, positions
    
    def _execute_rebalance_sell(self, stock_code: str, adjustment_amount: float, 
                               sell_price: float, cash: float, positions: Dict,
                               current_date: datetime, result: BacktestResult, portfolio_value: float) -> tuple:
        """执行再平衡卖出"""
        current_quantity = positions[stock_code]['quantity']
        avg_cost = positions[stock_code]['avg_price']
        target_quantity = int(adjustment_amount / sell_price)
        quantity = min(target_quantity, current_quantity)
        quantity = (quantity // self.min_trade_unit) * self.min_trade_unit
        
        if quantity <= 0:
            return cash, positions
        
        # 计算交易金额和手续费
        trade_amount = quantity * sell_price
        commission = trade_amount * self.commission
        net_amount = trade_amount - commission
        
        # 计算收益率
        return_rate = self._calculate_return_rate(sell_price, avg_cost)
        
        # 计算持有天数
        holding_days = self._calculate_holding_days(stock_code, current_date, result.trades, self.trading_calendar)
        
        # 计算仓位占比
        position_ratio = (trade_amount / portfolio_value * 100) if portfolio_value > 0 else 0
        
        # 更新现金
        cash += net_amount
        
        # 更新持仓
        new_quantity = current_quantity - quantity
        if new_quantity > 0:
            positions[stock_code]['quantity'] = new_quantity
            positions[stock_code]['current_price'] = sell_price
            positions[stock_code]['market_value'] = new_quantity * sell_price
        else:
            # 完全卖出
            del positions[stock_code]
        
        # 记录交易
        trade_record = {
            'date': current_date,
            'stock_code': stock_code,
            'action': 'rebalance_sell',
            'quantity': quantity,
            'price': sell_price,
            'amount': trade_amount,
            'commission': commission,
            'net_amount': net_amount,
            'position_ratio': position_ratio,
            'avg_cost': avg_cost,
            'return_rate': return_rate,
            'holding_days': holding_days
        }
        result.add_trade_record(trade_record)
        
        if self.logger:
            self.logger.debug(f"{current_date.strftime('%Y-%m-%d')} 再平衡减仓 {stock_code}: 数量={quantity}, 价格={sell_price:.2f}")
        
        return cash, positions
    
    def _execute_new_position(self, stock_code: str, target_amount: float, 
                             buy_price: float, cash: float, positions: Dict,
                             current_date: datetime, result: BacktestResult, portfolio_value: float) -> tuple:
        """执行新建仓位"""
        # 计算买入数量（向下取整到最小交易单位）
        target_quantity = int(target_amount / buy_price)
        quantity = (target_quantity // self.min_trade_unit) * self.min_trade_unit
        
        if quantity <= 0:
            return cash, positions
            
        # 计算实际交易金额和手续费
        trade_amount = quantity * buy_price
        commission = trade_amount * self.commission
        total_cost = trade_amount + commission
        
        # 检查现金是否足够
        if total_cost > cash:
            quantity = (int(cash / (buy_price * (1 + self.commission))) // self.min_trade_unit) * self.min_trade_unit
            trade_amount = quantity * buy_price
            commission = trade_amount * self.commission
            total_cost = trade_amount + trade_amount * self.commission
            if total_cost > cash:
                if self.logger:
                    self.logger.warning(f"现金不足，无法买入 {stock_code}")
                return cash, positions
        
        # 计算仓位占比
        position_ratio = (trade_amount / portfolio_value * 100) if portfolio_value > 0 else 0
        
        # 更新现金
        cash -= total_cost
        
        # 新建持仓
        positions[stock_code] = {
            'quantity': quantity,
            'avg_price': buy_price,
            'current_price': buy_price,
            'market_value': quantity * buy_price
        }
        
        # 记录交易
        trade_record = {
            'date': current_date,
            'stock_code': stock_code,
            'action': 'buy',
            'quantity': quantity,
            'price': buy_price,
            'amount': trade_amount,
            'commission': commission,
            'total_cost': total_cost,
            'position_ratio': position_ratio
        }
        result.add_trade_record(trade_record)
        
        if self.logger:
            self.logger.debug(f"{current_date.strftime('%Y-%m-%d')} 新建仓位 {stock_code}: 数量={quantity}, 价格={buy_price:.2f}")
        
        return cash, positions
