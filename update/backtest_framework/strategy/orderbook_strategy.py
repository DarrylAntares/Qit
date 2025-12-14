"""
订单簿策略实现
基于外部订单文件的策略，支持T+1交易逻辑
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import os

try:
    from .base_strategy import BaseStrategy
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from strategy.base_strategy import BaseStrategy

class OrderBookStrategy(BaseStrategy):
    """订单簿策略"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.order_file_path = config.get('ORDER_FILE_PATH', 'orderbook/order.xlsx')
        self.signal_data = None  # 存储信号数据
        
        # 信号生成模式配置
        self.signal_generation_mode = config.get('SIGNAL_GENERATION_MODE', 'T+1')
        
        # 涨停价剔除逻辑配置
        self.enable_limit_up_filter = config.get('ENABLE_LIMIT_UP_FILTER', False)
        
        # 验证信号生成模式
        valid_modes = ['T', 'T+1']
        if self.signal_generation_mode not in valid_modes:
            if self.logger:
                self.logger.warning(f"无效的信号生成模式: {self.signal_generation_mode}, 使用默认值 'T+1'")
            self.signal_generation_mode = 'T+1'
        
        if self.logger:
            self.logger.info(f"信号生成模式: {self.signal_generation_mode}")
            if self.signal_generation_mode == 'T':
                self.logger.info("当日交易模式: T日信号生成T日交易")
            else:
                self.logger.info("次日交易模式: T日信号生成T+1日交易")
            
            self.logger.info(f"涨停价剔除逻辑: {'启用' if self.enable_limit_up_filter else '禁用'}")
        
        self.load_order_data()
    
    def load_order_data(self):
        """加载订单数据"""
        try:
            # 检查文件是否存在
            if os.path.exists(self.order_file_path):
                # 根据文件扩展名选择读取方式
                if self.order_file_path.endswith('.csv'):
                    df = pd.read_csv(self.order_file_path)
                    if self.logger:
                        self.logger.info(f"使用CSV格式读取订单文件: {self.order_file_path}")
                else:
                    # 尝试读取Excel文件
                    try:
                        import openpyxl
                        df = pd.read_excel(self.order_file_path, engine='openpyxl')
                    except ImportError:
                        # 如果没有openpyxl，尝试使用xlrd
                        df = pd.read_excel(self.order_file_path)
                
                if self.logger:
                    self.logger.info(f"成功加载订单文件: {self.order_file_path}")
                    self.logger.info(f"订单数据形状: {df.shape}")
                    self.logger.info(f"列名: {df.columns.tolist()}")
                
                # 假设第一列是交易日期，第二列是股票代码
                df.columns = ['trade_date', 'ts_code']
                
                # 统一的日期解析方式（与main_orderbook.py保持一致）
                df['trade_date'] = df['trade_date'].astype(str)
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                
                # 转换股票代码格式：SH000000 -> 000000.SH, SZ000000 -> 000000.SZ
                df['stock_code'] = df['ts_code'].apply(self._convert_stock_code)
                
                # 按日期分组，每个日期对应一个股票列表
                self.signal_data = df.groupby('trade_date')['stock_code'].apply(list).to_dict()
                
                if self.logger:
                    self.logger.info(f"处理后的信号数据包含 {len(self.signal_data)} 个交易日")
                    
            else:
                if self.logger:
                    self.logger.error(f"订单文件不存在: {self.order_file_path}")
                self.signal_data = {}
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"加载订单数据时发生错误: {str(e)}")
            self.signal_data = {}
    
    def _convert_stock_code(self, ts_code: str) -> str:
        """
        转换股票代码格式
        SH000000 -> 000000.SH
        SZ000000 -> 000000.SZ
        """
        if isinstance(ts_code, str) and len(ts_code) >= 8:
            if ts_code.startswith('SH'):
                return ts_code[2:] + '.SH'
            elif ts_code.startswith('SZ'):
                return ts_code[2:] + '.SZ'
        return ts_code
    
    def _get_limit_up_ratio(self, stock_code: str) -> float:
        """
        根据股票代码获取涨停幅度
        
        参数:
            stock_code: 股票代码，格式如 '000001.SZ' 或 '600000.SH'
            
        返回:
            float: 涨停幅度（小数形式，如0.1表示10%，0.2表示20%）
        """
        if not isinstance(stock_code, str):
            return 0.1  # 默认10%
        
        # 提取股票代码数字部分
        code_part = stock_code.split('.')[0] if '.' in stock_code else stock_code
        
        # 判断是否为创业板（300开头）或科创板（688开头）或主板（600开头）
        if code_part.startswith('300') or code_part.startswith('688'):
            return 0.2  # 20%涨停
        elif code_part.startswith('600'):
            return 0.2  # 20%涨停（根据您的需求，600开头也是20%）
        else:
            return 0.1  # 其他情况默认10%涨停
    
    def _is_limit_up_price(self, stock_code: str, current_price: float, 
                          prev_close_price: float) -> bool:
        """
        判断当前价格是否为涨停价
        
        参数:
            stock_code: 股票代码
            current_price: 当前价格（通常是开盘价）
            prev_close_price: 昨日收盘价
            
        返回:
            bool: True表示是涨停价，False表示不是
        """
        if pd.isna(current_price) or pd.isna(prev_close_price) or prev_close_price <= 0:
            return False
        
        # 获取涨停幅度
        limit_up_ratio = self._get_limit_up_ratio(stock_code)
        
        # 计算理论涨停价（四舍五入到分）
        theoretical_limit_up = round(prev_close_price * (1 + limit_up_ratio), 2)
        
        # 判断当前价格是否等于涨停价（允许小的浮点数误差）
        is_limit_up = abs(current_price - theoretical_limit_up) < 0.005
        
        return is_limit_up
    
    def _filter_limit_up_stocks(self, target_stocks: List[str], 
                               stock_data: Dict[str, pd.DataFrame],
                               current_date: datetime) -> List[str]:
        """
        剔除涨停价开盘的股票
        
        参数:
            target_stocks: 目标股票列表
            stock_data: 股票数据
            current_date: 当前日期
            
        返回:
            List[str]: 剔除涨停股票后的列表
        """
        if not self.enable_limit_up_filter:
            return target_stocks
        
        filtered_stocks = []
        filtered_count = 0
        
        for stock_code in target_stocks:
            if stock_code not in stock_data:
                continue
            
            stock_df = stock_data[stock_code]
            
            # 检查当前日期的数据是否存在
            if current_date not in stock_df.index:
                continue
            
            # 获取当前日期的开盘价
            current_open = stock_df.loc[current_date, 'open']
            if pd.isna(current_open):
                continue
            
            # 获取前一交易日的收盘价
            try:
                # 找到当前日期在DataFrame中的位置
                current_idx = stock_df.index.get_loc(current_date)
                if current_idx > 0:
                    prev_close = stock_df.iloc[current_idx - 1]['close']
                    
                    # 判断是否为涨停价
                    if self._is_limit_up_price(stock_code, current_open, prev_close):
                        filtered_count += 1
                        if self.logger:
                            limit_ratio = self._get_limit_up_ratio(stock_code)
                            self.logger.info(f"剔除涨停股票: {stock_code}, "
                                           f"开盘价: {current_open:.2f}, "
                                           f"昨收: {prev_close:.2f}, "
                                           f"涨停幅度: {limit_ratio*100:.0f}%")
                        continue
                
                # 如果不是涨停价，加入过滤后的列表
                filtered_stocks.append(stock_code)
                
            except (KeyError, IndexError):
                # 如果无法获取前一日收盘价，保留该股票
                filtered_stocks.append(stock_code)
        
        if self.logger and filtered_count > 0:
            self.logger.info(f"涨停价剔除: 原始股票数 {len(target_stocks)}, "
                           f"剔除 {filtered_count} 只涨停股票, "
                           f"剩余 {len(filtered_stocks)} 只")
        
        return filtered_stocks
    
    def get_signal_date_range(self):
        """获取信号数据的日期范围"""
        if not self.signal_data:
            return None, None
        
        dates = list(self.signal_data.keys())
        return min(dates), max(dates)
    
    def need_rebalance(self, current_date: datetime) -> bool:
        """
        判断是否需要调仓
        
        根据信号生成模式判断：
        - T模式：当日有信号则当日交易
        - T+1模式：前一交易日有信号则当日交易
        """
        try:
            # 将current_date转换为date对象进行比较
            if isinstance(current_date, datetime):
                current_date_obj = current_date.date()
            else:
                current_date_obj = current_date
                
            # 在交易日历中查找当前日期
            trading_dates = [d.date() if isinstance(d, datetime) else d for d in self.trading_calendar]
            
            if current_date_obj not in trading_dates:
                return False
            
            if self.signal_generation_mode == 'T':
                # T模式：检查当日是否有信号
                current_datetime = current_date if isinstance(current_date, datetime) else datetime.combine(current_date, datetime.min.time())
                has_signal = current_datetime in self.signal_data
                
                if self.logger and has_signal:
                    self.logger.info(f"{current_date.strftime('%Y-%m-%d')}: 检测到T日交易信号（当日交易）")
                
                return has_signal
                
            else:  # T+1模式
                # T+1模式：检查前一个交易日是否有信号
                current_idx = trading_dates.index(current_date_obj)
                if current_idx > 0:
                    prev_trading_day = self.trading_calendar[current_idx - 1]
                    # 将prev_trading_day转换为datetime进行比较
                    if hasattr(prev_trading_day, 'date'):
                        prev_trading_datetime = prev_trading_day
                    else:
                        prev_trading_datetime = datetime.combine(prev_trading_day, datetime.min.time())
                    
                    # 检查前一个交易日是否有信号
                    has_signal = prev_trading_datetime in self.signal_data
                    
                    if self.logger and has_signal:
                        self.logger.info(f"{current_date.strftime('%Y-%m-%d')}: 检测到T+1交易信号（次日交易）")
                    
                    return has_signal
                
            return False
        except (ValueError, IndexError) as e:
            if self.logger:
                self.logger.debug(f"need_rebalance检查失败: {e}")
            return False
    
    def generate_signals(self, stock_data: Dict[str, pd.DataFrame], 
                        current_date: datetime) -> Dict[str, str]:
        """
        生成交易信号
        
        根据信号生成模式的逻辑：
        - T模式：当日有信号则当日交易
        - T+1模式：前一交易日有信号则当日交易
        
        交易逻辑：
        1. 先清空所有持仓（卖出信号）
        2. 对信号中的股票生成买入信号
        
        参数:
            stock_data: 股票数据
            current_date: 当前日期
            
        返回:
            Dict[str, str]: 股票代码到信号的映射
        """
        signals = {}
        
        try:
            # 使用与need_rebalance相同的逻辑
            if isinstance(current_date, datetime):
                current_date_obj = current_date.date()
            else:
                current_date_obj = current_date
                
            trading_dates = [d.date() if isinstance(d, datetime) else d for d in self.trading_calendar]
            
            if current_date_obj not in trading_dates:
                return signals
            
            # 根据信号生成模式确定信号日期
            signal_datetime = None
            
            if self.signal_generation_mode == 'T':
                # T模式：检查当日是否有信号
                current_datetime = current_date if isinstance(current_date, datetime) else datetime.combine(current_date, datetime.min.time())
                if current_datetime in self.signal_data:
                    signal_datetime = current_datetime
                    
            else:  # T+1模式
                # T+1模式：检查前一个交易日是否有信号
                current_idx = trading_dates.index(current_date_obj)
                if current_idx > 0:
                    prev_trading_day = self.trading_calendar[current_idx - 1]
                    # 转换为datetime进行比较
                    if hasattr(prev_trading_day, 'date'):
                        prev_trading_datetime = prev_trading_day
                    else:
                        prev_trading_datetime = datetime.combine(prev_trading_day, datetime.min.time())
                    
                    if prev_trading_datetime in self.signal_data:
                        signal_datetime = prev_trading_datetime
            
            # 如果找到信号，生成交易信号
            if signal_datetime is not None:
                # 1. 对所有当前持仓生成卖出信号
                for stock_code in self.current_positions:
                    signals[stock_code] = 'sell'
                
                # 2. 对信号中的股票生成买入信号
                target_stocks = self.signal_data[signal_datetime]
                
                # 应用涨停价剔除逻辑
                if self.enable_limit_up_filter:
                    target_stocks = self._filter_limit_up_stocks(target_stocks, stock_data, current_date)
                
                # 过滤出在股票数据中存在且有效的股票
                valid_stocks = []
                for stock_code in target_stocks:
                    if stock_code in stock_data:
                        stock_df = stock_data[stock_code]
                        if current_date in stock_df.index and not pd.isna(stock_df.loc[current_date, 'open']):
                            valid_stocks.append(stock_code)
                            signals[stock_code] = 'buy'
                
                if self.logger:
                    mode_desc = "当日交易" if self.signal_generation_mode == 'T' else "次日交易"
                    self.logger.info(f"{current_date.strftime('%Y-%m-%d')}: {mode_desc}日")
                    self.logger.info(f"信号日期: {signal_datetime.strftime('%Y-%m-%d')}")
                    self.logger.info(f"目标股票数: {len(target_stocks)}, 有效股票数: {len(valid_stocks)}")
                    self.logger.info(f"清仓股票数: {len([s for s in signals.values() if s == 'sell'])}")
                    self.logger.info(f"买入股票数: {len([s for s in signals.values() if s == 'buy'])}")
        
        except (ValueError, IndexError):
            pass
        
        return signals
    
    def calculate_position_weights(self, signals: Dict[str, str], 
                                  stock_data: Dict[str, pd.DataFrame],
                                  current_date: datetime) -> Dict[str, float]:
        """
        计算持仓权重
        
        对于订单簿策略，采用等权重分配
        
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
        start_date, end_date = self.get_signal_date_range()
        
        # 构建描述信息
        description = f"基于外部订单文件的{self.signal_generation_mode}交易策略，信号日清仓并等权买入目标股票"
        if self.enable_limit_up_filter:
            description += "，启用涨停价剔除逻辑"
        
        return {
            'strategy_name': '订单簿策略',
            'strategy_type': '信号驱动',
            'signal_generation_mode': self.signal_generation_mode,
            'limit_up_filter': '启用' if self.enable_limit_up_filter else '禁用',
            'order_file': self.order_file_path,
            'signal_date_range': f"{start_date} 到 {end_date}" if start_date else "无数据",
            'total_signal_days': len(self.signal_data) if self.signal_data else 0,
            'description': description
        }
