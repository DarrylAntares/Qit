"""
Feather格式数据源实现
"""
import os
import glob
import pandas as pd
import pyarrow.feather as feather
from datetime import datetime
from typing import Dict, List, Optional
from tqdm import tqdm

try:
    from .base_data_source import BaseDataSource
    from ..utils.date_utils import DateUtils
except ImportError:
    # 当直接运行时的绝对导入
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_source.base_data_source import BaseDataSource
    from utils.date_utils import DateUtils

class FeatherDataSource(BaseDataSource):
    """Feather格式数据源"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.data_path = config.get('DATA_PATH', 'dataset/stock_daily_forward_feather')
        self.calendar_path = config.get('CALENDAR_PATH', 'dataset/calendars/calendars.csv')
        self.index_weight_path = config.get('INDEX_WEIGHT_PATH', 'dataset/indice_weight')
        self.max_stocks = config.get('MAX_STOCKS', None)
    
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
        if self.logger:
            self.logger.info(f"开始加载股票数据，日期范围: {start_date} 到 {end_date}")
        
        # 获取所有股票文件
        all_files = [f for f in glob.glob(os.path.join(self.data_path, "*.feather")) 
                    if f.endswith('_SH.feather') or f.endswith('_SZ.feather')]
        
        # 限制股票数量
        if self.max_stocks and self.max_stocks < len(all_files):
            all_files = all_files[:self.max_stocks]
        
        if self.logger:
            self.logger.info(f"找到 {len(all_files)} 个股票文件")
        
        stock_data = {}
        loaded_count = 0
        
        # 使用进度条显示加载进度
        with tqdm(total=len(all_files), desc="加载股票数据", unit="股票") as pbar:
            for file_path in all_files:
                try:
                    # 读取数据
                    df = pd.read_feather(file_path)
                    
                    # 转换日期格式
                    df['trade_date'] = pd.to_datetime(df['trade_date'])
                    df.set_index('trade_date', inplace=True)
                    
                    # 过滤日期范围
                    df = df[(df.index >= start_date) & (df.index <= end_date)]
                    
                    if len(df) > 0:
                        # 获取股票代码
                        stock_code = df['ts_code'].iloc[0] if 'ts_code' in df.columns else os.path.basename(file_path).split('.')[0]
                        
                        # 如果指定了股票代码列表，则过滤
                        if stock_codes and stock_code not in stock_codes:
                            pbar.update(1)
                            continue
                        
                        # 只保留需要的列
                        df = df[['open', 'high', 'low', 'close', 'vol']]
                        
                        # 确保数据质量
                        df = df.dropna()
                        
                        if len(df) > 0:
                            stock_data[stock_code] = df
                            loaded_count += 1
                    
                    pbar.set_postfix_str(f"已加载: {loaded_count}")
                    pbar.update(1)
                    
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"加载文件 {file_path} 失败: {str(e)}")
                    pbar.update(1)
                    continue
        
        if self.logger:
            self.logger.info(f"成功加载 {loaded_count} 只股票的数据")
        
        return stock_data
    
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
        if self.logger:
            self.logger.info(f"开始加载 {index_code} 指数成分股数据")
        
        constituent_file = os.path.join(self.index_weight_path, f'{index_code.lower()}.txt')
        
        try:
            # 读取成分股数据
            df = pd.read_csv(constituent_file, sep='\t', header=None, 
                           names=['code', 'start_date', 'end_date'],
                           parse_dates=['start_date', 'end_date'])
            
            # 将日期格式化为字符串
            df['start_date'] = df['start_date'].dt.strftime('%Y-%m-%d')
            df['end_date'] = df['end_date'].dt.strftime('%Y-%m-%d')
            
            constituents = {}
            
            # 按日期分组，生成每个交易日的成分股列表
            for _, row in df.iterrows():
                start_dt = pd.to_datetime(row['start_date'])
                end_dt = pd.to_datetime(row['end_date'])
                code = row['code']
                
                # 生成日期范围
                date_range = pd.date_range(start=start_dt, end=end_dt)
                for d in date_range:
                    if start_date <= d <= end_date:
                        date_str = d.strftime('%Y-%m-%d')
                        if date_str not in constituents:
                            constituents[date_str] = []
                        constituents[date_str].append(code)
            
            if self.logger:
                self.logger.info(f"成功加载 {len(df)} 条成分股数据，覆盖 {len(constituents)} 个交易日")
            
            return constituents
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"加载指数成分股数据失败: {str(e)}")
            raise
    
    def load_trading_calendar(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        """
        加载交易日历
        
        参数:
            start_date: 开始日期
            end_date: 结束日期
            
        返回:
            List[datetime]: 交易日列表
        """
        if self.logger:
            self.logger.info(f"开始加载交易日历，日期范围: {start_date} 到 {end_date}")
        
        try:
            trading_dates = DateUtils.load_trading_calendar(self.calendar_path, start_date, end_date)
            
            if self.logger:
                self.logger.info(f"成功加载交易日历，共 {len(trading_dates)} 个交易日")
            
            # 转换为datetime对象
            return [datetime.combine(d, datetime.min.time()) for d in trading_dates]
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"加载交易日历失败: {str(e)}")
            raise
