"""
股票信息数据源
读取A股基本信息数据，包括证券代码、简称、上市板块、行业分类等
"""
import pandas as pd
import os
from typing import Dict, List, Optional
from datetime import datetime
from .base_data_source import BaseDataSource
from config.config import Config


class StockInfoDataSource(BaseDataSource):
    """股票信息数据源类"""
    
    def __init__(self, config: Dict = None):
        """初始化股票信息数据源"""
        if config is None:
            config = {}
        super().__init__(config)
        self.stock_info_path = Config.STOCK_INFO_PATH
        self._stock_info_data = None
        self._load_stock_info()
    
    def _load_stock_info(self):
        """加载股票信息数据"""
        try:
            if not os.path.exists(self.stock_info_path):
                print(f"警告：股票信息文件不存在: {self.stock_info_path}")
                self._stock_info_data = pd.DataFrame()
                return
            
            print(f"正在加载股票信息数据: {self.stock_info_path}")
            
            # 读取Excel文件
            self._stock_info_data = pd.read_csv(self.stock_info_path)
            
            # 检查必要字段
            required_columns = ['ts_code', 'ts_name', 'list_market', 'sw_industry_1', 'sw_industry_2']
            missing_columns = [col for col in required_columns if col not in self._stock_info_data.columns]
            
            if missing_columns:
                print(f"警告：股票信息文件缺少字段: {missing_columns}")
                print(f"实际字段: {list(self._stock_info_data.columns)}")
            
            # 标准化字段名（处理可能的拼写错误）
            column_mapping = {
                'ts_cide': 'ts_code',  # 修正可能的拼写错误
                'ts_sname': 'ts_name',  # 修正可能的拼写错误
                'sw_indsutry_2': 'sw_industry_2'  # 修正可能的拼写错误
            }
            
            for old_name, new_name in column_mapping.items():
                if old_name in self._stock_info_data.columns:
                    self._stock_info_data = self._stock_info_data.rename(columns={old_name: new_name})
                    print(f"字段名修正: {old_name} -> {new_name}")
            
            # 确保ts_code为字符串格式
            if 'ts_code' in self._stock_info_data.columns:
                self._stock_info_data['ts_code'] = self._stock_info_data['ts_code'].astype(str)
                # 设置ts_code为索引，便于快速查询
                self._stock_info_data = self._stock_info_data.set_index('ts_code')
            
            print(f"股票信息数据加载完成，共 {len(self._stock_info_data)} 条记录")
            print(f"数据字段: {list(self._stock_info_data.columns)}")
            
            # 显示数据概览
            if len(self._stock_info_data) > 0:
                print("\n数据概览:")
                print(f"  - 证券数量: {len(self._stock_info_data)}")
                if 'list_market' in self._stock_info_data.columns:
                    market_counts = self._stock_info_data['list_market'].value_counts()
                    print(f"  - 上市板块分布: {dict(market_counts)}")
                if 'sw_industry_1' in self._stock_info_data.columns:
                    industry_counts = self._stock_info_data['sw_industry_1'].value_counts()
                    print(f"  - 一级行业数量: {len(industry_counts)}")
                    print(f"  - 主要行业: {dict(industry_counts.head())}")
            
        except Exception as e:
            print(f"加载股票信息数据时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            self._stock_info_data = pd.DataFrame()
    
    def get_stock_info(self, ts_code: str) -> Optional[Dict]:
        """获取单个股票的信息
        
        参数:
            ts_code: 股票代码
            
        返回:
            股票信息字典，如果不存在则返回None
        """
        if self._stock_info_data is None or len(self._stock_info_data) == 0:
            return None
        
        try:
            if ts_code in self._stock_info_data.index:
                stock_info = self._stock_info_data.loc[ts_code]
                return {
                    'ts_code': ts_code,
                    'ts_name': stock_info.get('ts_name', ''),
                    'list_market': stock_info.get('list_market', ''),
                    'sw_industry_1': stock_info.get('sw_industry_1', ''),
                    'sw_industry_2': stock_info.get('sw_industry_2', '')
                }
            else:
                return None
        except Exception as e:
            print(f"获取股票信息时出错 {ts_code}: {str(e)}")
            return None
    
    def get_stocks_by_market(self, market: str) -> List[str]:
        """根据上市板块获取股票列表
        
        参数:
            market: 上市板块名称
            
        返回:
            股票代码列表
        """
        if self._stock_info_data is None or len(self._stock_info_data) == 0:
            return []
        
        try:
            if 'list_market' in self._stock_info_data.columns:
                filtered_stocks = self._stock_info_data[
                    self._stock_info_data['list_market'] == market
                ]
                return filtered_stocks.index.tolist()
            else:
                return []
        except Exception as e:
            print(f"根据板块筛选股票时出错 {market}: {str(e)}")
            return []
    
    def get_stocks_by_industry(self, industry: str, level: int = 1) -> List[str]:
        """根据行业分类获取股票列表
        
        参数:
            industry: 行业名称
            level: 行业级别，1为一级行业，2为二级行业
            
        返回:
            股票代码列表
        """
        if self._stock_info_data is None or len(self._stock_info_data) == 0:
            return []
        
        try:
            column_name = f'sw_industry_{level}'
            if column_name in self._stock_info_data.columns:
                filtered_stocks = self._stock_info_data[
                    self._stock_info_data[column_name] == industry
                ]
                return filtered_stocks.index.tolist()
            else:
                return []
        except Exception as e:
            print(f"根据行业筛选股票时出错 {industry}: {str(e)}")
            return []
    
    def get_all_markets(self) -> List[str]:
        """获取所有上市板块列表
        
        返回:
            上市板块列表
        """
        if self._stock_info_data is None or len(self._stock_info_data) == 0:
            return []
        
        try:
            if 'list_market' in self._stock_info_data.columns:
                return self._stock_info_data['list_market'].dropna().unique().tolist()
            else:
                return []
        except Exception as e:
            print(f"获取板块列表时出错: {str(e)}")
            return []
    
    def get_all_industries(self, level: int = 1) -> List[str]:
        """获取所有行业分类列表
        
        参数:
            level: 行业级别，1为一级行业，2为二级行业
            
        返回:
            行业分类列表
        """
        if self._stock_info_data is None or len(self._stock_info_data) == 0:
            return []
        
        try:
            column_name = f'sw_industry_{level}'
            if column_name in self._stock_info_data.columns:
                return self._stock_info_data[column_name].dropna().unique().tolist()
            else:
                return []
        except Exception as e:
            print(f"获取行业列表时出错: {str(e)}")
            return []
    
    def get_stock_count_by_market(self) -> Dict[str, int]:
        """获取各板块的股票数量统计
        
        返回:
            板块股票数量字典
        """
        if self._stock_info_data is None or len(self._stock_info_data) == 0:
            return {}
        
        try:
            if 'list_market' in self._stock_info_data.columns:
                return self._stock_info_data['list_market'].value_counts().to_dict()
            else:
                return {}
        except Exception as e:
            print(f"统计板块股票数量时出错: {str(e)}")
            return {}
    
    def get_stock_count_by_industry(self, level: int = 1) -> Dict[str, int]:
        """获取各行业的股票数量统计
        
        参数:
            level: 行业级别，1为一级行业，2为二级行业
            
        返回:
            行业股票数量字典
        """
        if self._stock_info_data is None or len(self._stock_info_data) == 0:
            return {}
        
        try:
            column_name = f'sw_industry_{level}'
            if column_name in self._stock_info_data.columns:
                return self._stock_info_data[column_name].value_counts().to_dict()
            else:
                return {}
        except Exception as e:
            print(f"统计行业股票数量时出错: {str(e)}")
            return {}
    
    def search_stocks(self, keyword: str, search_in: str = 'name') -> List[Dict]:
        """搜索股票
        
        参数:
            keyword: 搜索关键词
            search_in: 搜索字段，'name'为股票名称，'code'为股票代码
            
        返回:
            匹配的股票信息列表
        """
        if self._stock_info_data is None or len(self._stock_info_data) == 0:
            return []
        
        try:
            results = []
            if search_in == 'name' and 'ts_name' in self._stock_info_data.columns:
                # 在股票名称中搜索
                mask = self._stock_info_data['ts_name'].str.contains(keyword, na=False)
                matched_stocks = self._stock_info_data[mask]
            elif search_in == 'code':
                # 在股票代码中搜索
                mask = self._stock_info_data.index.str.contains(keyword, na=False)
                matched_stocks = self._stock_info_data[mask]
            else:
                return []
            
            for ts_code, row in matched_stocks.iterrows():
                results.append({
                    'ts_code': ts_code,
                    'ts_name': row.get('ts_name', ''),
                    'list_market': row.get('list_market', ''),
                    'sw_industry_1': row.get('sw_industry_1', ''),
                    'sw_industry_2': row.get('sw_industry_2', '')
                })
            
            return results
        except Exception as e:
            print(f"搜索股票时出错: {str(e)}")
            return []
    
    def get_data_summary(self) -> Dict:
        """获取数据概览信息
        
        返回:
            数据概览字典
        """
        if self._stock_info_data is None or len(self._stock_info_data) == 0:
            return {
                'total_stocks': 0,
                'markets': [],
                'industries_level1': [],
                'industries_level2': [],
                'data_loaded': False
            }
        
        try:
            return {
                'total_stocks': len(self._stock_info_data),
                'markets': self.get_all_markets(),
                'industries_level1': self.get_all_industries(1),
                'industries_level2': self.get_all_industries(2),
                'market_distribution': self.get_stock_count_by_market(),
                'industry_distribution_level1': self.get_stock_count_by_industry(1),
                'data_loaded': True,
                'data_path': self.stock_info_path
            }
        except Exception as e:
            print(f"获取数据概览时出错: {str(e)}")
            return {'data_loaded': False, 'error': str(e)}
    
    def load_data(self, start_date=None, end_date=None, symbols=None):
        """实现基类的抽象方法（股票信息数据不需要日期范围）"""
        return self._stock_info_data
    
    def get_available_symbols(self):
        """获取可用的股票代码列表"""
        if self._stock_info_data is None or len(self._stock_info_data) == 0:
            return []
        return self._stock_info_data.index.tolist()
    
    # 实现BaseDataSource的抽象方法
    def load_stock_data(self, start_date: datetime, end_date: datetime, 
                       stock_codes: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        股票信息数据源不提供时间序列数据，返回空字典
        """
        if self.logger:
            self.logger.warning("StockInfoDataSource不提供时间序列股票数据")
        return {}
    
    def load_index_constituents(self, index_code: str, start_date: datetime, 
                               end_date: datetime) -> Dict[str, List[str]]:
        """
        股票信息数据源不提供指数成分股时间序列，返回空字典
        """
        if self.logger:
            self.logger.warning("StockInfoDataSource不提供指数成分股时间序列数据")
        return {}
    
    def load_trading_calendar(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        """
        股票信息数据源不提供交易日历，返回空列表
        """
        if self.logger:
            self.logger.warning("StockInfoDataSource不提供交易日历数据")
        return []
