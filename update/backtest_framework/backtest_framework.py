"""
回测框架主类
"""
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from config.config import Config
from utils.logger import Logger
from data_source.feather_data_source import FeatherDataSource
from data_source.stock_info_data_source import StockInfoDataSource
from data_processor.standard_data_processor import StandardDataProcessor
from strategy.hs300_equal_weight_strategy import HS300EqualWeightStrategy
from backtest_engine.simple_backtest_engine import SimpleBacktestEngine
from performance.standard_performance_analyzer import StandardPerformanceAnalyzer

class BacktestFramework:
    """回测框架主类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化回测框架
        
        参数:
            config: 配置字典，如果为None则使用默认配置
        """
        # 使用传入的配置或默认配置
        if config is None:
            self.config = Config.__dict__.copy()
        else:
            self.config = {**Config.__dict__, **config}
        
        # 初始化日志
        self.logger_manager = Logger(
            log_file=Config.get_log_file(),
            log_level=self.config.get('LOG_LEVEL', 'DEBUG'),
            enable_console=True,  # 启用控制台输出
            console_level=self.config.get('CONSOLE_LOG_LEVEL', 'WARNING')  # 控制台只显示WARNING及以上级别
        )
        self.logger = self.logger_manager.get_logger('BacktestFramework')
        
        # 初始化各层组件
        self.data_source = None
        self.stock_info_source = None  # 股票信息数据源
        self.data_processor = None
        self.strategy = None
        self.backtest_engine = None
        self.performance_analyzer = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """初始化各层组件"""
        self.logger.info("初始化回测框架组件")
        
        # 初始化数据源层
        self.data_source = FeatherDataSource(self.config)
        self.data_source.set_logger(self.logger_manager.get_logger('DataSource'))
        
        # 初始化股票信息数据源
        self.stock_info_source = StockInfoDataSource(self.config)
        self.stock_info_source.set_logger(self.logger_manager.get_logger('StockInfoSource'))
        
        # 初始化数据处理层
        self.data_processor = StandardDataProcessor(self.config)
        self.data_processor.set_logger(self.logger_manager.get_logger('DataProcessor'))
        
        # 初始化策略层
        self.strategy = HS300EqualWeightStrategy(self.config)
        self.strategy.set_logger(self.logger_manager.get_logger('Strategy'))
        
        # 初始化回测引擎
        self.backtest_engine = SimpleBacktestEngine(self.config)
        self.backtest_engine.set_logger(self.logger_manager.get_logger('BacktestEngine'))
        
        # 初始化绩效分析器
        self.performance_analyzer = StandardPerformanceAnalyzer(self.config)
        self.performance_analyzer.set_logger(self.logger_manager.get_logger('PerformanceAnalyzer'))
        
        self.logger.info("回测框架组件初始化完成")
    
    def run_backtest(self, start_date: Optional[datetime] = None, 
                    end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        运行完整的回测流程
        
        参数:
            start_date: 回测开始日期，如果为None则使用配置中的日期
            end_date: 回测结束日期，如果为None则使用配置中的日期
            
        返回:
            Dict[str, Any]: 回测结果和分析结果
        """
        # 使用传入的日期或配置中的日期
        start_date = start_date or self.config.get('START_DATE')
        end_date = end_date or self.config.get('END_DATE')
        
        self.logger.info(f"开始回测流程: {start_date} 到 {end_date}")
        
        try:
            # 第一步：加载数据
            self.logger.info("=" * 50)
            self.logger.info("第一步：数据加载")
            self.logger.info("=" * 50)
            
            # 加载交易日历
            trading_calendar = self.data_source.load_trading_calendar(start_date, end_date)
            
            # 加载指数成分股
            constituents = self.data_source.load_index_constituents('hs300', start_date, end_date)
            
            # 加载股票数据
            stock_data = self.data_source.load_stock_data(start_date, end_date)
            
            # 第二步：数据处理（修正处理顺序）
            self.logger.info("=" * 50)
            self.logger.info("第二步：数据处理")
            self.logger.info("=" * 50)
            
            # 第一步：先与交易日历对齐
            aligned_raw_data = self.data_processor.align_data_with_calendar_first(stock_data, trading_calendar)
            
            # 第二步：数据清洗和缺值填充
            cleaned_data = self.data_processor.clean_data(aligned_raw_data)
            
            # 第三步：计算因子
            factor_data = self.data_processor.calculate_factors(cleaned_data)
            
            # 第四步：最终数据对齐
            aligned_data = self.data_processor.align_data_with_calendar(factor_data, trading_calendar)
            
            # 第三步：策略运行
            self.logger.info("=" * 50)
            self.logger.info("第三步：策略运行")
            self.logger.info("=" * 50)
            
            # 设置策略参数
            self.strategy.set_trading_calendar(trading_calendar)
            self.strategy.set_constituents(constituents)
            
            # 运行策略生成订单
            order_book = self.strategy.run_strategy(aligned_data)
            
            # 第四步：回测执行
            self.logger.info("=" * 50)
            self.logger.info("第四步：回测执行")
            self.logger.info("=" * 50)
            
            # 执行回测
            backtest_result = self.backtest_engine.run_backtest(
                order_book, aligned_data, trading_calendar
            )
            
            # 第五步：绩效分析
            self.logger.info("=" * 50)
            self.logger.info("第五步：绩效分析")
            self.logger.info("=" * 50)
            
            # 分析绩效
            analysis_result = self.performance_analyzer.analyze(backtest_result)
            
            # 生成报告
            report = self.performance_analyzer.generate_report(analysis_result, backtest_result)
            print(report)
            
            # 保存结果到统一的输出目录
            output_dir = Config.get_output_dir()
            self.performance_analyzer.save_results(analysis_result, backtest_result, output_dir)
            
            self.logger.info("回测流程完成")
            
            return {
                'backtest_result': backtest_result,
                'analysis_result': analysis_result,
                'report': report,
                'strategy_info': self.strategy.get_strategy_info()
            }
            
        except Exception as e:
            self.logger.error(f"回测过程中发生错误: {str(e)}")
            raise
    
    def get_framework_info(self) -> Dict[str, Any]:
        """获取框架信息"""
        return {
            'framework_name': '五层架构回测框架',
            'version': '1.0.0',
            'layers': {
                'data_source': '数据源层 - 负责原始数据的加载',
                'data_processor': '数据处理层 - 负责数据清洗和因子加工',
                'strategy': '策略层 - 负责生成买卖信号和orderbook',
                'backtest_engine': '回测层 - 负责接收买卖信号进行回测',
                'performance_analyzer': '绩效层 - 负责分析回测结果输出报表'
            },
            'config': self.config
        }
    
    def update_config(self, new_config: Dict[str, Any]):
        """更新配置"""
        self.config.update(new_config)
        self.logger.info("配置已更新")
    
    def set_strategy(self, strategy_class, strategy_config: Optional[Dict] = None):
        """设置自定义策略"""
        config = {**self.config, **(strategy_config or {})}
        self.strategy = strategy_class(config)
        self.strategy.set_logger(self.logger_manager.get_logger('Strategy'))
        self.logger.info(f"策略已更新为: {strategy_class.__name__}")
    
    def set_data_source(self, data_source_class, data_source_config: Optional[Dict] = None):
        """设置自定义数据源"""
        config = {**self.config, **(data_source_config or {})}
        self.data_source = data_source_class(config)
        self.data_source.set_logger(self.logger_manager.get_logger('DataSource'))
        self.logger.info(f"数据源已更新为: {data_source_class.__name__}")
    
    def set_backtest_engine(self, engine_class, engine_config: Optional[Dict] = None):
        """设置自定义回测引擎"""
        config = {**self.config, **(engine_config or {})}
        self.backtest_engine = engine_class(config)
        self.backtest_engine.set_logger(self.logger_manager.get_logger('BacktestEngine'))
        self.logger.info(f"回测引擎已更新为: {engine_class.__name__}")
    
    def set_performance_analyzer(self, analyzer_class, analyzer_config: Optional[Dict] = None):
        """设置自定义绩效分析器"""
        config = {**self.config, **(analyzer_config or {})}
        self.performance_analyzer = analyzer_class(config)
        self.performance_analyzer.set_logger(self.logger_manager.get_logger('PerformanceAnalyzer'))
        self.logger.info(f"绩效分析器已更新为: {analyzer_class.__name__}")
    
    # 股票信息相关方法
    def get_stock_info(self, ts_code: str):
        """获取股票基本信息"""
        if self.stock_info_source:
            return self.stock_info_source.get_stock_info(ts_code)
        return None
    
    def get_stocks_by_market(self, market: str):
        """根据上市板块获取股票列表"""
        if self.stock_info_source:
            return self.stock_info_source.get_stocks_by_market(market)
        return []
    
    def get_stocks_by_industry(self, industry: str, level: int = 1):
        """根据行业分类获取股票列表"""
        if self.stock_info_source:
            return self.stock_info_source.get_stocks_by_industry(industry, level)
        return []
    
    def get_all_markets(self):
        """获取所有上市板块列表"""
        if self.stock_info_source:
            return self.stock_info_source.get_all_markets()
        return []
    
    def get_all_industries(self, level: int = 1):
        """获取所有行业分类列表"""
        if self.stock_info_source:
            return self.stock_info_source.get_all_industries(level)
        return []
    
    def search_stocks(self, keyword: str, search_in: str = 'name'):
        """搜索股票"""
        if self.stock_info_source:
            return self.stock_info_source.search_stocks(keyword, search_in)
        return []
    
    def get_stock_info_summary(self):
        """获取股票信息数据概览"""
        if self.stock_info_source:
            return self.stock_info_source.get_data_summary()
        return {'data_loaded': False}
