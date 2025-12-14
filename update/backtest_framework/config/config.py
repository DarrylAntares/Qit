"""
回测框架配置文件
"""
from datetime import datetime
import os

class Config:
    """回测配置类"""
    
    # 数据路径配置
    DATA_PATH = "../../dataset/stock_daily_forward_feather"
    #DATA_PATH = "../../dataset/fund_daily_forward_feather"
    CALENDAR_PATH = "../../dataset/calendars/calendars.csv"
    INDEX_WEIGHT_PATH = "../../dataset/indice_weight"
    #BENCHMARK_PATH = "../../dataset/index_daily/000300_SH.feather"  # 沪深300基准数据
    BENCHMARK_PATH = "../../dataset/index_daily/881001_WI.feather"  # Wind全A基准数据
    STOCK_INFO_PATH = "../../dataset/stock_info/ashare_info.csv"  # A股基本信息数据
    
    # 回测参数配置
    START_DATE = datetime(2020, 1, 1)
    END_DATE = datetime(2025, 8, 29)
    INITIAL_CASH = 1000000
    COMMISSION = 0.001  # 0.1%佣金
    
    # 交易价格配置
    TRADE_PRICE_TYPE = 'close'  # 交易价格类型：'open'开盘价, 'close'收盘价, 'high'最高价, 'low'最低价
    POSITION_VALUE_PRICE_TYPE = 'close'  # 持仓估值价格类型：'open'开盘价, 'close'收盘价, 'high'最高价, 'low'最低价
    
    # 订单簿策略信号配置
    SIGNAL_GENERATION_MODE = 'T'  # 信号生成模式：'T'当日交易, 'T+1'次日交易
    ENABLE_LIMIT_UP_FILTER = False  # 是否启用涨停价剔除逻辑：True启用, False禁用
    
    # 策略参数配置
    REBALANCE_FREQ = 'D'  # 调仓频率：'M'月，'W'周，'D'日
    MAX_STOCKS = 5630  # 最大股票数量限制

    # 输出配置
    BASE_OUTPUT_DIR = "output"  # 基础输出目录
    SAVE_DETAILED_METRICS = True
    GENERATE_PLOTS = True
    
    # 日志配置
    LOG_LEVEL = 'DEBUG'  # 文件中记录DEBUG级别日志
    CONSOLE_LOG_LEVEL = 'WARNING'  # 控制台只显示WARNING及以上
    
    # 全局输出目录变量（在运行时设置）
    _current_output_dir = None
    
    @classmethod
    def initialize_output_dir(cls):
        """初始化输出目录（在程序开始时调用一次）"""
        if cls._current_output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cls._current_output_dir = os.path.join(cls.BASE_OUTPUT_DIR, f"backtest_{timestamp}")
            os.makedirs(cls._current_output_dir, exist_ok=True)
        return cls._current_output_dir
    
    @classmethod
    def get_output_dir(cls):
        """获取当前输出目录"""
        if cls._current_output_dir is None:
            return cls.initialize_output_dir()
        return cls._current_output_dir
    
    @classmethod
    def get_log_file(cls):
        """获取日志文件路径"""
        output_dir = cls.get_output_dir()
        return os.path.join(output_dir, "backtest.log")
