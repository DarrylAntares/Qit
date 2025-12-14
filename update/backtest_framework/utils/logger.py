"""
日志工具类
"""
import logging
import os
from datetime import datetime

class Logger:
    """日志管理器"""
    
    def __init__(self, log_file=None, log_level='INFO', enable_console=False, console_level='WARNING'):
        self.log_file = log_file or f'backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        self.log_level = getattr(logging, log_level.upper())
        self.console_level = getattr(logging, console_level.upper())
        self.enable_console = enable_console
        self.setup_logging()
    
    def setup_logging(self):
        """配置日志记录"""
        # 创建日志目录
        log_dir = os.path.dirname(self.log_file) if os.path.dirname(self.log_file) else 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 清除现有的日志处理器
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # 配置文件日志 - 记录所有级别的日志
        file_handler = logging.FileHandler(self.log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(self.log_level)
        
        # 设置日志格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        handlers = [file_handler]
        
        # 配置控制台日志 - 只显示重要信息
        if self.enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.console_level)  # 控制台只显示WARNING及以上级别
            
            # 控制台使用简化格式
            console_formatter = logging.Formatter('%(levelname)s: %(message)s')
            console_handler.setFormatter(console_formatter)
            handlers.append(console_handler)
        
        # 配置根日志记录器
        logging.basicConfig(
            level=self.log_level,
            handlers=handlers,
            force=True
        )
        
        # 设置第三方库的日志级别，避免过多的DEBUG信息
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('matplotlib.pyplot').setLevel(logging.WARNING)
        logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    def get_logger(self, name):
        """获取指定名称的日志记录器"""
        return logging.getLogger(name)
