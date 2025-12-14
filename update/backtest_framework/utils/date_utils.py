"""
日期工具类
"""
import pandas as pd
from datetime import datetime, date
from typing import List, Union

class DateUtils:
    """日期处理工具类"""
    
    @staticmethod
    def load_trading_calendar(calendar_file: str, start_date: datetime, end_date: datetime) -> List[date]:
        """
        加载交易日历数据
        
        参数:
            calendar_file: 日历文件路径
            start_date: 开始日期
            end_date: 结束日期
            
        返回:
            List[date]: 交易日历日期列表
        """
        try:
            # 读取日历文件
            calendar_df = pd.read_csv(calendar_file, dtype={'cal_date': str})
            
            # 将字符串日期转换为datetime
            calendar_df['cal_date'] = pd.to_datetime(calendar_df['cal_date'], format='%Y%m%d')
            
            # 筛选指定日期范围内的交易日
            mask = (calendar_df['cal_date'] >= start_date) & (calendar_df['cal_date'] <= end_date)
            calendar_df = calendar_df.loc[mask].copy()
            
            if calendar_df.empty:
                raise ValueError(f"在 {start_date} 到 {end_date} 范围内没有找到交易日数据")
            
            # 转换为日期列表并返回
            trading_dates = calendar_df['cal_date'].dt.date.tolist()
            return trading_dates
            
        except Exception as e:
            raise Exception(f"加载交易日历失败: {str(e)}")
    
    @staticmethod
    def is_trading_day(target_date: Union[date, datetime], trading_calendar: List[date]) -> bool:
        """
        判断是否为交易日
        
        参数:
            target_date: 目标日期
            trading_calendar: 交易日历
            
        返回:
            bool: 是否为交易日
        """
        if isinstance(target_date, datetime):
            target_date = target_date.date()
        
        return target_date in trading_calendar
    
    @staticmethod
    def get_next_trading_day(target_date: Union[date, datetime], trading_calendar: List[date]) -> date:
        """
        获取下一个交易日
        
        参数:
            target_date: 目标日期
            trading_calendar: 交易日历
            
        返回:
            date: 下一个交易日
        """
        if isinstance(target_date, datetime):
            target_date = target_date.date()
        
        for trading_date in trading_calendar:
            if trading_date > target_date:
                return trading_date
        
        raise ValueError(f"未找到 {target_date} 之后的交易日")
    
    @staticmethod
    def get_prev_trading_day(target_date: Union[date, datetime], trading_calendar: List[date]) -> date:
        """
        获取上一个交易日
        
        参数:
            target_date: 目标日期
            trading_calendar: 交易日历
            
        返回:
            date: 上一个交易日
        """
        if isinstance(target_date, datetime):
            target_date = target_date.date()
        
        for trading_date in reversed(trading_calendar):
            if trading_date < target_date:
                return trading_date
        
        raise ValueError(f"未找到 {target_date} 之前的交易日")
