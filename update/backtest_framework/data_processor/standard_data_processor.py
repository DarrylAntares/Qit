"""
标准数据处理器实现
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
from tqdm import tqdm

try:
    from .base_data_processor import BaseDataProcessor
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_processor.base_data_processor import BaseDataProcessor

class StandardDataProcessor(BaseDataProcessor):
    """标准数据处理器"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
    
    def align_data_with_calendar_first(self, stock_data: Dict[str, pd.DataFrame], 
                                      trading_calendar: List[datetime]) -> Dict[str, pd.DataFrame]:
        """
        第一步：先与交易日历对齐，确保所有股票都有完整的交易日序列
        
        参数:
            stock_data: 原始股票数据字典
            trading_calendar: 交易日历
            
        返回:
            Dict[str, pd.DataFrame]: 对齐后的股票数据
        """
        if self.logger:
            self.logger.info("第一步：数据与交易日历对齐...")
        
        aligned_data = {}
        calendar_df = pd.DataFrame({'date': trading_calendar})
        calendar_df['date'] = pd.to_datetime(calendar_df['date'])
        
        for stock_code, df in tqdm(stock_data.items(), desc="交易日对齐"):
            if df.empty:
                continue
                
            # 确保日期列为datetime类型
            df_copy = df.copy()
            if df_copy.index.name != 'trade_date':
                df_copy = df_copy.reset_index()
            
            # 检查日期列名
            if 'date' in df_copy.columns:
                date_col = 'date'
            elif 'trade_date' in df_copy.columns:
                date_col = 'trade_date'
            elif df_copy.index.name == 'trade_date':
                df_copy = df_copy.reset_index()
                date_col = 'trade_date'
            else:
                # 使用索引作为日期
                df_copy = df_copy.reset_index()
                date_col = df_copy.columns[0]  # 假设第一列是日期
            
            df_copy[date_col] = pd.to_datetime(df_copy[date_col])
            
            # 与交易日历对齐
            if date_col != 'date':
                df_copy = df_copy.rename(columns={date_col: 'date'})
            aligned_df = calendar_df.merge(df_copy, on='date', how='left')
            aligned_df = aligned_df.set_index('date')
            
        
            # 只保留有数据的股票（至少有30%的交易日有数据）
            if 'close' in aligned_df.columns:
                aligned_data[stock_code] = aligned_df
            else:
                if self.logger:
                    self.logger.warning(f"股票 {stock_code} 缺少close列")
        
        if self.logger:
            self.logger.info(f"交易日对齐完成，保留 {len(aligned_data)} 只股票")
        
        return aligned_data

    def clean_data(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        第二步：数据清洗和缺值填充
        
        参数:
            stock_data: 对齐后的股票数据字典
            
        返回:
            Dict[str, pd.DataFrame]: 清洗后的股票数据
        """
        if self.logger:
            self.logger.info("第二步：数据清洗和缺值填充...")
        
        cleaned_data = {}
        
        for stock_code, df in tqdm(stock_data.items(), desc="数据清洗"):
            if df.empty:
                continue
                
            # 复制数据避免修改原始数据
            clean_df = df.copy()
            
            # 删除全部为空的行
            clean_df = clean_df.dropna(how='all')
            
            # 价格数据前向填充（处理停牌）
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in clean_df.columns:
                    clean_df[col] = clean_df[col].ffill()
            
            # 成交量缺失填充为0（停牌期间成交量为0）
            volume_col = 'volume' if 'volume' in clean_df.columns else 'vol'
            if volume_col in clean_df.columns:
                clean_df[volume_col] = clean_df[volume_col].fillna(0)
            
            # 删除仍有缺失价格数据的行（通常是开始几天）
            clean_df = clean_df.dropna(subset=price_columns)
            
            # 异常值检查和处理
            if len(clean_df) > 0:
                # 删除异常价格数据
                volume_col = 'volume' if 'volume' in clean_df.columns else 'vol'
                clean_df = clean_df[
                    (clean_df['open'] > 0) & 
                    (clean_df['high'] > 0) & 
                    (clean_df['low'] > 0) & 
                    (clean_df['close'] > 0) &
                    (clean_df[volume_col] >= 0)
                ]
                
                # OHLC逻辑检查
                clean_df = clean_df[
                    (clean_df['high'] >= clean_df[['open', 'close']].max(axis=1)) &
                    (clean_df['low'] <= clean_df[['open', 'close']].min(axis=1))
                ]
                
                # 成交量异常值处理
                if len(clean_df) > 20:  # 至少需要20个数据点
                    volume_col = 'volume' if 'volume' in clean_df.columns else 'vol'
                    volume_mean = clean_df[volume_col].mean()
                    volume_std = clean_df[volume_col].std()
                    if volume_std > 0:
                        volume_threshold = volume_mean + 10 * volume_std
                        clean_df = clean_df[clean_df[volume_col] <= volume_threshold]
            
            # 保留有足够数据的股票
            if len(clean_df) >= 1:  # 至少1个交易日
                cleaned_data[stock_code] = clean_df
        
        if self.logger:
            self.logger.info(f"数据清洗完成，保留 {len(cleaned_data)} 只股票")
        
        return cleaned_data
    
    def calculate_factors(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        计算因子
        
        参数:
            stock_data: 股票数据
            
        返回:
            Dict[str, pd.DataFrame]: 包含因子的股票数据
        """
        if self.logger:
            self.logger.info("开始计算因子")
        
        factor_data = {}
        
        for stock_code, df in tqdm(stock_data.items(), desc="计算因子"):
            factor_df = df.copy()
            
            # 1. 计算收益率
            factor_df['return'] = factor_df['close'].pct_change()
            factor_df['return_1d'] = factor_df['return']
            factor_df['return_5d'] = factor_df['close'].pct_change(5)
            factor_df['return_20d'] = factor_df['close'].pct_change(20)
            
            # 2. 计算移动平均线
            factor_df['ma_5'] = factor_df['close'].rolling(window=5).mean()
            factor_df['ma_10'] = factor_df['close'].rolling(window=10).mean()
            factor_df['ma_20'] = factor_df['close'].rolling(window=20).mean()
            factor_df['ma_60'] = factor_df['close'].rolling(window=60).mean()
            
            # 3. 计算波动率
            factor_df['volatility_20d'] = factor_df['return'].rolling(window=20).std()
            factor_df['volatility_60d'] = factor_df['return'].rolling(window=60).std()
            
            # 4. 计算成交量相关因子
            factor_df['volume_ma_20'] = factor_df['vol'].rolling(window=20).mean()
            factor_df['volume_ratio'] = factor_df['vol'] / factor_df['volume_ma_20']
            
            # 5. 计算技术指标
            # RSI
            factor_df['rsi'] = self._calculate_rsi(factor_df['close'])
            
            # MACD
            macd_data = self._calculate_macd(factor_df['close'])
            factor_df['macd'] = macd_data['macd']
            factor_df['macd_signal'] = macd_data['signal']
            factor_df['macd_hist'] = macd_data['histogram']
            
            # 6. 计算价格位置因子
            factor_df['high_20d'] = factor_df['high'].rolling(window=20).max()
            factor_df['low_20d'] = factor_df['low'].rolling(window=20).min()
            factor_df['price_position'] = (factor_df['close'] - factor_df['low_20d']) / (factor_df['high_20d'] - factor_df['low_20d'])
            
            factor_data[stock_code] = factor_df
        
        if self.logger:
            self.logger.info(f"因子计算完成，共处理 {len(factor_data)} 只股票")
        
        return factor_data
    
    def align_data_with_calendar(self, stock_data: Dict[str, pd.DataFrame], 
                                trading_calendar: List[datetime]) -> Dict[str, pd.DataFrame]:
        """
        第三步：最终数据对齐（已经清洗过的数据）
        
        参数:
            stock_data: 已清洗的股票数据
            trading_calendar: 交易日历
            
        返回:
            Dict[str, pd.DataFrame]: 最终对齐的股票数据
        """
        if self.logger:
            self.logger.info("第三步：最终数据对齐...")
        
        aligned_data = {}
        calendar_index = pd.DatetimeIndex(trading_calendar)
        
        for stock_code, df in tqdm(stock_data.items(), desc="最终对齐"):
            # 重新索引到交易日历
            aligned_df = df.reindex(calendar_index)
            
            # 对于已清洗的数据，只需要简单的前向填充
            aligned_df = aligned_df.ffill()
            
            # 删除所有数据都为NaN的行
            aligned_df = aligned_df.dropna(how='all')
            
            if len(aligned_df) > 0:
                aligned_data[stock_code] = aligned_df
        
        if self.logger:
            self.logger.info(f"最终数据对齐完成，保留 {len(aligned_data)} 只股票")
        
        return aligned_data
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """计算MACD指标"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        
        return {
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        }
