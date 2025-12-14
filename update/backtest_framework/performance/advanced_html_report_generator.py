"""
高级HTML业绩归因报告生成器
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple
import calendar

class AdvancedHTMLReportGenerator:
    """高级HTML报告生成器"""
    
    def __init__(self):
        self.chinese_colors = {
            'up': '#FF4444',      # 红色表示上涨
            'down': '#00AA00',    # 绿色表示下跌
            'neutral': '#888888', # 灰色表示中性
            'primary': '#1f77b4', # 主色调
            'secondary': '#ff7f0e' # 次要色调
        }
    
    def generate_report(self, portfolio_returns: pd.Series, 
                       benchmark_returns: pd.Series,
                       analysis_result: Dict[str, Any],
                       output_file: str):
        """生成完整的HTML业绩归因报告"""
        
        try:
            # 准备数据
            data = self._prepare_report_data(portfolio_returns, benchmark_returns, analysis_result)
            
            # 生成HTML内容
            html_content = self._generate_html_content(data)
            
            # 写入文件
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
        except Exception as e:
            print(f"HTML报告生成详细错误: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # 生成简化版报告
            self._generate_fallback_report(portfolio_returns, benchmark_returns, 
                                         analysis_result, output_file)
    
    def _prepare_report_data(self, portfolio_returns: pd.Series, 
                           benchmark_returns: pd.Series,
                           analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """准备报告所需的所有数据"""
        
        # 计算累积净值
        portfolio_cumulative = (1 + portfolio_returns).cumprod()
        benchmark_cumulative = (1 + benchmark_returns).cumprod() if benchmark_returns is not None else []
        
        # 准备日期数据
        dates = [date.strftime('%Y-%m-%d') for date in portfolio_returns.index]
        
        # 计算回撤序列
        drawdown_series = self._calculate_drawdown_series(portfolio_returns)
        
        # 计算总交易次数（从analysis_result中获取）
        total_trades = self._calculate_total_trades(analysis_result)
        
        # 计算年度统计
        yearly_stats = self._calculate_yearly_statistics(portfolio_returns, benchmark_returns, analysis_result)
        
        # 计算月度收益率矩阵
        monthly_returns = self._calculate_monthly_returns_matrix(portfolio_returns, benchmark_returns)
        
        # 计算超额收益率序列
        excess_returns_series = self._calculate_excess_returns_series(portfolio_returns, benchmark_returns)
        
        # 计算月度统计数据
        monthly_statistics = self._calculate_monthly_statistics(monthly_returns)
        
        # 计算回撤统计数据
        drawdown_statistics = self._calculate_drawdown_statistics(portfolio_returns)
        
        # 计算滚动Alpha和Beta
        rolling_alpha_beta = self._calculate_rolling_alpha_beta(portfolio_returns, benchmark_returns)
        
        # 计算交易统计（传入收益率序列用于计算回测天数）
        trade_statistics = self._calculate_trade_statistics(analysis_result, portfolio_returns)
        
        # 计算月度换手率
        monthly_turnover = self._calculate_monthly_turnover(analysis_result, portfolio_returns, benchmark_returns)
        
        # 计算行业业绩分析
        industry_performance = self._calculate_industry_performance(analysis_result, portfolio_returns, stock_info_source=None)
        
        return {
            'portfolio_cumulative': portfolio_cumulative.tolist(),
            'benchmark_cumulative': benchmark_cumulative.tolist() if len(benchmark_cumulative) > 0 else [],
            'dates': dates,
            'drawdown_series': drawdown_series,
            'drawdown_statistics': drawdown_statistics,
            'excess_returns_series': excess_returns_series,
            'yearly_stats': yearly_stats,
            'monthly_returns': monthly_returns,
            'monthly_statistics': monthly_statistics,
            'analysis_result': analysis_result,
            'total_trades': total_trades,
            'rolling_alpha_beta': rolling_alpha_beta,
            'trade_statistics': trade_statistics,
            'monthly_turnover': monthly_turnover,
            'industry_performance': industry_performance
        }
    
    def _calculate_total_trades(self, analysis_result: Dict[str, Any]) -> int:
        """计算总交易次数（买入+卖出+再平衡买入+再平衡卖出）"""
        try:
            # 从analysis_result中获取交易记录
            trades_data = analysis_result.get('trades_data', [])
            if isinstance(trades_data, list) and len(trades_data) > 0:
                # 如果trades_data是DataFrame的记录列表
                return len(trades_data)
            
            # 尝试从其他可能的字段获取交易次数
            total_trades = analysis_result.get('total_trades', 0)
            if total_trades > 0:
                return total_trades
                
            # 如果都没有，返回默认值
            return 0
        except Exception:
            return 0
    
    def _calculate_yearly_trades(self, year: int, analysis_result: Dict[str, Any]) -> int:
        """计算指定年份的交易次数"""
        try:
            # 从analysis_result中获取交易记录
            trades_data = analysis_result.get('trades_data', [])
            if isinstance(trades_data, list) and len(trades_data) > 0:
                # 统计该年份的交易次数
                yearly_trades = 0
                for trade in trades_data:
                    # 假设交易记录中有日期字段
                    trade_date = trade.get('date', '')
                    if isinstance(trade_date, str) and trade_date.startswith(str(year)):
                        yearly_trades += 1
                    elif hasattr(trade_date, 'year') and trade_date.year == year:
                        yearly_trades += 1
                return yearly_trades
            
            # 如果没有详细交易记录，返回0
            return 0
        except Exception:
            return 0
    
    def _calculate_yearly_turnover(self, year: int, analysis_result: Dict[str, Any], 
                                 portfolio_returns: pd.Series) -> float:
        """计算指定年份的换手率"""
        try:
            # 获取该年份的组合净值数据
            year_portfolio = portfolio_returns[portfolio_returns.index.year == year]
            if len(year_portfolio) == 0:
                return 0.0
            
            # 计算该年份的平均资产总额
            # 这里需要从净值序列推算资产总额，假设初始资金为analysis_result中的信息
            portfolio_series = analysis_result.get('portfolio_series', pd.Series())
            if len(portfolio_series) == 0:
                return 0.0
            
            year_portfolio_values = portfolio_series[portfolio_series.index.year == year]
            if len(year_portfolio_values) == 0:
                return 0.0
            
            avg_portfolio_value = year_portfolio_values.mean()
            
            # 获取该年份的交易记录
            trades_data = analysis_result.get('trades_data', [])
            if not isinstance(trades_data, list) or len(trades_data) == 0:
                return 0.0
            
            # 统计该年份的买入和卖出金额
            yearly_buy_amount = 0.0
            yearly_sell_amount = 0.0
            
            for trade in trades_data:
                # 检查交易日期是否在该年份
                trade_date = trade.get('date', '')
                trade_year = None
                
                if isinstance(trade_date, str) and len(trade_date) >= 4:
                    try:
                        trade_year = int(trade_date[:4])
                    except:
                        continue
                elif hasattr(trade_date, 'year'):
                    trade_year = trade_date.year
                else:
                    continue
                
                if trade_year != year:
                    continue
                
                # 获取交易金额
                amount = trade.get('amount', 0.0)
                action = trade.get('action', '').lower()
                
                if action == 'buy':
                    yearly_buy_amount += amount
                elif action == 'sell':
                    yearly_sell_amount += amount
            
            # 计算年度换手率：min(买入金额, 卖出金额) / 平均资产总额 * 100
            if avg_portfolio_value > 0:
                min_amount = min(yearly_buy_amount, yearly_sell_amount)
                yearly_turnover = (min_amount / avg_portfolio_value) * 100
                return yearly_turnover
            else:
                return 0.0
                
        except Exception as e:
            # 如果计算失败，返回0
            return 0.0
    
    def _calculate_drawdown_series(self, portfolio_returns: pd.Series) -> List[float]:
        """计算回撤序列"""
        cumulative_returns = (1 + portfolio_returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.tolist()
    
    def _calculate_drawdown_statistics(self, portfolio_returns: pd.Series) -> Dict:
        """计算回撤统计指标"""
        cumulative_returns = (1 + portfolio_returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        
        # 历史最大回撤（从第一日起）
        historical_max_dd = drawdown.min() * 100
        historical_max_dd_idx = drawdown.idxmin()
        
        # 找到历史最大回撤的起始日期（前一个峰值日期）
        historical_start_idx = None
        for i in range(len(cumulative_returns)):
            if cumulative_returns.index[i] >= historical_max_dd_idx:
                break
            if cumulative_returns.iloc[i] == peak.iloc[i]:
                historical_start_idx = cumulative_returns.index[i]
        
        historical_start_date = historical_start_idx.strftime('%Y-%m-%d') if historical_start_idx else portfolio_returns.index[0].strftime('%Y-%m-%d')
        historical_end_date = historical_max_dd_idx.strftime('%Y-%m-%d')
        
        # 检查历史最大回撤是否修复
        historical_recovered = False
        historical_recovery_date = None
        historical_recovery_days = 0
        
        if historical_start_idx:
            peak_value = cumulative_returns.loc[historical_start_idx]
            # 从最大回撤点之后查找是否恢复到前高
            after_dd = cumulative_returns[cumulative_returns.index > historical_max_dd_idx]
            recovered_idx = after_dd[after_dd >= peak_value]
            if len(recovered_idx) > 0:
                historical_recovered = True
                historical_recovery_date = recovered_idx.index[0].strftime('%Y-%m-%d')
                historical_recovery_days = (recovered_idx.index[0] - historical_max_dd_idx).days
        
        # 近1年最大回撤
        last_date = portfolio_returns.index[-1]
        one_year_ago = last_date - pd.DateOffset(years=1)
        recent_1y_data = portfolio_returns[portfolio_returns.index >= one_year_ago]
        
        recent_1y_max_dd = 0.0
        recent_1y_start_date = ''
        recent_1y_end_date = ''
        
        if len(recent_1y_data) > 0:
            recent_1y_cumulative = (1 + recent_1y_data).cumprod()
            recent_1y_peak = recent_1y_cumulative.expanding().max()
            recent_1y_drawdown = (recent_1y_cumulative - recent_1y_peak) / recent_1y_peak
            recent_1y_max_dd = recent_1y_drawdown.min() * 100
            recent_1y_max_dd_idx = recent_1y_drawdown.idxmin()
            
            # 找到近1年最大回撤的起始日期
            recent_1y_start_idx = None
            for i in range(len(recent_1y_cumulative)):
                if recent_1y_cumulative.index[i] >= recent_1y_max_dd_idx:
                    break
                if recent_1y_cumulative.iloc[i] == recent_1y_peak.iloc[i]:
                    recent_1y_start_idx = recent_1y_cumulative.index[i]
            
            recent_1y_start_date = recent_1y_start_idx.strftime('%Y-%m-%d') if recent_1y_start_idx else recent_1y_data.index[0].strftime('%Y-%m-%d')
            recent_1y_end_date = recent_1y_max_dd_idx.strftime('%Y-%m-%d')
        
        # 今年最大回撤（最后一年）
        last_year = portfolio_returns.index[-1].year
        year_data = portfolio_returns[portfolio_returns.index.year == last_year]
        if len(year_data) > 0:
            year_cumulative = (1 + year_data).cumprod()
            year_peak = year_cumulative.expanding().max()
            year_drawdown = (year_cumulative - year_peak) / year_peak
            ytd_max_dd = year_drawdown.min() * 100
        else:
            ytd_max_dd = 0.0
        
        # 近3个月最大回撤
        three_months_ago = last_date - pd.DateOffset(months=3)
        recent_3m_data = portfolio_returns[portfolio_returns.index >= three_months_ago]
        if len(recent_3m_data) > 0:
            recent_3m_cumulative = (1 + recent_3m_data).cumprod()
            recent_3m_peak = recent_3m_cumulative.expanding().max()
            recent_3m_drawdown = (recent_3m_cumulative - recent_3m_peak) / recent_3m_peak
            recent_3m_max_dd = recent_3m_drawdown.min() * 100
        else:
            recent_3m_max_dd = 0.0
        
        return {
            'historical_max_dd': historical_max_dd,
            'historical_start_date': historical_start_date,
            'historical_end_date': historical_end_date,
            'historical_recovered': historical_recovered,
            'historical_recovery_date': historical_recovery_date if historical_recovery_date else '',
            'historical_recovery_days': historical_recovery_days,
            'ytd_max_dd': ytd_max_dd,
            'recent_3m_max_dd': recent_3m_max_dd,
            'recent_1y_max_dd': recent_1y_max_dd,
            'recent_1y_start_date': recent_1y_start_date,
            'recent_1y_end_date': recent_1y_end_date
        }
    
    def _calculate_trade_statistics(self, analysis_result: Dict[str, Any], portfolio_returns: pd.Series = None) -> Dict:
        """计算交易统计指标
        
        参数:
            analysis_result: 分析结果字典，包含trades_data
            portfolio_returns: 组合收益率序列，用于计算回测天数
            
        返回:
            包含交易统计指标的字典
        """
        try:
            # 获取交易记录
            trades_data = analysis_result.get('trades_data', [])
            if not trades_data or len(trades_data) == 0:
                return {
                    'has_data': False,
                    'turnover_rate': 0,
                    'win_rate': 0,
                    'profit_loss_ratio': 0,
                    'avg_holding_days': 0,
                    'avg_profit_holding_days': 0,
                    'avg_loss_holding_days': 0,
                    'total_sell_trades': 0,
                    'win_trades': 0,
                    'loss_trades': 0
                }
            
            # 转换为DataFrame
            import pandas as pd
            trades_df = pd.DataFrame(trades_data)
            
            # 筛选卖出交易（sell和rebalance_sell）
            sell_trades = trades_df[trades_df['action'].isin(['sell', 'rebalance_sell'])].copy()
            
            if len(sell_trades) == 0:
                return {
                    'has_data': False,
                    'turnover_rate': 0,
                    'win_rate': 0,
                    'profit_loss_ratio': 0,
                    'avg_holding_days': 0,
                    'avg_profit_holding_days': 0,
                    'avg_loss_holding_days': 0,
                    'total_sell_trades': 0,
                    'win_trades': 0,
                    'loss_trades': 0
                }
            
            # 统计指标
            total_sell_trades = len(sell_trades)
            
            # 0. 换手率：双边交易金额 / 组合市值 / 2，并年化
            # 计算所有交易的总金额（买入+卖出）
            if 'amount' in trades_df.columns and 'position_ratio' in trades_df.columns:
                # 通过position_ratio反推组合市值，取平均值
                trades_with_ratio = trades_df[trades_df['position_ratio'] > 0].copy()
                if len(trades_with_ratio) > 0:
                    # portfolio_value = amount / (position_ratio / 100)
                    trades_with_ratio['portfolio_value'] = trades_with_ratio['amount'] / (trades_with_ratio['position_ratio'] / 100)
                    avg_portfolio_value = trades_with_ratio['portfolio_value'].mean()
                    
                    # 计算双边交易总金额
                    total_trade_amount = trades_df['amount'].sum()
                    
                    # 换手率 = 双边交易金额 / 组合市值 / 2
                    turnover_rate = (total_trade_amount / avg_portfolio_value / 2 * 100) if avg_portfolio_value > 0 else 0
                    
                    # 年化换手率
                    if portfolio_returns is not None and len(portfolio_returns) > 0:
                        # 计算回测天数
                        backtest_days = len(portfolio_returns)
                        # 年化系数（假设一年252个交易日）
                        annual_factor = 252 / backtest_days
                        turnover_rate = turnover_rate * annual_factor
                else:
                    turnover_rate = 0
            else:
                turnover_rate = 0
            
            # 1. 胜率：收益率>0的占比
            win_trades = sell_trades[sell_trades['return_rate'] > 0]
            loss_trades = sell_trades[sell_trades['return_rate'] < 0]
            win_count = len(win_trades)
            loss_count = len(loss_trades)
            win_rate = (win_count / total_sell_trades * 100) if total_sell_trades > 0 else 0
            
            # 2. 盈亏比：盈利平均收益率 / 亏损平均收益率的绝对值
            avg_profit_return = win_trades['return_rate'].mean() if len(win_trades) > 0 else 0
            avg_loss_return = loss_trades['return_rate'].mean() if len(loss_trades) > 0 else 0
            profit_loss_ratio = (avg_profit_return / abs(avg_loss_return)) if avg_loss_return != 0 else 0
            
            # 3. 平均持有天数
            avg_holding_days = sell_trades['holding_days'].mean() if 'holding_days' in sell_trades.columns else 0
            
            # 4. 盈利持有天数
            avg_profit_holding_days = win_trades['holding_days'].mean() if len(win_trades) > 0 and 'holding_days' in win_trades.columns else 0
            
            # 5. 亏损持有天数
            avg_loss_holding_days = loss_trades['holding_days'].mean() if len(loss_trades) > 0 and 'holding_days' in loss_trades.columns else 0
            
            return {
                'has_data': True,
                'turnover_rate': float(turnover_rate),
                'win_rate': float(win_rate),
                'profit_loss_ratio': float(profit_loss_ratio),
                'avg_holding_days': float(avg_holding_days),
                'avg_profit_holding_days': float(avg_profit_holding_days),
                'avg_loss_holding_days': float(avg_loss_holding_days),
                'total_sell_trades': int(total_sell_trades),
                'win_trades': int(win_count),
                'loss_trades': int(loss_count),
                'avg_profit_return': float(avg_profit_return),
                'avg_loss_return': float(avg_loss_return)
            }
            
        except Exception as e:
            print(f"计算交易统计时出错: {str(e)}")
            return {
                'has_data': False,
                'turnover_rate': 0,
                'win_rate': 0,
                'profit_loss_ratio': 0,
                'avg_holding_days': 0,
                'avg_profit_holding_days': 0,
                'avg_loss_holding_days': 0,
                'total_sell_trades': 0,
                'win_trades': 0,
                'loss_trades': 0
            }
    
    def _calculate_monthly_turnover(self, analysis_result: Dict[str, Any], 
                                   portfolio_returns: pd.Series,
                                   benchmark_returns: pd.Series = None) -> Dict:
        """计算月度日均换手率
        
        参数:
            analysis_result: 分析结果字典，包含trades_data
            portfolio_returns: 组合收益率序列
            benchmark_returns: 基准收益率序列
            
        返回:
            包含月度换手率数据的字典
        """
        try:
            # 获取交易记录
            trades_data = analysis_result.get('trades_data', [])
            if not trades_data or len(trades_data) == 0:
                return {'has_data': False}
            
            import pandas as pd
            trades_df = pd.DataFrame(trades_data)
            
            # 检查必要字段
            if 'amount' not in trades_df.columns or 'position_ratio' not in trades_df.columns or 'date' not in trades_df.columns:
                return {'has_data': False}
            
            # 转换日期列
            trades_df['date'] = pd.to_datetime(trades_df['date'])
            
            # 通过position_ratio反推组合市值
            trades_with_ratio = trades_df[trades_df['position_ratio'] > 0].copy()
            if len(trades_with_ratio) == 0:
                return {'has_data': False}
            
            trades_with_ratio['portfolio_value'] = trades_with_ratio['amount'] / (trades_with_ratio['position_ratio'] / 100)
            
            # 按月分组计算
            trades_with_ratio['year_month'] = trades_with_ratio['date'].dt.to_period('M')
            
            monthly_data = []
            for year_month, group in trades_with_ratio.groupby('year_month'):
                # 计算该月的交易金额和平均组合市值
                total_amount = group['amount'].sum()
                avg_portfolio_value = group['portfolio_value'].mean()
                
                # 计算该月的交易天数
                trading_days = group['date'].nunique()
                
                # 月度换手率 = (双边交易金额 / 平均组合市值 / 2) / 交易天数
                if avg_portfolio_value > 0 and trading_days > 0:
                    daily_turnover = (total_amount / avg_portfolio_value / 2 / trading_days) * 100
                else:
                    daily_turnover = 0
                
                monthly_data.append({
                    'year_month': str(year_month),
                    'date': year_month.to_timestamp(),
                    'daily_turnover': daily_turnover
                })
            
            if len(monthly_data) == 0:
                return {'has_data': False}
            
            monthly_df = pd.DataFrame(monthly_data)
            
            # 准备基准净值数据（月度）
            benchmark_monthly = []
            if benchmark_returns is not None and len(benchmark_returns) > 0:
                # 计算基准的累积净值
                benchmark_cumulative = (1 + benchmark_returns).cumprod()
                # 按月采样
                benchmark_monthly_series = benchmark_cumulative.resample('M').last()
                
                for date, value in benchmark_monthly_series.items():
                    benchmark_monthly.append({
                        'date': date,
                        'value': value
                    })
            
            return {
                'has_data': True,
                'dates': [item['date'].strftime('%Y-%m') for item in monthly_data],
                'daily_turnover': [item['daily_turnover'] for item in monthly_data],
                'benchmark_dates': [item['date'].strftime('%Y-%m') for item in benchmark_monthly],
                'benchmark_values': [item['value'] for item in benchmark_monthly]
            }
            
        except Exception as e:
            print(f"计算月度换手率时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'has_data': False}
    
    def _calculate_rolling_alpha_beta(self, portfolio_returns: pd.Series, 
                                      benchmark_returns: pd.Series,
                                      window: int = 250) -> Dict:
        """计算滚动窗口的Alpha和Beta时序数据
        
        参数:
            portfolio_returns: 组合收益率序列
            benchmark_returns: 基准收益率序列
            window: 滚动窗口大小，默认250个交易日
            
        返回:
            包含alpha和beta时序数据的字典，如果数据不足则返回None
        """
        # 如果基准数据为空或数据不足，返回None
        if benchmark_returns is None or len(benchmark_returns) == 0:
            return None
        
        # 对齐数据
        aligned_portfolio, aligned_benchmark = portfolio_returns.align(benchmark_returns, join='inner')
        
        # 如果对齐后的数据不足窗口期，返回None
        if len(aligned_portfolio) < window:
            return None
        
        # 存储结果
        dates = []
        alphas = []
        betas = []
        
        # 滚动窗口计算
        for i in range(window, len(aligned_portfolio) + 1):
            # 获取窗口数据
            window_portfolio = aligned_portfolio.iloc[i-window:i]
            window_benchmark = aligned_benchmark.iloc[i-window:i]
            
            # 计算Beta（使用协方差和方差）
            if window_benchmark.var() != 0:
                covariance = np.cov(window_portfolio, window_benchmark)[0, 1]
                beta = covariance / window_benchmark.var()
            else:
                beta = 0
            
            # 计算Alpha（CAPM模型）
            # Alpha = 组合年化收益率 - (无风险利率 + Beta × (基准年化收益率 - 无风险利率))
            risk_free_rate = 0  # 假设无风险利率为3%
            portfolio_annual_return = (1 + window_portfolio).prod() ** (252 / len(window_portfolio)) - 1
            benchmark_annual_return = (1 + window_benchmark).prod() ** (252 / len(window_benchmark)) - 1
            alpha = portfolio_annual_return - (risk_free_rate + beta * (benchmark_annual_return - risk_free_rate))
            
            # 记录结果
            dates.append(aligned_portfolio.index[i-1].strftime('%Y-%m-%d'))
            alphas.append(alpha * 100)  # 转换为百分比
            betas.append(beta)
        
        return {
            'dates': dates,
            'alphas': alphas,
            'betas': betas,
            'has_data': True
        }
    
    def _calculate_yearly_statistics(self, portfolio_returns: pd.Series, 
                                   benchmark_returns: pd.Series = None,
                                   analysis_result: Dict[str, Any] = None) -> List[Dict]:
        """计算逐年统计数据"""
        yearly_stats = []
        
        for year in portfolio_returns.index.year.unique():
            year_data = portfolio_returns[portfolio_returns.index.year == year]
            
            if len(year_data) == 0:
                continue
                
            # 计算年度指标
            annual_return = (1 + year_data).prod() - 1
            volatility = year_data.std() * np.sqrt(252)
            
            # 最大回撤
            cumulative = (1 + year_data).cumprod()
            peak = cumulative.expanding().max()
            drawdown = ((cumulative - peak) / peak).min()
            
            # 胜率和盈亏比：基于sell交易记录计算
            win_rate = 0
            profit_loss_ratio = 0
            
            # 从analysis_result中获取交易记录
            trades_data = analysis_result.get('trades_data', [])
            trade_df = pd.DataFrame(trades_data)
            if not trade_df.empty:
                # 筛选该年度的sell交易
                trade_df['trade_date'] = pd.to_datetime(trade_df['date'])
                year_sell_trades = trade_df[
                        (trade_df['trade_date'].dt.year == year) & 
                        (trade_df['action'] == 'sell')
                    ]
                    
                if len(year_sell_trades) > 0:
                    # 计算胜率：收益率>0的sell交易占比
                    win_trades = year_sell_trades[year_sell_trades['return_rate'] > 0]
                    loss_trades = year_sell_trades[year_sell_trades['return_rate'] < 0]
                    win_rate = len(win_trades) / len(year_sell_trades)
                        
                    # 计算盈亏比：盈利平均收益率 / 亏损平均收益率的绝对值
                    if len(win_trades) > 0 and len(loss_trades) > 0:
                        avg_profit_return = win_trades['return_rate'].mean()
                        avg_loss_return = loss_trades['return_rate'].mean()
                        profit_loss_ratio = avg_profit_return / abs(avg_loss_return) if avg_loss_return != 0 else 0
            
            # 计算该年度的交易次数（从交易记录中统计）
            trading_count = self._calculate_yearly_trades(year, analysis_result)
            
            # 计算该年度的换手率
            yearly_turnover = self._calculate_yearly_turnover(year, analysis_result, portfolio_returns)
            
            stats = {
                'year': int(year),
                'annual_return': annual_return * 100,
                'volatility': volatility * 100,
                'max_drawdown': drawdown * 100,
                'win_rate': win_rate * 100,
                'profit_loss_ratio': profit_loss_ratio,
                'trading_count': trading_count,
                'trading_days': len(year_data),
                'turnover_rate': yearly_turnover
            }
            
            # 如果有基准数据，计算相对指标
            if benchmark_returns is not None:
                bench_year = benchmark_returns[benchmark_returns.index.year == year]
                if len(bench_year) > 0:
                    # 对齐数据
                    aligned_portfolio, aligned_benchmark = year_data.align(bench_year, join='inner')
                    if len(aligned_portfolio) > 0:
                        bench_return = (1 + aligned_benchmark).prod() - 1
                        excess_return = annual_return - bench_return
                        
                        # 夏普比率 = 超额收益率 / 年化波动率
                        sharpe = excess_return / volatility if volatility > 0 else 0
                        
                        stats['benchmark_return'] = bench_return * 100
                        stats['excess_return'] = excess_return * 100
                        stats['sharpe_ratio'] = sharpe
                    else:
                        # 如果没有基准数据，使用传统夏普比率计算
                        stats['sharpe_ratio'] = (annual_return - 0.03) / volatility if volatility > 0 else 0
                else:
                    stats['sharpe_ratio'] = (annual_return - 0.03) / volatility if volatility > 0 else 0
            else:
                # 没有基准时，使用传统夏普比率计算
                stats['sharpe_ratio'] = (annual_return - 0.03) / volatility if volatility > 0 else 0
            
            yearly_stats.append(stats)
        
        return yearly_stats
    
    def _calculate_monthly_returns_matrix(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series = None) -> Dict:
        """计算月度收益率矩阵用于热力图（包含组合收益率和超额收益率）"""
        monthly_data = {}
        
        for year in portfolio_returns.index.year.unique():
            year_portfolio = portfolio_returns[portfolio_returns.index.year == year]
            monthly_data[str(year)] = {
                'portfolio': {},  # 组合收益率
                'excess': {}      # 超额收益率
            }
            
            # 获取对应年份的基准数据
            year_benchmark = None
            if benchmark_returns is not None:
                year_benchmark = benchmark_returns[benchmark_returns.index.year == year]
            
            for month in range(1, 13):
                month_portfolio = year_portfolio[year_portfolio.index.month == month]
                
                if len(month_portfolio) > 0:
                    # 计算组合月度收益率
                    monthly_portfolio_return = (1 + month_portfolio).prod() - 1
                    monthly_data[str(year)]['portfolio'][str(month)] = monthly_portfolio_return * 100
                    
                    # 计算超额收益率
                    if year_benchmark is not None:
                        month_benchmark = year_benchmark[year_benchmark.index.month == month]
                        if len(month_benchmark) > 0:
                            # 对齐数据
                            aligned_portfolio, aligned_benchmark = month_portfolio.align(month_benchmark, join='inner')
                            if len(aligned_portfolio) > 0:
                                monthly_benchmark_return = (1 + aligned_benchmark).prod() - 1
                                excess_return = monthly_portfolio_return - monthly_benchmark_return
                                monthly_data[str(year)]['excess'][str(month)] = excess_return * 100
                            else:
                                monthly_data[str(year)]['excess'][str(month)] = None
                        else:
                            monthly_data[str(year)]['excess'][str(month)] = None
                    else:
                        monthly_data[str(year)]['excess'][str(month)] = None
                else:
                    monthly_data[str(year)]['portfolio'][str(month)] = None
                    monthly_data[str(year)]['excess'][str(month)] = None
        
        return monthly_data
    
    
    def _calculate_monthly_statistics(self, monthly_data: Dict) -> Dict:
        """计算月度统计数据用于分析文本"""
        total_months = 0
        positive_months = 0
        outperform_months = 0
        best_month_return = float('-inf')
        best_month_date = None
        
        for year, data in monthly_data.items():
            for month, portfolio_return in data['portfolio'].items():
                if portfolio_return is not None:
                    total_months += 1
                    # 统计正收益月份
                    if portfolio_return > 0:
                        positive_months += 1
                    # 统计跑赢基准月份
                    excess_return = data['excess'].get(month)
                    if excess_return is not None and excess_return > 0:
                        outperform_months += 1
                    # 记录最佳月份
                    if portfolio_return > best_month_return:
                        best_month_return = portfolio_return
                        best_month_date = f"{year}年{month}月"
        
        return {
            'total_months': total_months,
            'positive_months': positive_months,
            'outperform_months': outperform_months,
            'monthly_win_rate': (positive_months / total_months * 100) if total_months > 0 else 0,
            'relative_win_rate': (outperform_months / total_months * 100) if total_months > 0 else 0,
            'best_month_date': best_month_date,
            'best_month_return': best_month_return
        }
    
    def _calculate_excess_returns_series(self, portfolio_returns: pd.Series, 
                                       benchmark_returns: pd.Series = None) -> List[float]:
        """计算超额收益率序列"""
        if benchmark_returns is None or len(benchmark_returns) == 0:
            return [0.0] * len(portfolio_returns)
        
        try:
            # 对齐数据
            aligned_portfolio, aligned_benchmark = portfolio_returns.align(benchmark_returns, join='inner')
            
            # 分别计算累积净值
            portfolio_cumulative = (1 + aligned_portfolio).cumprod()
            benchmark_cumulative = (1 + aligned_benchmark).cumprod()
            
            # 直接相减得到超额收益率序列
            excess_returns_series = portfolio_cumulative - benchmark_cumulative
            
            return excess_returns_series.tolist()
        except Exception:
            return [0.0] * len(portfolio_returns)
    
    def _calculate_industry_performance(self, analysis_result: Dict[str, Any], portfolio_returns: pd.Series, stock_info_source=None) -> Dict[str, Any]:
        """计算行业业绩分析"""
        try:
            # 获取交易记录
            trades_data = analysis_result.get('trades_data', [])
            if not trades_data:
                return {
                    'industry_performance': {},
                    'top_10_industries': [],
                    'bottom_10_industries': [],
                    'total_industries': 0,
                    'total_sell_trades': 0
                }
            
            # 转换为DataFrame便于处理
            import pandas as pd
            import numpy as np
            trades_df = pd.DataFrame(trades_data)
            
            # 1. 筛选卖出和再平衡卖出交易
            target_trades = trades_df.copy()
            if target_trades.empty:
                return {
                    'industry_performance': {},
                    'top_10_industries': [],
                    'bottom_10_industries': [],
                    'total_industries': 0,
                    'total_target_trades': 0
                }
            
            # 获取最新组合市值（从analysis_result中的portfolio_series获取）
            portfolio_series = analysis_result.get('portfolio_series', None)
            if portfolio_series is not None and len(portfolio_series) > 0:
                latest_portfolio_value = portfolio_series.iloc[-1]
            else:
                # 如果没有portfolio_series，使用end_value作为备选
                latest_portfolio_value = analysis_result.get('end_value', 1.0)
            print(f"最新组合市值: {latest_portfolio_value}")
            
            # 2. 计算每笔交易的净损益：净损益 = net_amount - avg_cost * quantity
            target_trades['net_pnl'] = target_trades['net_amount'] - (target_trades['avg_cost'] * target_trades['quantity'])
            print(f"计算净损益完成，交易数量: {len(target_trades)}")
            
            # 获取或创建stock_info数据源
            if stock_info_source is None:
                try:
                    # 尝试相对导入
                    from ..data_source.stock_info_data_source import StockInfoDataSource
                    stock_info_source = StockInfoDataSource(config=None)
                except (ImportError, ValueError):
                    try:
                        # 如果相对导入失败，尝试绝对导入
                        import sys
                        import os
                        
                        # 添加项目根目录到sys.path
                        current_dir = os.path.dirname(os.path.abspath(__file__))
                        project_root = os.path.dirname(current_dir)
                        if project_root not in sys.path:
                            sys.path.insert(0, project_root)
                        
                        from data_source.stock_info_data_source import StockInfoDataSource
                        stock_info_source = StockInfoDataSource(config=None)
                    except Exception as e:
                        print(f"无法加载股票信息数据源: {str(e)}")
                        print(f"错误详情: {type(e).__name__}: {e}")
                        import traceback
                        traceback.print_exc()
                        return {
                            'industry_performance': {},
                            'top_10_industries': [],
                            'bottom_10_industries': [],
                            'total_industries': 0,
                            'total_sell_trades': 0
                        }
            
            # 3. 为每笔交易添加行业信息
            target_trades['sw_industry_1'] = None
            for idx, trade in target_trades.iterrows():
                stock_code = trade['stock_code']
                stock_info = stock_info_source.get_stock_info(stock_code)
                if stock_info:
                    target_trades.at[idx, 'sw_industry_1'] = stock_info.get('sw_industry_1', '其它')
                else:
                    target_trades.at[idx, 'sw_industry_1'] = '其它'
            
            # 3.1. 按照买入和卖出筛选
            buy_trades = target_trades[target_trades['action'].isin(['buy', 'rebalance_buy'])].copy()
            target_trades = target_trades[target_trades['action'].isin(['sell', 'rebalance_sell'])].copy()
            
            
            
            # 4. 计算总交易额（用于计算成交占比）
            total_trade_amount = target_trades['amount'].sum() if 'amount' in target_trades.columns else 0
            print(f"总卖出交易额: {total_trade_amount}")
            
            # 5. 按行业统计累计净损益、损益边际贡献和成交占比
            industry_performance = {}
            
            for industry in target_trades['sw_industry_1'].unique():
                if pd.isna(industry) or industry == '其它':
                    continue
                    
                industry_trades = target_trades[target_trades['sw_industry_1'] == industry]
                
                # 计算该行业的累计净损益
                cumulative_net_pnl = industry_trades['net_pnl'].sum()
                
                # 计算损益边际贡献 = 累计净损益 / 最新组合市值
                marginal_contribution = cumulative_net_pnl / latest_portfolio_value if latest_portfolio_value != 0 else 0
                
                # 计算该行业的交易额
                industry_amount = industry_trades['amount'].sum() if 'amount' in industry_trades.columns else 0
                
                # 计算成交占比 = 该行业交易额 / 总交易额
                amount_ratio = industry_amount / total_trade_amount * 100 if total_trade_amount != 0 else 0
                
                # 计算其他统计指标
                returns = industry_trades['return_rate'].dropna() if 'return_rate' in industry_trades.columns else pd.Series([])
                
                industry_performance[industry] = {
                    'cumulative_net_pnl': cumulative_net_pnl,
                    'marginal_contribution': marginal_contribution,
                    'marginal_contribution_pct': marginal_contribution * 100,  # 百分比形式
                    'amount_ratio_pct': amount_ratio,  # 成交占比（百分比）
                    'trade_count': len(industry_trades),
                    'total_amount': industry_amount,
                    'avg_return': returns.mean() if len(returns) > 0 else 0,
                    'std_return': returns.std() if len(returns) > 1 else 0
                }
                
            print(f"行业统计完成，涉及行业数: {len(industry_performance)}")
            

            # 5. 按损益边际贡献排序
            sorted_industries = sorted(industry_performance.items(), 
                                     key=lambda x: x[1]['marginal_contribution'], 
                                     reverse=True)
            
            # 获取前十和后十
            top_10 = sorted_industries[:10]
            bottom_10 = sorted_industries[-10:] if len(sorted_industries) > 10 else []
            
            print(f"排序完成，前十行业: {[item[0] for item in top_10[:3]]}")
            print(f"后十行业: {[item[0] for item in bottom_10[-3:]] if bottom_10 else '无'}")

            # 6. 计算季度行业成交额统计
            quarterly_industry_stats = self._calculate_quarterly_industry_stats(buy_trades)
            
            return {
                'industry_performance': industry_performance,
                'top_10_industries': top_10,
                'bottom_10_industries': bottom_10,
                'total_industries': len(industry_performance),
                'total_target_trades': len(target_trades),
                'latest_portfolio_value': latest_portfolio_value,
                'quarterly_industry_stats': quarterly_industry_stats
            }
            
        except Exception as e:
            print(f"行业业绩分析失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'industry_performance': {},
                'top_10_industries': [],
                'bottom_10_industries': [],
                'total_industries': 0,
                'total_sell_trades': 0,
                'quarterly_industry_stats': {'quarters': [], 'industry_stats': {}}
            }
    
    def _calculate_quarterly_industry_stats(self, target_trades) -> Dict:
        """计算季度行业成交额统计"""
        try:
            import pandas as pd
            from datetime import datetime
            
            if target_trades.empty:
                return {'quarters': [], 'industry_stats': {}}
            
            # 为交易数据添加季度信息
            target_trades = target_trades.copy()
            
            # 解析交易日期并添加季度信息
            if 'date' in target_trades.columns:
                target_trades['trade_date'] = pd.to_datetime(target_trades['date'])
            elif 'timestamp' in target_trades.columns:
                target_trades['trade_date'] = pd.to_datetime(target_trades['timestamp'])
            else:
                # 如果没有日期字段，返回空结果
                print("警告：交易数据中没有找到日期字段，无法计算季度统计")
                return {'quarters': [], 'industry_stats': {}}
            
            # 添加季度标识
            target_trades['year'] = target_trades['trade_date'].dt.year
            target_trades['quarter'] = target_trades['trade_date'].dt.quarter
            target_trades['quarter_str'] = target_trades['year'].astype(str).str[-2:] + 'Q' + target_trades['quarter'].astype(str)
            
            # 按季度和行业统计成交额
            quarterly_stats = {}
            quarters = sorted(target_trades['quarter_str'].unique())
            
            for quarter in quarters:
                quarter_data = target_trades[target_trades['quarter_str'] == quarter]
                
                # 按行业统计该季度的成交额
                industry_amounts = quarter_data.groupby('sw_industry_1')['amount'].sum().sort_values(ascending=False)
                total_quarter_amount = industry_amounts.sum()
                
                # 计算成交额占比并取前十
                top_10_industries = {}
                for i, (industry, amount) in enumerate(industry_amounts.head(10).items()):
                    if pd.notna(industry) and industry != '其它':
                        ratio = (amount / total_quarter_amount * 100) if total_quarter_amount > 0 else 0
                        top_10_industries[f'industry_{i+1}'] = {
                            'name': industry,
                            'ratio': ratio,
                            'amount': amount
                        }
                
                quarterly_stats[quarter] = {
                    'total_amount': total_quarter_amount,
                    'top_industries': top_10_industries
                }
            
            print(f"季度统计完成，涉及季度: {quarters}")
            
            return {
                'quarters': quarters,
                'industry_stats': quarterly_stats
            }
            
        except Exception as e:
            print(f"季度行业统计计算失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'quarters': [], 'industry_stats': {}}
    
    def _generate_html_content(self, data: Dict[str, Any]) -> str:
        """生成完整的HTML内容"""
        
        # 准备JavaScript数据，确保所有数据都是JSON可序列化的
        def convert_to_json_serializable(obj):
            """将numpy类型转换为Python原生类型"""
            import numpy as np
            
            if isinstance(obj, np.ndarray):
                if obj.size == 1:
                    return obj.item()
                else:
                    return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif hasattr(obj, 'item') and hasattr(obj, 'size'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            else:
                return obj
        
        js_data = {
            'portfolioData': data['portfolio_cumulative'],
            'benchmarkData': data['benchmark_cumulative'],
            'dates': data['dates'],
            'drawdownData': data['drawdown_series'],
            'excessReturnsData': data['excess_returns_series'],
            'yearlyStats': convert_to_json_serializable(data['yearly_stats']),
            'monthlyReturns': convert_to_json_serializable(data['monthly_returns']),
            'rollingAlphaBeta': convert_to_json_serializable(data['rolling_alpha_beta']) if data['rolling_alpha_beta'] else None,
            'monthlyTurnover': convert_to_json_serializable(data['monthly_turnover']) if data['monthly_turnover'] else None,
            'industryPerformance': convert_to_json_serializable(data['industry_performance'])
        }
        
        analysis = data['analysis_result']
        
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>投资组合业绩归因分析报告</title>
    <!-- Bootstrap CSS -->
    <link href="https://cdn.bootcdn.net/ajax/libs/bootstrap/5.3.0/css/bootstrap.min.css" rel="stylesheet" 
          onerror="this.onerror=null;this.href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css';">
    <!-- Font Awesome -->
    <link href="https://cdn.bootcdn.net/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet"
          onerror="this.onerror=null;this.href='https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css';">
    <!-- Plotly.js -->
    <script>
        window.plotlyLoaded = false;
    </script>
    <script src="https://fastly.jsdelivr.net/npm/plotly.js-dist@2.26.0/plotly.min.js" 
            onload="window.plotlyLoaded = true;" 
            onerror="console.log('Plotly CDN failed, using fallback')"></script>
    <style>
        .metric-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }}
        .section-header {{ border-left: 4px solid #007bff; padding-left: 15px; margin: 30px 0 20px 0; }}
        .chart-container {{ background: #f8f9fa; border-radius: 10px; padding: 20px; margin: 20px 0; }}
        .monthly-table {{ font-size: 12px; }}
        .monthly-table th {{ background-color: #343a40; color: white; text-align: center; font-weight: bold; }}
        .monthly-table td {{ text-align: center; font-weight: bold; }}
        .positive-return {{ background-color: #fc9d9a !important; color: #c62828; }}
        .negative-return {{ background-color: #acd59f !important; color: #2e7d32; }}
        .no-data {{ background-color: #f5f5f5 !important; color: #757575; }}
        .year-label {{ background-color: #f5f5f5 !important; color: #1565c0; font-weight: bold; }}
        .stats-table {{ font-size: 11px; }}
        .stats-table th {{ background-color: #343a40; color: white; text-align: center; font-weight: bold; font-size: 10px; }}
        .stats-table td {{ text-align: center; }}
        .positive {{ color: #fc9d9a; font-weight: bold; }}
        .negative {{ color: #acd59f; font-weight: bold; }}
        
        .heatmap-container {{
            display: grid;
            grid-template-columns: repeat(13, 1fr);
            gap: 1px;
            margin: 20px 0;
        }}
        
        .heatmap-cell {{
            height: 35px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75em;
            font-weight: bold;
            border-radius: 3px;
            color: white;
            min-width: 0;
        }}
        
        .month-header {{
            background: #34495e;
            color: white;
            font-weight: bold;
            height: 40px;
            font-size: 0.8em;
        }}
        
        .year-header {{
            background: #2c3e50;
            color: white;
            font-weight: bold;
            font-size: 0.7em;
        }}
        
        .excess-header {{
            background: #6c757d;
            color: white;
            font-weight: bold;
            font-size: 0.7em;
        }}
        
        /* 季度行业统计表格样式 */
        .quarterly-container {{
            display: grid;
            grid-template-columns: 80px repeat(10, 1fr);
            gap: 1px;
            margin: 20px 0;
        }}
        
        .quarterly-cell {{
            height: 30px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.7em;
            font-weight: bold;
            border-radius: 3px;
            color: white;
            min-width: 0;
            background: #f8f9fa;
            color: #495057;
            border: 1px solid #dee2e6;
        }}
        
        .quarterly-header {{
            background: #34495e;
            color: white;
            font-weight: bold;
            height: 35px;
            font-size: 0.75em;
        }}
        
        .quarterly-quarter {{
            background: #2c3e50;
            color: white;
            font-weight: bold;
            font-size: 0.7em;
            width: 80px;
        }}
        
        .quarterly-industry {{
            background: #f8f9fa;
            color: #495057;
            font-size: 0.7em;
            line-height: 1.2;
            padding: 2px;
            text-align: center;
            border: 1px solid #dee2e6;
            font-weight: bold;
        }}
        
        .quarterly-empty {{
            background: #f8f9fa;
            color: #6c757d;
            font-size: 0.7em;
        }}
        
        /* 月度分析文本框样式 */
        .monthly-analysis-box {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-left: 4px solid #667eea;
            border-radius: 8px;
            padding: 20px 25px;
            margin: 20px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        
        .monthly-analysis-box p {{
            margin: 0;
            line-height: 1.8;
            font-size: 15px;
            color: #2c3e50;
            text-align: justify;
        }}
        
        .monthly-analysis-box .highlight {{
            color: #667eea;
            font-weight: bold;
        }}
        
        .monthly-analysis-box .positive {{
            color: #e74c3c;
            font-weight: bold;
        }}
        
        .monthly-analysis-box .negative {{
            color: #27ae60;
            font-weight: bold;
        }}
        
        /* 回撤指标展示样式 */
        .drawdown-metrics-row {{
            display: flex;
            justify-content: space-around;
            margin-bottom: 20px;
            gap: 15px;
        }}
        
        .drawdown-metric-item {{
            flex: 1;
            text-align: center;
            padding: 15px;
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
            border-radius: 8px;
            color: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .drawdown-metric-label {{
            font-size: 14px;
            font-weight: 500;
            margin-bottom: 8px;
            opacity: 0.9;
        }}
        
        .drawdown-metric-value {{
            font-size: 24px;
            font-weight: bold;
        }}
        
        /* 回撤分析文本框样式 */
        .drawdown-analysis-box {{
            background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
            border-left: 4px solid #e17055;
            border-radius: 8px;
            padding: 20px 25px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .drawdown-analysis-box p {{
            margin: 0;
            line-height: 1.8;
            font-size: 15px;
            color: #2c3e50;
            text-align: justify;
        }}
        
        .drawdown-analysis-box .dd-value {{
            color: #d63031;
            font-weight: bold;
        }}
        
        .drawdown-analysis-box .dd-date {{
            color: #e17055;
            font-weight: bold;
        }}
        
        .drawdown-analysis-box .dd-recovered {{
            color: #00b894;
            font-weight: bold;
        }}
        
        .drawdown-analysis-box .dd-not-recovered {{
            color: #d63031;
            font-weight: bold;
        }}
        
        /* 收益指标展示样式 */
        .return-metrics-row {{
            display: flex;
            justify-content: space-around;
            margin-bottom: 20px;
            gap: 15px;
        }}
        
        .return-metric-item {{
            flex: 1;
            text-align: center;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 8px;
            color: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .return-metric-label {{
            font-size: 14px;
            font-weight: 500;
            margin-bottom: 8px;
            opacity: 0.9;
        }}
        
        .return-metric-value {{
            font-size: 24px;
            font-weight: bold;
        }}
        
        /* 风险收益指标展示样式 */
        .risk-return-metrics-row {{
            display: flex;
            justify-content: space-around;
            margin-bottom: 20px;
            gap: 15px;
        }}
        
        .risk-return-metric-item {{
            flex: 1;
            text-align: center;
            padding: 15px;
            background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
            border-radius: 8px;
            color: #2e7d32;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .risk-return-metric-label {{
            font-size: 14px;
            font-weight: 500;
            margin-bottom: 8px;
            opacity: 0.8;
        }}
        
        .risk-return-metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #1b5e20;
        }}
        
        .risk-return-metric-desc {{
            font-size: 12px;
            color: #666;
            margin-top: 4px;
            opacity: 0.7;
        }}
    </style>
</head>
<body>
    <div class="container-fluid py-4">
        <!-- 标题部分 -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card shadow-sm">
                    <div class="card-body text-center">
                        <h1 class="display-4 mb-3"><i class="fas fa-chart-line me-3"></i>投资组合业绩归因分析报告</h1>
                        <div class="row">
                            <div class="col-md-6">
                                <h5><i class="fas fa-calendar-alt me-2"></i>回测开始日期</h5>
                                <p class="lead">{analysis.get('start_date', '').strftime('%Y-%m-%d') if analysis.get('start_date') else 'N/A'}</p>
                            </div>
                            <div class="col-md-6">
                                <h5><i class="fas fa-calendar-check me-2"></i>回测结束日期</h5>
                                <p class="lead">{analysis.get('end_date', '').strftime('%Y-%m-%d') if analysis.get('end_date') else 'N/A'}</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- 关键指标卡片 -->
        <div class="row mb-4">
            <div class="col-12">
                <h2 class="section-header"><i class="fas fa-table me-2"></i>核心业绩指标</h2>
                <div class="chart-container">
                    <div class="table-responsive">
                        <table class="table stats-table table-striped" id="yearlyStatsTable">
                            <thead>
                                <tr>
                                    <th>年份</th>
                                    <th>年化收益率(%)</th>
                                    <th>超额收益率(%)</th>
                                    <th>年化波动率(%)</th>
                                    <th>夏普比率</th>
                                    <th>最大回撤(%)</th>
                                    <th>胜率(%)</th>
                                    <th>盈亏比</th>
                                    <th>交易次数</th>
                                    <th>换手率(%)</th>
                                    <th>交易天数</th>
                                </tr>
                            </thead>
                            <tbody id="yearlyStatsBody">
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- 净值走势图 -->
        <div class="row mb-4">
            <div class="col-12">
                <h2 class="section-header"><i class="fas fa-chart-line me-2"></i>净值走势分析</h2>
                <div class="chart-container">
                    <!-- 收益指标展示行 -->
                    <div class="return-metrics-row">
                        <div class="return-metric-item">
                            <div class="return-metric-label"><i class="fas fa-percentage me-2"></i>总收益率</div>
                            <div class="return-metric-value">{analysis.get('total_return', 0):.2f}%</div>
                        </div>
                        <div class="return-metric-item">
                            <div class="return-metric-label"><i class="fas fa-calendar-alt me-2"></i>年化收益率</div>
                            <div class="return-metric-value">{analysis.get('annual_return', 0):.2f}%</div>
                        </div>
                        <div class="return-metric-item">
                            <div class="return-metric-label"><i class="fas fa-arrow-up me-2"></i>超额收益率</div>
                            <div class="return-metric-value">{analysis.get('excess_return', 0):.2f}%</div>
                        </div>
                        <div class="return-metric-item">
                            <div class="return-metric-label"><i class="fas fa-rocket me-2"></i>年化超额收益率</div>
                            <div class="return-metric-value">{analysis.get('annual_excess_return', 0):.2f}%</div>
                        </div>
                    </div>
                    <!-- 净值走势图表 -->
                    <div id="netValueChart" style="height: 600px;"></div>
                    
                    <!-- 月度收益分析 -->
                    <div class="monthly-analysis-box">
                        <p>
                            从策略月度收益情况来看，回测期间共计<span class="highlight">{data['monthly_statistics']['total_months']}</span>个交易月，
                            其中<span class="positive">{data['monthly_statistics']['positive_months']}</span>个月实现正收益，
                            <span class="highlight">{data['monthly_statistics']['outperform_months']}</span>个月成功跑赢基准指数。
                            策略的月度胜率达到<span class="highlight">{data['monthly_statistics']['monthly_win_rate']:.2f}%</span>，
                            相对胜率为<span class="highlight">{data['monthly_statistics']['relative_win_rate']:.2f}%</span>，
                            其中，
                            <span class="highlight">{data['monthly_statistics']['best_month_date']}</span>表现最为出色，
                            单月收益率为<span class="positive">{data['monthly_statistics']['best_month_return']:.2f}%</span>。
                        </p>
                    </div>
                    
                    <!-- 月度收益率热力图 -->
                    <div id="heatmapContainer"></div>
                </div>
            </div>
        </div>
        
        <!-- 回撤走势图 -->
        <div class="row mb-4">
            <div class="col-12">
                <h2 class="section-header"><i class="fas fa-arrow-down me-2"></i>回撤分析</h2>
                <div class="chart-container">
                    <!-- 回撤指标展示行 -->
                    <div class="drawdown-metrics-row">
                        <div class="drawdown-metric-item">
                            <div class="drawdown-metric-label"><i class="fas fa-history me-2"></i>历史最大回撤</div>
                            <div class="drawdown-metric-value">{data['drawdown_statistics']['historical_max_dd']:.2f}%</div>
                        </div>
                        <div class="drawdown-metric-item">
                            <div class="drawdown-metric-label"><i class="fas fa-calendar-alt me-2"></i>今年最大回撤</div>
                            <div class="drawdown-metric-value">{data['drawdown_statistics']['ytd_max_dd']:.2f}%</div>
                        </div>
                        <div class="drawdown-metric-item">
                            <div class="drawdown-metric-label"><i class="fas fa-calendar-week me-2"></i>近3个月最大回撤</div>
                            <div class="drawdown-metric-value">{data['drawdown_statistics']['recent_3m_max_dd']:.2f}%</div>
                        </div>
                        <div class="drawdown-metric-item">
                            <div class="drawdown-metric-label"><i class="fas fa-calendar-day me-2"></i>近1年最大回撤</div>
                            <div class="drawdown-metric-value">{data['drawdown_statistics']['recent_1y_max_dd']:.2f}%</div>
                        </div>
                    </div>
                    
                    <!-- 回撤分析文本框 -->
                    <div class="drawdown-analysis-box">
                        <p>
                            数据显示，组合历史最大回撤为<span class="dd-value">{data['drawdown_statistics']['historical_max_dd']:.2f}%</span>，
                            发生于<span class="dd-date">{data['drawdown_statistics']['historical_start_date']}</span>至<span class="dd-date">{data['drawdown_statistics']['historical_end_date']}</span>。
                            {'最大回撤于<span class="dd-recovered">' + data['drawdown_statistics']['historical_recovery_date'] + '</span>实现修复，回撤修复耗时<span class="dd-recovered">' + str(data['drawdown_statistics']['historical_recovery_days']) + '</span>天。' if data['drawdown_statistics']['historical_recovered'] else '截止最新，<span class="dd-not-recovered">最大回撤未实现修复</span>。'}
                            近一年最大回撤为<span class="dd-value">{data['drawdown_statistics']['recent_1y_max_dd']:.2f}%</span>，
                            发生于<span class="dd-date">{data['drawdown_statistics']['recent_1y_start_date']}</span>至<span class="dd-date">{data['drawdown_statistics']['recent_1y_end_date']}</span>。
                        </p>
                    </div>
                    
                    <!-- 回撤走势图表 -->
                    <div id="drawdownChart" style="height: 500px;"></div>
                </div>
            </div>
        </div>
        
        <!-- 风险收益分析 -->
        <div class="row mb-4">
            <div class="col-12">
                <h2 class="section-header"><i class="fas fa-balance-scale me-2"></i>风险收益分析</h2>
                <div class="chart-container">
                    <!-- 风险收益指标展示行 -->
                    <div class="risk-return-metrics-row">
                        <div class="risk-return-metric-item">
                            <div class="risk-return-metric-label"><i class="fas fa-chart-line me-2"></i>夏普比率</div>
                            <div class="risk-return-metric-value">{data['analysis_result'].get('sharpe_ratio', 0):.4f}</div>
                        </div>
                        <div class="risk-return-metric-item">
                            <div class="risk-return-metric-label"><i class="fas fa-trophy me-2"></i>卡玛比率</div>
                            <div class="risk-return-metric-value">{data['analysis_result'].get('calmar_ratio', 0):.4f}</div>
                        </div>
                        <div class="risk-return-metric-item">
                            <div class="risk-return-metric-label"><i class="fas fa-shield-alt me-2"></i>索提诺比率</div>
                            <div class="risk-return-metric-value">{data['analysis_result'].get('sortino_ratio', 0):.4f}</div>
                        </div>
                        <div class="risk-return-metric-item">
                            <div class="risk-return-metric-label"><i class="fas fa-wave-square me-2"></i>年化波动率</div>
                            <div class="risk-return-metric-value">{data['analysis_result'].get('annual_volatility', 0):.2f}%</div>
                        </div>
                        <div class="risk-return-metric-item">
                            <div class="risk-return-metric-label"><i class="fas fa-arrow-down me-2"></i>下行波动率</div>
                            <div class="risk-return-metric-value">{data['analysis_result'].get('downside_volatility', 0):.2f}%</div>
                        </div>
                    </div>
                    
                    <!-- Alpha分析图表（条件显示） -->
                    {'<!-- Alpha分析 -->' if data['rolling_alpha_beta'] and data['rolling_alpha_beta'].get('has_data') else ''}
                    {'<div id="alphaChart" style="height: 500px; margin-top: 20px;"></div>' if data['rolling_alpha_beta'] and data['rolling_alpha_beta'].get('has_data') else ''}
                </div>
            </div>
        </div>
        
        <!-- 交易统计分析 -->
        {'<!-- 交易统计 -->' if data['trade_statistics'] and data['trade_statistics'].get('has_data') else ''}
        {f'''
        <div class="row mb-4">
            <div class="col-12">
                <h2 class="section-header"><i class="fas fa-exchange-alt me-2"></i>交易统计分析</h2>
                <div class="chart-container">
                    <!-- 交易统计指标展示行 -->
                    <div class="risk-return-metrics-row">
                        <div class="risk-return-metric-item">
                            <div class="risk-return-metric-label"><i class="fas fa-sync-alt me-2"></i>年化换手率</div>
                            <div class="risk-return-metric-value">{data['trade_statistics']['turnover_rate']:.2f}%</div>
                            <div class="risk-return-metric-desc">双边交易/(组合市值×2)</div>
                        </div>
                        <div class="risk-return-metric-item">
                            <div class="risk-return-metric-label"><i class="fas fa-trophy me-2"></i>胜率</div>
                            <div class="risk-return-metric-value">{data['trade_statistics']['win_rate']:.2f}%</div>
                            <div class="risk-return-metric-desc">{data['trade_statistics']['win_trades']}胜/{data['trade_statistics']['loss_trades']}负</div>
                        </div>
                        <div class="risk-return-metric-item">
                            <div class="risk-return-metric-label"><i class="fas fa-balance-scale me-2"></i>盈亏比</div>
                            <div class="risk-return-metric-value">{data['trade_statistics']['profit_loss_ratio']:.2f}</div>
                            <div class="risk-return-metric-desc">平均盈利/平均亏损</div>
                        </div>
                        <div class="risk-return-metric-item">
                            <div class="risk-return-metric-label"><i class="fas fa-calendar-alt me-2"></i>平均持有</div>
                            <div class="risk-return-metric-value">{data['trade_statistics']['avg_holding_days']:.0f}天</div>
                            <div class="risk-return-metric-desc">所有卖出交易</div>
                        </div>
                        <div class="risk-return-metric-item">
                            <div class="risk-return-metric-label"><i class="fas fa-arrow-up me-2"></i>盈利持有</div>
                            <div class="risk-return-metric-value">{data['trade_statistics']['avg_profit_holding_days']:.0f}天</div>
                            <div class="risk-return-metric-desc">盈利交易平均</div>
                        </div>
                        <div class="risk-return-metric-item">
                            <div class="risk-return-metric-label"><i class="fas fa-arrow-down me-2"></i>亏损持有</div>
                            <div class="risk-return-metric-value">{data['trade_statistics']['avg_loss_holding_days']:.0f}天</div>
                            <div class="risk-return-metric-desc">亏损交易平均</div>
                        </div>
                    </div>
                    
                    <!-- 交易统计分析文字 -->
                    <div class="monthly-analysis-box">
                        <p>
                            从交易统计来看，策略的年化换手率为<span class="highlight">{data['trade_statistics']['turnover_rate']:.2f}%</span>，
                            回测期间共完成<span class="highlight">{data['trade_statistics']['total_sell_trades']}</span>笔卖出交易，
                            其中<span class="positive">{data['trade_statistics']['win_trades']}</span>笔盈利，
                            <span class="negative">{data['trade_statistics']['loss_trades']}</span>笔亏损，
                            胜率达到<span class="highlight">{data['trade_statistics']['win_rate']:.2f}%</span>。
                            盈利交易的平均收益率为<span class="positive">{data['trade_statistics']['avg_profit_return']:.2f}%</span>，
                            亏损交易的平均收益率为<span class="negative">{data['trade_statistics']['avg_loss_return']:.2f}%</span>，
                            盈亏比为<span class="highlight">{data['trade_statistics']['profit_loss_ratio']:.2f}</span>。
                            从持有周期来看，平均持有天数为<span class="highlight">{data['trade_statistics']['avg_holding_days']:.0f}</span>个交易日，
                            其中盈利交易平均持有<span class="highlight">{data['trade_statistics']['avg_profit_holding_days']:.0f}</span>天，
                            亏损交易平均持有<span class="highlight">{data['trade_statistics']['avg_loss_holding_days']:.0f}</span>天。
                        </p>
                    </div>
                    
                    <!-- 月度换手率走势图（条件显示） -->
                    {'<!-- 月度换手率走势 -->' if data['monthly_turnover'] and data['monthly_turnover'].get('has_data') else ''}
                    {'<div id="monthlyTurnoverChart" style="height: 500px; margin-top: 20px;"></div>' if data['monthly_turnover'] and data['monthly_turnover'].get('has_data') else ''}
                </div>
            </div>
        </div>
        ''' if data['trade_statistics'] and data['trade_statistics'].get('has_data') else ''}
        

        <!-- 行业业绩评估 -->
        <div class="row mb-4">
            <div class="col-12">
                <h2 class="section-header"><i class="fas fa-industry me-2"></i>行业业绩评估</h2>
                <div class="chart-container">
                    <!-- 行业业绩条形图 -->
                    <div class="row mb-4">
                        <div class="col-md-6">
                            <div class="chart-container">
                                <h5 class="text-center mb-3">损益贡献前十行业</h5>
                                <div id="topIndustriesChart" style="height: 400px;"></div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="chart-container">
                                <h5 class="text-center mb-3">损益贡献后十行业</h5>
                                <div id="bottomIndustriesChart" style="height: 400px;"></div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- 季度行业成交额统计表 -->
                    <div class="row mb-4">
                        <div class="col-12">
                            <div id="quarterlyIndustryContainer"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.bootcdn.net/ajax/libs/bootstrap/5.3.0/js/bootstrap.bundle.min.js" 
            onerror="this.onerror=null;this.src='https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js';"></script>
    
    <script>
        // 数据准备
        const reportData = {json.dumps(js_data, ensure_ascii=False)};
        
        // 中国股市颜色配置
        const colors = {{
            up: '#fc9d9a',
            down: '#acd59f',
            primary: '#1f77b4',
            secondary: '#ff7f0e'
        }};
        
        // 初始化所有图表
        document.addEventListener('DOMContentLoaded', function() {{
            console.log('DOM加载完成，开始初始化图表...');
            
            // 检查Plotly是否可用
            if (typeof Plotly === 'undefined') {{
                console.error('Plotly.js未加载');
                alert('图表库加载失败，请刷新页面重试');
                return;
            }}
            
            console.log('Plotly.js已加载');
            console.log('报告数据:', reportData);
            
            try {{
                initNetValueChart();
                console.log('净值图表初始化完成');
            }} catch (error) {{
                console.error('净值图表初始化失败:', error);
            }}
            
            try {{
                initDrawdownChart();
                console.log('回撤图表初始化完成');
            }} catch (error) {{
                console.error('回撤图表初始化失败:', error);
            }}
            
            try {{
                initAlphaChart();
                console.log('Alpha图表初始化完成');
            }} catch (error) {{
                console.error('Alpha图表初始化失败:', error);
            }}
            
            try {{
                populateYearlyStatsTable();
                console.log('年度统计表格填充完成');
            }} catch (error) {{
                console.error('年度统计表格填充失败:', error);
            }}
            
            try {{
                createHeatmap();
                console.log('热力图创建完成');
            }} catch (error) {{
                console.error('热力图创建失败:', error);
            }}
            
            try {{
                initMonthlyTurnoverChart();
                console.log('月度换手率图表初始化完成');
            }} catch (error) {{
                console.error('月度换手率图表初始化失败:', error);
            }}
            
            try {{
                initIndustryPerformanceCharts();
                console.log('行业业绩图表初始化完成');
            }} catch (error) {{
                console.error('行业业绩图表初始化失败:', error);
            }}
            
            try {{
                populateQuarterlyIndustryTable();
                console.log('季度行业统计表格填充完成');
            }} catch (error) {{
                console.error('季度行业统计表格填充失败:', error);
            }}
            
            
            console.log('所有图表初始化完成');
        }});
        
        // 净值走势图
        function initNetValueChart() {{
            const traces = [
                // 组合净值曲线
                {{
                    x: reportData.dates,
                    y: reportData.portfolioData,
                    type: 'scatter',
                    mode: 'lines',
                    name: '组合净值',
                    line: {{
                        color: '#1f77b4',
                        width: 3
                    }},
                    yaxis: 'y'
                }}
            ];
            
            // 基准净值曲线
            if (reportData.benchmarkData.length > 0) {{
                traces.push({{
                    x: reportData.dates,
                    y: reportData.benchmarkData,
                    type: 'scatter',
                    mode: 'lines',
                    name: '基准指数',
                    line: {{
                        color: '#4caf50',
                        width: 3
                    }},
                    yaxis: 'y'
                }});
                
                // 超额收益曲线（填充区域）
                if (reportData.excessReturnsData.length > 0) {{
                    // 计算超额收益率百分比
                    const excessReturnsPct = reportData.excessReturnsData.map(val => val * 100);
                    
                    traces.push({{
                        x: reportData.dates,
                        y: excessReturnsPct,
                        type: 'scatter',
                        mode: 'lines',
                        name: '超额收益(%)',
                        line: {{
                            color: 'rgba(233, 30, 99, 0.2)',
                            width: 2,
                            dash: 'dash'
                        }},
                        fill: 'tozeroy',
                        fillcolor: 'rgba(233, 30, 99, 0.2)',
                        yaxis: 'y2'
                    }});
                }}
            }}
            
            const layout = {{
                xaxis: {{
                    type: 'date',
                    tickformat: '%Y-%m-%d',
                    tickfont: {{ size: 12 }}
                }},
                yaxis: {{
                    title: '累计净值',
                    side: 'left',
                    tickformat: '.2f',
                    tickfont: {{ size: 12 }},
                    gridcolor: '#E5E5E5'
                }},
                yaxis2: {{
                    title: '超额收益率(%)',
                    side: 'right',
                    overlaying: 'y',
                    tickformat: '.1f',
                    tickfont: {{ size: 12 }},
                    gridcolor: '#FFFFFF'
                }},
                legend: {{
                    x: 0.02,
                    y: 0.98,
                    bgcolor: 'rgba(255,255,255,0.8)',
                    bordercolor: '#E5E5E5',
                    borderwidth: 1
                }},
                margin: {{ t: 60, b: 60, l: 60, r: 60 }},
                hovermode: 'x unified',
                font: {{ family: 'Arial, sans-serif' }}
            }};
            
            const config = {{
                responsive: true,
                displayModeBar: false
            }};
            
            Plotly.newPlot('netValueChart', traces, layout, config);
        }}
        
        // 回撤走势图
        function initDrawdownChart() {{
            const traces = [
                // 累计净值曲线
                {{
                    x: reportData.dates,
                    y: reportData.portfolioData,
                    type: 'scatter',
                    mode: 'lines',
                    name: '累计净值',
                    line: {{
                        color: '#1f77b4',
                        width: 3
                    }},
                    yaxis: 'y'
                }},
                // 回撤曲线
                {{
                    x: reportData.dates,
                    y: reportData.drawdownData.map(val => val * 100),
                    type: 'scatter',
                    mode: 'lines',
                    name: '回撤(%)',
                    fill: 'tonexty',
                    fillcolor: 'rgba(172, 213, 159, 0.3)',
                    line: {{
                        color: colors.down,
                        width: 3
                    }},
                    yaxis: 'y2'
                }}
            ];
            
            const layout = {{
                xaxis: {{
                    type: 'date',
                    tickformat: '%Y-%m-%d',
                    tickfont: {{ size: 12 }}
                }},
                yaxis: {{
                    title: '累计净值',
                    side: 'left',
                    tickformat: '.2f',
                    tickfont: {{ size: 12 }},
                    gridcolor: '#FFFFFF'
                }},
                yaxis2: {{
                    title: '回撤(%)',
                    side: 'right',
                    overlaying: 'y',
                    tickformat: '.1f',
                    tickfont: {{ size: 12 }},
                    range: [Math.min(...reportData.drawdownData.map(val => val * 100)) * 1.1, 5],
                    zeroline: true,
                    zerolinecolor: '#999999',
                    zerolinewidth: 1
                }},
                legend: {{
                    x: 0.02,
                    y: 0.98,
                    bgcolor: 'rgba(255,255,255,0.8)',
                    bordercolor: '#E5E5E5',
                    borderwidth: 1
                }},
                margin: {{ t: 60, b: 60, l: 60, r: 60 }},
                hovermode: 'x unified',
                font: {{ family: 'Arial, sans-serif' }}
            }};
            
            const config = {{
                responsive: true,
                displayModeBar: false
            }};
            
            Plotly.newPlot('drawdownChart', traces, layout, config);
        }}
        
        // 月度换手率走势图
        function initMonthlyTurnoverChart() {{
            // 检查数据是否存在
            if (!reportData.monthlyTurnover || !reportData.monthlyTurnover.has_data) {{
                console.log('月度换手率数据不存在，跳过图表初始化');
                return;
            }}
            
            const turnoverData = reportData.monthlyTurnover;
            
            const traces = [
                // 日均换手率柱状图
                {{
                    x: turnoverData.dates,
                    y: turnoverData.daily_turnover,
                    type: 'bar',
                    name: '日均换手率(%)',
                    marker: {{
                        color: colors.down,
                        opacity: 0.7,
                        line: {{
                            color: colors.down,
                            width: 1
                        }}
                    }},
                    yaxis: 'y2'
                }},
                // 基准净值曲线
                {{
                    x: turnoverData.benchmark_dates,
                    y: turnoverData.benchmark_values,
                    type: 'scatter',
                    mode: 'lines',
                    name: '基准净值',
                    line: {{
                        color: '#1f77b4',
                        width: 3
                    }},
                    yaxis: 'y'
                }}
            ];
            
            const layout = {{
                title: {{
                    text: '月度日均换手率走势',
                    font: {{ size: 16, color: '#2c3e50' }}
                }},
                xaxis: {{
                    type: 'date',
                    tickformat: '%Y-%m',
                    tickfont: {{ size: 12 }}
                }},
                yaxis: {{
                    title: '基准净值',
                    side: 'left',
                    tickformat: '.2f',
                    tickfont: {{ size: 12 }},
                    gridcolor: '#FFFFFF'
                }},
                yaxis2: {{
                    title: '日均换手率(%)',
                    side: 'right',
                    overlaying: 'y',
                    tickformat: '.2f',
                    tickfont: {{ size: 12 }},
                    rangemode: 'tozero',
                    zeroline: true,
                    zerolinecolor: '#999999',
                    zerolinewidth: 1
                }},
                legend: {{
                    x: 0.02,
                    y: 0.98,
                    bgcolor: 'rgba(255,255,255,0.8)',
                    bordercolor: '#E5E5E5',
                    borderwidth: 1
                }},
                margin: {{ t: 60, b: 60, l: 60, r: 60 }},
                hovermode: 'x unified',
                font: {{ family: 'Arial, sans-serif' }}
            }};
            
            const config = {{
                responsive: true,
                displayModeBar: false
            }};
            
            Plotly.newPlot('monthlyTurnoverChart', traces, layout, config);
        }}
        
        // Alpha和Beta时序图
        function initAlphaChart() {{
            // 检查是否有数据
            if (!reportData.rollingAlphaBeta || !reportData.rollingAlphaBeta.has_data) {{
                console.log('Alpha数据不足，跳过图表初始化');
                return;
            }}
            
            const data = reportData.rollingAlphaBeta;
            
            // 计算Alpha和Beta的范围，用于对齐0刻度
            const alphaMin = Math.min(...data.alphas);
            const alphaMax = Math.max(...data.alphas);
            const betaMin = Math.min(...data.betas);
            const betaMax = Math.max(...data.betas);
            
            // 计算对齐0刻度的范围
            // 如果数据跨越0，则确保0在相同的相对位置
            let alphaRange, betaRange;
            
            if (alphaMin < 0 && alphaMax > 0) {{
                // Alpha跨越0
                const alphaAbsMax = Math.max(Math.abs(alphaMin), Math.abs(alphaMax));
                alphaRange = [-alphaAbsMax * 1.1, alphaAbsMax * 1.1];
            }} else if (alphaMax <= 0) {{
                // Alpha全为负
                alphaRange = [alphaMin * 1.1, 0];
            }} else {{
                // Alpha全为正
                alphaRange = [0, alphaMax * 1.1];
            }}
            
            if (betaMin < 0 && betaMax > 0) {{
                // Beta跨越0
                const betaAbsMax = Math.max(Math.abs(betaMin), Math.abs(betaMax));
                betaRange = [-betaAbsMax * 1.1, betaAbsMax * 1.1];
            }} else if (betaMax <= 0) {{
                // Beta全为负
                betaRange = [betaMin * 1.1, 0];
            }} else {{
                // Beta全为正
                betaRange = [0, betaMax * 1.1];
            }}
            
            // 如果两者都跨越0，调整范围使0刻度对齐
            if (alphaMin < 0 && alphaMax > 0 && betaMin < 0 && betaMax > 0) {{
                const alphaZeroPos = Math.abs(alphaRange[0]) / (alphaRange[1] - alphaRange[0]);
                const betaZeroPos = Math.abs(betaRange[0]) / (betaRange[1] - betaRange[0]);
                
                if (alphaZeroPos > betaZeroPos) {{
                    // 调整Beta范围
                    const betaNegRange = betaRange[1] * alphaZeroPos / (1 - alphaZeroPos);
                    betaRange = [-betaNegRange, betaRange[1]];
                }} else if (betaZeroPos > alphaZeroPos) {{
                    // 调整Alpha范围
                    const alphaNegRange = alphaRange[1] * betaZeroPos / (1 - betaZeroPos);
                    alphaRange = [-alphaNegRange, alphaRange[1]];
                }}
            }}
            
            const traces = [
                {{
                    x: data.dates,
                    y: data.betas,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Beta',
                    line: {{
                        color: '#1f77b4',
                        width: 3
                    }},
                    yaxis: 'y2'
                }},
                {{
                    x: data.dates,
                    y: data.alphas,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Alpha(%)',
                    line: {{
                        color: 'rgba(233, 30, 99, 0.2)',
                        width: 2,
                        dash: 'dash'
                    }},
                    fill: 'tozeroy',
                    fillcolor: 'rgba(233, 30, 99, 0.2)',
                    yaxis: 'y'
                }}
            ];
            
            const layout = {{
                xaxis: {{
                    type: 'date',
                    tickformat: '%Y-%m-%d',
                    tickfont: {{ size: 12 }}
                }},
                yaxis: {{
                    title: 'Alpha(%)',
                    side: 'left',
                    tickformat: '.2f',
                    tickfont: {{ size: 12 }},
                    gridcolor: '#E5E5E5',
                    range: alphaRange,
                    zeroline: true,
                    zerolinecolor: '#999999',
                    zerolinewidth: 1
                }},
                yaxis2: {{
                    title: 'Beta',
                    side: 'right',
                    overlaying: 'y',
                    tickformat: '.2f',
                    tickfont: {{ size: 12 }},
                    gridcolor: '#FFFFFF',
                    range: betaRange,
                    zeroline: true,
                    zerolinecolor: '#999999',
                    zerolinewidth: 1
                }},
                legend: {{
                    x: 0.02,
                    y: 0.98,
                    bgcolor: 'rgba(255,255,255,0.8)',
                    bordercolor: '#E5E5E5',
                    borderwidth: 1
                }},
                margin: {{ t: 60, b: 60, l: 60, r: 60 }},
                hovermode: 'x unified',
                font: {{ family: 'Arial, sans-serif' }}
            }};
            
            const config = {{
                responsive: true,
                displayModeBar: false
            }};
            
            Plotly.newPlot('alphaChart', traces, layout, config);
        }}
        
        // 填充年度统计表格
        function populateYearlyStatsTable() {{
            const tbody = document.getElementById('yearlyStatsBody');
            
            reportData.yearlyStats.forEach(stats => {{
                const row = document.createElement('tr');
                
                const returnClass = stats.annual_return >= 0 ? 'positive' : 'negative';
                const excessClass = (stats.excess_return || 0) >= 0 ? 'positive' : 'negative';
                
                row.innerHTML = `
                    <td>${{stats.year}}</td>
                    <td class="${{returnClass}}">${{stats.annual_return.toFixed(2)}}%</td>
                    <td class="${{excessClass}}">${{(stats.excess_return || 0).toFixed(2)}}%</td>
                    <td>${{stats.volatility.toFixed(2)}}%</td>
                    <td>${{stats.sharpe_ratio.toFixed(4)}}</td>
                    <td class="negative">${{stats.max_drawdown.toFixed(2)}}%</td>
                    <td>${{stats.win_rate.toFixed(1)}}%</td>
                    <td>${{stats.profit_loss_ratio.toFixed(2)}}</td>
                    <td>${{stats.trading_count}}</td>
                    <td>${{(stats.turnover_rate || 0).toFixed(2)}}%</td>
                    <td>${{stats.trading_days}}</td>
                `;
                
                tbody.appendChild(row);
            }});
        }}
        
        // 创建月度收益率热力图
        function createHeatmap() {{
            const container = document.getElementById('heatmapContainer');
            const monthlyData = reportData.monthlyReturns;
            
            // 创建网格
            const grid = document.createElement('div');
            grid.className = 'heatmap-container';
            
            // 添加月份标题
            const months = ['年份/类型', '1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月'];
            months.forEach(month => {{
                const cell = document.createElement('div');
                cell.className = 'heatmap-cell month-header';
                cell.textContent = month;
                grid.appendChild(cell);
            }});
            
            // 按年份排序并添加数据行
            const sortedYears = Object.keys(monthlyData).sort();
            
            sortedYears.forEach(year => {{
                const yearData = monthlyData[year];
                
                // 第一行：年份 + 组合收益率
                const yearCell = document.createElement('div');
                yearCell.className = 'heatmap-cell year-header';
                yearCell.textContent = year + '/收益';
                grid.appendChild(yearCell);
                
                // 组合收益率月度数据
                for (let month = 1; month <= 12; month++) {{
                    const cell = document.createElement('div');
                    cell.className = 'heatmap-cell';
                    
                    const value = yearData.portfolio ? yearData.portfolio[month.toString()] : null;
                    
                    if (value !== null && value !== undefined) {{
                        cell.textContent = value.toFixed(1) + '%';
                        
                        // 根据收益率设置颜色 - 使用中国股市颜色习惯
                        if (value > 0) {{
                            const intensity = Math.min(Math.abs(value) / 10, 1);
                            cell.style.backgroundColor = 'rgba(252, 157, 154, ' + (0.3 + intensity * 0.7) + ')';
                            cell.style.color = '#c62828';
                        }} else if (value < 0) {{
                            const intensity = Math.min(Math.abs(value) / 10, 1);
                            cell.style.backgroundColor = 'rgba(172, 213, 159, ' + (0.3 + intensity * 0.7) + ')';
                            cell.style.color = '#2e7d32';
                        }} else {{
                            // 收益率为0%使用浅灰色
                            cell.style.backgroundColor = '#e0e0e0';
                            cell.style.color = '#666666';
                        }}
                    }} else {{
                        cell.textContent = '-';
                        cell.style.backgroundColor = '#f5f5f5';
                        cell.style.color = '#757575';
                    }}
                    
                    grid.appendChild(cell);
                }}
                
                // 第二行：超额收益率（如果有基准数据）
                if (yearData.excess) {{
                    const excessCell = document.createElement('div');
                    excessCell.className = 'heatmap-cell excess-header';
                    excessCell.textContent = year + '/超额';
                    grid.appendChild(excessCell);
                    
                    // 超额收益率月度数据
                    for (let month = 1; month <= 12; month++) {{
                        const cell = document.createElement('div');
                        cell.className = 'heatmap-cell';
                        
                        const value = yearData.excess[month.toString()];
                        
                        if (value !== null && value !== undefined) {{
                            cell.textContent = value.toFixed(1) + '%';
                            
                            // 超额收益率使用与组合收益率相同的红绿配色
                            if (value > 0) {{
                                const intensity = Math.min(Math.abs(value) / 10, 1);
                                cell.style.backgroundColor = 'rgba(252, 157, 154, ' + (0.3 + intensity * 0.7) + ')';
                                cell.style.color = '#c62828';
                            }} else if (value < 0) {{
                                const intensity = Math.min(Math.abs(value) / 10, 1);
                                cell.style.backgroundColor = 'rgba(172, 213, 159, ' + (0.3 + intensity * 0.7) + ')';
                                cell.style.color = '#2e7d32';
                            }} else {{
                                // 超额收益率为0%使用浅灰色
                                cell.style.backgroundColor = '#e0e0e0';
                                cell.style.color = '#666666';
                            }}
                        }} else {{
                            cell.textContent = '-';
                            cell.style.backgroundColor = '#f5f5f5';
                            cell.style.color = '#757575';
                        }}
                        
                        grid.appendChild(cell);
                    }}
                }}
            }});
            
            container.appendChild(grid);
        }}
        
        // 行业业绩图表初始化
        function initIndustryPerformanceCharts() {{
            const industryData = reportData.industryPerformance;
            console.log('行业业绩数据:', industryData);
            
            if (!industryData || !industryData.top_10_industries) {{
                console.log('没有行业业绩数据或前十行业数据');
                return;
            }}
            
            // 准备前十行业数据（按损益边际贡献，反转顺序让最大值在上方）
            const topIndustries = industryData.top_10_industries || [];
            const topLabels = topIndustries.map(item => item[0]).reverse();
            const topMarginalContribution = topIndustries.map(item => item[1].marginal_contribution_pct).reverse();
            const topAmountRatio = topIndustries.map(item => item[1].amount_ratio_pct).reverse();
            
            // 准备后十行业数据（按损益边际贡献，反转顺序让较大值在上方）
            const bottomIndustries = industryData.bottom_10_industries || [];
            const bottomLabels = bottomIndustries.map(item => item[0]);
            const bottomMarginalContribution = bottomIndustries.map(item => item[1].marginal_contribution_pct);
            const bottomAmountRatio = bottomIndustries.map(item => item[1].amount_ratio_pct);
            
            // 创建前十行业双条形图
            console.log('前十行业数据:', topLabels, topMarginalContribution, topAmountRatio);
            if (topLabels.length > 0) {{
                // 损益边际贡献条形
                const topMarginalTrace = {{
                    x: topMarginalContribution,
                    y: topLabels,
                    name: '损益边际贡献(%)',
                    type: 'bar',
                    orientation: 'h',
                    marker: {{
                        color: '#1f77b4',
                        line: {{
                            color: '#1565c0',
                            width: 1
                        }}
                    }},
                    text: topMarginalContribution.map(v => v.toFixed(2) + '%'),
                    textposition: 'outside',
                    textfont: {{
                        size: 10,
                        color: '#333333'
                    }},
                    hovertemplate: '<b>%{{y}}</b><br>损益边际贡献: %{{x:.2f}}%<extra></extra>'
                }};
                
                // 成交占比条形
                const topAmountTrace = {{
                    x: topAmountRatio,
                    y: topLabels,
                    name: '成交占比(%)',
                    type: 'bar',
                    orientation: 'h',
                    marker: {{
                        color: '#acd59f',
                        line: {{
                            color: '#81c784',
                            width: 1
                        }}
                    }},
                    text: topAmountRatio.map(v => v.toFixed(2) + '%'),
                    textposition: 'outside',
                    textfont: {{
                        size: 10,
                        color: '#333333'
                    }},
                    hovertemplate: '<b>%{{y}}</b><br>成交占比: %{{x:.2f}}%<extra></extra>'
                }};
                
                const topLayout = {{
                    xaxis: {{
                        tickfont: {{ size: 12 }},
                        showgrid: true,
                        gridcolor: '#E5E5E5',
                        zeroline: true,
                        zerolinecolor: '#999999',
                        zerolinewidth: 1
                    }},
                    yaxis: {{
                        title: '',
                        automargin: true,
                        tickfont: {{ size: 12 }},
                        gridcolor: '#E5E5E5'
                    }},
                    margin: {{ t: 60, b: 60, l: 60, r: 60 }},
                    plot_bgcolor: 'white',
                    paper_bgcolor: 'white',
                    showlegend: true,
                    legend: {{
                        x: 0.7,
                        y: 0.1,
                        bgcolor: 'rgba(255,255,255,0.8)',
                        bordercolor: '#E5E5E5',
                        borderwidth: 1
                    }},
                    barmode: 'group',
                    hovermode: 'x unified',
                    font: {{ family: 'Arial, sans-serif' }}
                }};
                
                const topConfig = {{
                    responsive: true,
                    displayModeBar: false
                }};
                
                Plotly.newPlot('topIndustriesChart', [topMarginalTrace, topAmountTrace], topLayout, topConfig);
                console.log('前十行业图表创建成功');
            }} else {{
                console.log('前十行业数据为空，无法创建图表');
            }}
            
            // 创建后十行业双条形图
            console.log('后十行业数据:', bottomLabels, bottomMarginalContribution, bottomAmountRatio);
            if (bottomLabels.length > 0) {{
                // 损益边际贡献条形
                const bottomMarginalTrace = {{
                    x: bottomMarginalContribution,
                    y: bottomLabels,
                    name: '损益边际贡献(%)',
                    type: 'bar',
                    orientation: 'h',
                    marker: {{
                        color: '#1f77b4',
                        line: {{
                            color: '#1565c0',
                            width: 1
                        }}
                    }},
                    text: bottomMarginalContribution.map(v => v.toFixed(2) + '%'),
                    textposition: 'outside',
                    textfont: {{
                        size: 10,
                        color: '#333333'
                    }},
                    hovertemplate: '<b>%{{y}}</b><br>损益边际贡献: %{{x:.2f}}%<extra></extra>'
                }};
                
                // 成交占比条形
                const bottomAmountTrace = {{
                    x: bottomAmountRatio,
                    y: bottomLabels,
                    name: '成交占比(%)',
                    type: 'bar',
                    orientation: 'h',
                    marker: {{
                        color: '#acd59f',
                        line: {{
                            color: '#81c784',
                            width: 1
                        }}
                    }},
                    text: bottomAmountRatio.map(v => v.toFixed(2) + '%'),
                    textposition: 'outside',
                    textfont: {{
                        size: 10,
                        color: '#333333'
                    }},
                    hovertemplate: '<b>%{{y}}</b><br>成交占比: %{{x:.2f}}%<extra></extra>'
                }};
                
                const bottomLayout = {{
                    xaxis: {{
                        tickfont: {{ size: 12 }},
                        showgrid: true,
                        gridcolor: '#E5E5E5',
                        zeroline: true,
                        zerolinecolor: '#999999',
                        zerolinewidth: 1
                    }},
                    yaxis: {{
                        title: '',
                        automargin: true,
                        tickfont: {{ size: 12 }},
                        gridcolor: '#E5E5E5'
                    }},
                    margin: {{ t: 60, b: 60, l: 60, r: 60 }},
                    plot_bgcolor: 'white',
                    paper_bgcolor: 'white',
                    showlegend: true,
                    legend: {{
                        x: 0.7,
                        y: 0.1,
                        bgcolor: 'rgba(255,255,255,0.8)',
                        bordercolor: '#E5E5E5',
                        borderwidth: 1
                    }},
                    barmode: 'group',
                    hovermode: 'x unified',
                    font: {{ family: 'Arial, sans-serif' }}
                }};
                
                const bottomConfig = {{
                    responsive: true,
                    displayModeBar: false
                }};
                
                Plotly.newPlot('bottomIndustriesChart', [bottomMarginalTrace, bottomAmountTrace], bottomLayout, bottomConfig);
                console.log('后十行业图表创建成功');
            }} else {{
                console.log('后十行业数据为空，可能行业总数不足10个');
                // 在后十行业图表容器中显示提示信息
                const bottomChartDiv = document.getElementById('bottomIndustriesChart');
                if (bottomChartDiv) {{
                    bottomChartDiv.innerHTML = '<div style="text-align: center; padding: 50px; color: #666; font-family: Arial, sans-serif; font-size: 14px;">行业数量不足，无后十行业数据</div>';
                }}
            }}
        }}
        
        // 填充季度行业成交额统计表格
        function populateQuarterlyIndustryTable() {{
            const quarterlyData = reportData.industryPerformance?.quarterly_industry_stats;
            
            if (!quarterlyData || !quarterlyData.quarters || quarterlyData.quarters.length === 0) {{
                console.log('没有季度行业统计数据');
                return;
            }}
            
            const container = document.getElementById('quarterlyIndustryContainer');
            if (!container) {{
                console.log('找不到季度统计容器');
                return;
            }}
            
            // 创建网格容器
            const grid = document.createElement('div');
            grid.className = 'quarterly-container';
            
            // 添加表头
            const headers = ['季度', '', '', '', '', '', '', '', '', '', ''];
            headers.forEach(header => {{
                const cell = document.createElement('div');
                cell.className = 'quarterly-cell quarterly-header';
                cell.textContent = header;
                grid.appendChild(cell);
            }});
            
            // 遍历每个季度
            quarterlyData.quarters.forEach(quarter => {{
                // 季度列
                const quarterCell = document.createElement('div');
                quarterCell.className = 'quarterly-cell quarterly-quarter';
                quarterCell.textContent = quarter;
                grid.appendChild(quarterCell);
                
                // 前十行业列
                for (let i = 1; i <= 10; i++) {{
                    const cell = document.createElement('div');
                    
                    const industryKey = `industry_${{i}}`;
                    const quarterStats = quarterlyData.industry_stats[quarter];
                    
                    if (quarterStats && quarterStats.top_industries && quarterStats.top_industries[industryKey]) {{
                        const industry = quarterStats.top_industries[industryKey];
                        cell.className = 'quarterly-cell quarterly-industry';
                        cell.innerHTML = `${{industry.name}}<span style="color: #ff6b6b;">(${{industry.ratio.toFixed(1)}}%)</span>`;
                    }} else {{
                        cell.className = 'quarterly-cell quarterly-empty';
                        cell.textContent = '--';
                    }}
                    
                    grid.appendChild(cell);
                }}
            }});
            
            // 清空容器并添加网格
            container.innerHTML = '';
            container.appendChild(grid);
            
            console.log('季度行业统计表格填充完成');
        }}
    </script>
</body>
</html>
"""
        
        return html_content
    
    def _generate_fallback_report(self, portfolio_returns: pd.Series, 
                                 benchmark_returns: pd.Series,
                                 analysis_result: Dict[str, Any],
                                 output_file: str):
        """生成简化版HTML报告作为备选"""
        try:
            total_return = analysis_result.get('total_return', 0)
            annual_return = analysis_result.get('annual_return', 0)
            max_drawdown = analysis_result.get('max_drawdown', 0)
            sharpe_ratio = analysis_result.get('sharpe_ratio', 0)
            
            simple_html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>简化版业绩报告</title>
    <!-- Bootstrap CSS -->
    <link href="https://cdn.bootcdn.net/ajax/libs/bootstrap/5.3.0/css/bootstrap.min.css" rel="stylesheet" 
          onerror="this.onerror=null;this.href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css';">
    <!-- Font Awesome -->
    <link href="https://cdn.bootcdn.net/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet"
          onerror="this.onerror=null;this.href='https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css';">
    <style>
        .metric-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }}
        .section-header {{ border-left: 4px solid #007bff; padding-left: 15px; margin: 30px 0 20px 0; }}
    </style>
</head>
<body>
    <div class="container-fluid py-4">
        <!-- 标题部分 -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card shadow-sm">
                    <div class="card-body text-center">
                        <h1 class="display-4 mb-3"><i class="fas fa-chart-line me-3"></i>投资组合业绩报告（简化版）</h1>
                        <div class="alert alert-warning" role="alert">
                            <i class="fas fa-exclamation-triangle me-2"></i>
                            注：由于数据处理错误，已生成简化版报告。请检查数据格式。
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- 关键指标卡片 -->
        <div class="row mb-4">
            <div class="col-12">
                <h2 class="section-header"><i class="fas fa-chart-bar me-2"></i>核心业绩指标</h2>
                <div class="row">
                    <div class="col-md-3">
                        <div class="card metric-card h-100">
                            <div class="card-body text-center">
                                <h5 class="card-title"><i class="fas fa-percentage me-2"></i>总收益率</h5>
                                <h2 class="card-text">{total_return:.2f}%</h2>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card metric-card h-100">
                            <div class="card-body text-center">
                                <h5 class="card-title"><i class="fas fa-calendar-alt me-2"></i>年化收益率</h5>
                                <h2 class="card-text">{annual_return:.2f}%</h2>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card metric-card h-100" style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);">
                            <div class="card-body text-center">
                                <h5 class="card-title"><i class="fas fa-arrow-down me-2"></i>最大回撤</h5>
                                <h2 class="card-text">{max_drawdown:.2f}%</h2>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card metric-card h-100" style="background: linear-gradient(135deg, #4834d4 0%, #686de0 100%);">
                            <div class="card-body text-center">
                                <h5 class="card-title"><i class="fas fa-chart-bar me-2"></i>夏普比率</h5>
                                <h2 class="card-text">{sharpe_ratio:.4f}</h2>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.bootcdn.net/ajax/libs/bootstrap/5.3.0/js/bootstrap.bundle.min.js" 
            onerror="this.onerror=null;this.src='https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js';"></script>
</body>
</html>
"""
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(simple_html)
                
        except Exception as e:
            print(f"连简化版报告也生成失败: {str(e)}")
