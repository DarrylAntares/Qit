"""
标准绩效分析器实现
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from datetime import datetime
from typing import Dict, Any
import warnings

# 导入自主HTML报告生成器
from .advanced_html_report_generator import AdvancedHTMLReportGenerator

# 设置matplotlib中文支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

try:
    from .base_performance_analyzer import BasePerformanceAnalyzer
    from ..backtest_engine.base_backtest_engine import BacktestResult
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from performance.base_performance_analyzer import BasePerformanceAnalyzer
    from backtest_engine.base_backtest_engine import BacktestResult

class StandardPerformanceAnalyzer(BasePerformanceAnalyzer):
    """标准绩效分析器"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 基准数据配置
        self.benchmark_path = config.get('BENCHMARK_PATH', '../../dataset/index_daily/000300_SH.feather')
        self.benchmark_data = None
    
    def _load_benchmark_data(self, start_date: datetime, end_date: datetime) -> pd.Series:
        """
        加载基准数据（沪深300指数）
        
        参数:
            start_date: 开始日期
            end_date: 结束日期
            
        返回:
            pd.Series: 基准收益率序列
        """
        try:
            if self.benchmark_data is None:
                # 加载基准数据
                if os.path.exists(self.benchmark_path):
                    benchmark_df = pd.read_feather(self.benchmark_path)
                    
                    # 处理不同的日期列名
                    date_col = None
                    if 'date' in benchmark_df.columns:
                        date_col = 'date'
                    elif 'trade_date' in benchmark_df.columns:
                        date_col = 'trade_date'
                    
                    if date_col:
                        # 转换日期格式
                        if benchmark_df[date_col].dtype == 'object' or benchmark_df[date_col].dtype == 'int64':
                            # 处理YYYYMMDD格式的整数日期
                            benchmark_df[date_col] = pd.to_datetime(benchmark_df[date_col], format='%Y%m%d')
                        else:
                            benchmark_df[date_col] = pd.to_datetime(benchmark_df[date_col])
                        
                        # 设置日期为索引
                        benchmark_df.set_index(date_col, inplace=True)
                        
                        # 按日期升序排序（确保时间序列正确）
                        benchmark_df.sort_index(inplace=True)
                    else:
                        if self.logger:
                            self.logger.warning("基准数据中未找到日期列（date或trade_date）")
                        return None
                    
                    self.benchmark_data = benchmark_df
                    if self.logger:
                        self.logger.info(f"成功加载基准数据: {self.benchmark_path}")
                        self.logger.info(f"基准数据日期范围: {benchmark_df.index.min()} 至 {benchmark_df.index.max()}")
                        self.logger.info(f"基准数据点数: {len(benchmark_df)}")
                else:
                    if self.logger:
                        self.logger.warning(f"基准数据文件不存在: {self.benchmark_path}")
                    return None
            
            # 筛选日期范围
            mask = (self.benchmark_data.index >= start_date) & (self.benchmark_data.index <= end_date)
            benchmark_subset = self.benchmark_data.loc[mask]
            
            if 'close' in benchmark_subset.columns:
                # 计算基准收益率
                benchmark_prices = benchmark_subset['close']
                benchmark_returns = benchmark_prices.pct_change().dropna()
                
                if self.logger:
                    self.logger.info(f"基准数据范围: {benchmark_returns.index[0]} 至 {benchmark_returns.index[-1]}")
                    self.logger.info(f"基准数据点数: {len(benchmark_returns)}")
                
                return benchmark_returns
            else:
                if self.logger:
                    self.logger.warning("基准数据中未找到'close'列")
                return None
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"加载基准数据失败: {str(e)}")
            return None
        
    def analyze(self, backtest_result: BacktestResult) -> Dict[str, Any]:
        """
        分析回测结果
        
        参数:
            backtest_result: 回测结果
            
        返回:
            Dict[str, Any]: 分析结果
        """
        if self.logger:
            self.logger.info("开始绩效分析")
            
        # 获取组合净值序列
        portfolio_series = backtest_result.get_portfolio_series()
        
        if len(portfolio_series) < 2:
            if self.logger:
                self.logger.warning("数据点不足，无法进行绩效分析")
            # 返回基本的空结果结构，避免KeyError
            return {
                'start_date': datetime.now(),
                'end_date': datetime.now(),
                'total_days': 0,
                'trading_days': 0,
                'years': 0,
                'start_value': 0,
                'end_value': 0,
                'total_return': 0,
                'annual_return': 0,
                'annual_volatility': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_loss_ratio': 0,
                'calmar_ratio': 0,
                'sortino_ratio': 0,
                'total_trades': 0,
                'total_commission': 0,
                'avg_positions': 0,
                'turnover_rate': 0
            }
            
        # 计算收益率序列
        returns = portfolio_series.pct_change().dropna()
        
        # 基本指标
        start_value = portfolio_series.iloc[0]
        end_value = portfolio_series.iloc[-1]
        total_return = (end_value / start_value - 1) * 100
        
        # 时间相关计算
        start_date = portfolio_series.index[0]
        end_date = portfolio_series.index[-1]
        total_days = (end_date - start_date).days
        trading_days = len(portfolio_series)
        years = total_days / 365.25
        
        # 年化收益率
        annual_return = (end_value / start_value) ** (1 / years) - 1 if years > 0 else 0
        
        # 波动率
        annual_volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        
        # 夏普比率（假设无风险利率为3%）
        risk_free_rate = 0.03
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        
        # 最大回撤
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min() * 100
        
        # 胜率和盈亏比：基于sell交易记录计算
        win_rate = 0
        profit_loss_ratio = 0
        
        # 获取交易记录
        trades_df = backtest_result.get_trades_dataframe()
        if not trades_df.empty and 'return_rate' in trades_df.columns:
            # 筛选sell交易
            sell_trades = trades_df[trades_df['action'] == 'sell']
            
            if len(sell_trades) > 0:
                # 计算胜率：收益率>0的sell交易占比
                win_trades = sell_trades[sell_trades['return_rate'] > 0]
                loss_trades = sell_trades[sell_trades['return_rate'] < 0]
                win_rate = len(win_trades) / len(sell_trades) * 100
                
                # 计算盈亏比：盈利平均收益率 / 亏损平均收益率的绝对值
                if len(win_trades) > 0 and len(loss_trades) > 0:
                    avg_profit_return = win_trades['return_rate'].mean()
                    avg_loss_return = loss_trades['return_rate'].mean()
                    profit_loss_ratio = avg_profit_return / abs(avg_loss_return) if avg_loss_return != 0 else 0
        
        # 卡尔玛比率
        calmar_ratio = annual_return / abs(max_drawdown / 100) if max_drawdown != 0 else 0
        
        # 索提诺比率
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 else 0
        sortino_ratio = (annual_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # 交易统计
        trades_df = backtest_result.get_trades_dataframe()
        total_trades = len(trades_df) if not trades_df.empty else 0
        total_commission = trades_df['commission'].sum() if not trades_df.empty and 'commission' in trades_df.columns else 0
        
        # 计算换手率
        turnover_rate = 0
        if not trades_df.empty and 'amount' in trades_df.columns:
            # 计算买方和卖方成交额
            buy_amount = trades_df[trades_df['action'] == 'buy']['amount'].sum() if 'action' in trades_df.columns else 0
            sell_amount = trades_df[trades_df['action'] == 'sell']['amount'].sum() if 'action' in trades_df.columns else 0
            
            # 换手率 = min(买方成交额, 卖方成交额) / 区间平均资产总额
            min_amount = min(buy_amount, sell_amount)
            avg_portfolio_value = portfolio_series.mean() if len(portfolio_series) > 0 else 0
            
            if avg_portfolio_value > 0:
                # 年化换手率
                turnover_rate = (min_amount / avg_portfolio_value) * (252 / trading_days) * 100
        
        # 持仓统计
        positions_df = backtest_result.get_positions_dataframe()
        avg_positions = len(positions_df.groupby('date')) / trading_days if not positions_df.empty else 0
        
        # 加载基准数据进行比较分析
        benchmark_returns = self._load_benchmark_data(start_date, end_date)
        benchmark_analysis = {}
        
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            # 对齐基准和组合收益率数据
            aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
            
            if len(aligned_returns) > 0:
                # 计算基准指标
                benchmark_total_return = (1 + aligned_benchmark).prod() - 1
                benchmark_annual_return = (1 + benchmark_total_return) ** (252 / len(aligned_benchmark)) - 1
                benchmark_volatility = aligned_benchmark.std() * np.sqrt(252)
                
                # 计算基准最大回撤
                benchmark_cumulative = (1 + aligned_benchmark).cumprod()
                benchmark_peak = benchmark_cumulative.expanding().max()
                benchmark_dd = (benchmark_cumulative - benchmark_peak) / benchmark_peak
                benchmark_max_drawdown = benchmark_dd.min() * 100
                
                # 计算基准夏普比率
                benchmark_sharpe = (benchmark_annual_return - 0.03) / benchmark_volatility if benchmark_volatility != 0 else 0
                
                # 计算相对指标 - 调整超额收益率计算逻辑为总收益率-基准总收益率
                # total_return已经是百分比形式，benchmark_total_return是小数形式，需要统一单位
                excess_return = total_return - (benchmark_total_return * 100)
                # 计算年化超额收益率
                annual_excess_return = annual_return - benchmark_annual_return
                tracking_error = (aligned_returns - aligned_benchmark).std() * np.sqrt(252)
                # 信息比率计算：超额收益率(%) / 跟踪误差(年化)
                # 需要将excess_return转换为小数形式与tracking_error匹配
                information_ratio = (excess_return / 100) / tracking_error if tracking_error != 0 else 0
                
                # 计算Beta和Alpha
                if len(aligned_returns) > 1 and aligned_benchmark.var() != 0:
                    beta = np.cov(aligned_returns, aligned_benchmark)[0, 1] / aligned_benchmark.var()
                    alpha = annual_return - (0.03 + beta * (benchmark_annual_return - 0.03))
                else:
                    beta = 0
                    alpha = 0
                
                benchmark_analysis = {
                    'benchmark_total_return': benchmark_total_return * 100,
                    'benchmark_annual_return': benchmark_annual_return * 100,
                    'benchmark_volatility': benchmark_volatility * 100,
                    'benchmark_max_drawdown': benchmark_max_drawdown,
                    'benchmark_sharpe': benchmark_sharpe,
                    'excess_return': excess_return,
                    'annual_excess_return': annual_excess_return * 100,
                    'tracking_error': tracking_error * 100,
                    'information_ratio': information_ratio,
                    'beta': beta,
                    'alpha': alpha * 100,
                    'benchmark_returns': aligned_benchmark,
                    'aligned_portfolio_returns': aligned_returns
                }
                
                if self.logger:
                    self.logger.info("基准比较分析:")
                    self.logger.info(f"基准总收益率: {benchmark_total_return * 100:.2f}%")
                    self.logger.info(f"基准年化收益率: {benchmark_annual_return * 100:.2f}%")
                    self.logger.info(f"基准年化波动率: {benchmark_volatility * 100:.2f}%")
                    self.logger.info(f"基准最大回撤: {benchmark_max_drawdown:.2f}%")
                    self.logger.info(f"超额收益: {excess_return:.2f}%")
                    self.logger.info(f"跟踪误差: {tracking_error * 100:.2f}%")
                    self.logger.info(f"信息比率: {information_ratio:.4f}")
                    self.logger.info(f"Beta: {beta:.4f}")
                    self.logger.info(f"Alpha: {alpha * 100:.2f}%")

        analysis_result = {
            # 基本信息
            'start_date': start_date,
            'end_date': end_date,
            'total_days': total_days,
            'trading_days': trading_days,
            'years': years,
            
            # 收益指标
            'start_value': start_value,
            'end_value': end_value,
            'total_return': total_return,
            'annual_return': annual_return * 100,
            
            # 风险指标
            'annual_volatility': annual_volatility * 100,
            'downside_volatility': downside_deviation * 100,
            'max_drawdown': max_drawdown,
            
            # 风险调整收益指标
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            
            # 交易统计
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'total_trades': total_trades,
            'total_commission': total_commission,
            'avg_positions': avg_positions,
            'turnover_rate': turnover_rate,
            
            # 序列数据
            'portfolio_series': portfolio_series,
            'returns_series': returns,
            'drawdown_series': drawdown * 100,
            
            # 基准比较数据
            **benchmark_analysis
        }
        
        if self.logger:
            self.logger.info("绩效分析完成")
            # 记录关键绩效归因指标
            self.logger.info("=" * 50)
            self.logger.info("绩效归因分析")
            self.logger.info("=" * 50)
            self.logger.info(f"回测期间: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
            self.logger.info(f"总收益率: {total_return:.2f}%")
            self.logger.info(f"年化收益率: {annual_return * 100:.2f}%")
            self.logger.info(f"年化波动率: {annual_volatility * 100:.2f}%")
            self.logger.info(f"下行波动率: {downside_deviation * 100:.2f}%")
            self.logger.info(f"最大回撤: {max_drawdown:.2f}%")
            self.logger.info(f"夏普比率: {sharpe_ratio:.4f}")
            self.logger.info(f"卡尔玛比率: {calmar_ratio:.4f}")
            self.logger.info(f"索提诺比率: {sortino_ratio:.4f}")
            self.logger.info(f"胜率: {win_rate:.2f}%")
            self.logger.info(f"盈亏比: {profit_loss_ratio:.4f}")
            self.logger.info(f"总交易次数: {total_trades}")
            self.logger.info(f"总手续费: {total_commission:.2f}")
            self.logger.info(f"平均持仓数: {avg_positions:.1f}")
            self.logger.info("=" * 50)
            
        return analysis_result
    
    def generate_report(self, analysis_result: Dict[str, Any], 
                       backtest_result: BacktestResult) -> str:
        """
        生成分析报告
        
        参数:
            analysis_result: 分析结果
            backtest_result: 回测结果
            
        返回:
            str: 报告内容
        """
        report = []
        report.append("=" * 80)
        report.append("回测绩效分析报告")
        report.append("=" * 80)
        
        # 基本信息
        report.append("\n【基本信息】")
        report.append(f"回测期间: {analysis_result['start_date'].strftime('%Y-%m-%d')} 至 {analysis_result['end_date'].strftime('%Y-%m-%d')}")
        report.append(f"回测天数: {analysis_result['total_days']} 天 ({analysis_result['trading_days']} 个交易日)")
        report.append(f"回测年数: {analysis_result['years']:.2f} 年")
        
        # 收益指标
        report.append("\n【收益指标】")
        report.append(f"初始资金: {analysis_result['start_value']:,.2f}")
        report.append(f"期末资金: {analysis_result['end_value']:,.2f}")
        report.append(f"总收益率: {analysis_result['total_return']:.2f}%")
        report.append(f"年化收益率: {analysis_result['annual_return']:.2f}%")
        
        # 风险指标
        report.append("\n【风险指标】")
        report.append(f"年化波动率: {analysis_result['annual_volatility']:.2f}%")
        report.append(f"最大回撤: {analysis_result['max_drawdown']:.2f}%")
        
        # 风险调整收益指标
        report.append("\n【风险调整收益指标】")
        report.append(f"夏普比率: {analysis_result['sharpe_ratio']:.4f}")
        report.append(f"卡尔玛比率: {analysis_result['calmar_ratio']:.4f}")
        report.append(f"索提诺比率: {analysis_result['sortino_ratio']:.4f}")
        
        # 交易统计
        report.append("\n【交易统计】")
        report.append(f"胜率: {analysis_result['win_rate']:.2f}%")
        report.append(f"盈亏比: {analysis_result['profit_loss_ratio']:.4f}")
        report.append(f"总交易次数: {analysis_result['total_trades']}")
        report.append(f"总手续费: {analysis_result['total_commission']:,.2f}")
        report.append(f"平均持仓数量: {analysis_result['avg_positions']:.1f}")
        report.append(f"年化换手率: {analysis_result['turnover_rate']:.2f}%")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def save_results(self, analysis_result: Dict[str, Any], 
                    backtest_result: BacktestResult, output_dir: str):
        """
        保存分析结果
        
        参数:
            analysis_result: 分析结果
            backtest_result: 回测结果
            output_dir: 输出目录
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存绩效指标
        metrics_df = pd.DataFrame([{
            'metric': k,
            'value': v
        } for k, v in analysis_result.items() 
        if not isinstance(v, (pd.Series, pd.DataFrame, datetime))])
        
        metrics_file = os.path.join(output_dir, f'performance_metrics_{timestamp}.csv')
        metrics_df.to_csv(metrics_file, index=False, encoding='utf-8-sig')
        
        # 保存净值序列
        portfolio_file = os.path.join(output_dir, f'portfolio_values_{timestamp}.csv')
        portfolio_series = analysis_result['portfolio_series']
        drawdown_series = analysis_result['drawdown_series']
        
        # 确保所有序列长度一致
        min_length = min(len(portfolio_series), len(drawdown_series))
        portfolio_series = portfolio_series.iloc[:min_length]
        drawdown_series = drawdown_series.iloc[:min_length]
        
        portfolio_df = pd.DataFrame({
            'date': portfolio_series.index,
            'portfolio_value': portfolio_series.values,
            'cumulative_return': (portfolio_series / analysis_result['start_value'] - 1) * 100,
            'drawdown': drawdown_series.values
        })
        portfolio_df.to_csv(portfolio_file, index=False, encoding='utf-8-sig')
        
        # 保存交易记录
        trades_df = backtest_result.get_trades_dataframe()
        if not trades_df.empty:
            trades_file = os.path.join(output_dir, f'trades_{timestamp}.csv')
            trades_df.to_csv(trades_file, index=False, encoding='utf-8-sig')
            
            # 将交易记录添加到分析结果中供HTML报告使用
            analysis_result['trades_data'] = trades_df.to_dict('records')
            analysis_result['total_trades'] = len(trades_df)
        else:
            analysis_result['trades_data'] = []
            analysis_result['total_trades'] = 0
        
        # 保存持仓记录
        positions_df = backtest_result.get_positions_dataframe()
        if not positions_df.empty:
            positions_file = os.path.join(output_dir, f'positions_{timestamp}.csv')
            positions_df.to_csv(positions_file, index=False, encoding='utf-8-sig')
        
        # 生成图表
        if self.config.get('GENERATE_PLOTS', True):
            self._generate_plots(analysis_result, output_dir, timestamp)
        
        # 保存报告
        report = self.generate_report(analysis_result, backtest_result)
        report_file = os.path.join(output_dir, f'performance_report_{timestamp}.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        if self.logger:
            self.logger.info(f"分析结果已保存到: {output_dir}")
    
    def _generate_plots(self, analysis_result: Dict[str, Any], output_dir: str, timestamp: str):
        """生成图表"""
        portfolio_series = analysis_result['portfolio_series']
        drawdown_series = analysis_result['drawdown_series']
        
        # 创建图形
        fig = plt.figure(figsize=(15, 10))
        
        # 创建网格布局
        gs = fig.add_gridspec(3, 2, height_ratios=[3, 1, 1], hspace=0.3, wspace=0.3)
        
        # 净值曲线
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(portfolio_series.index, portfolio_series.values, 
                label='组合净值', color='#1f77b4', linewidth=2)
        ax1.set_title('组合净值曲线', fontsize=16, pad=20)
        ax1.set_ylabel('净值', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend(fontsize=12)
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        # 回撤曲线
        ax2 = fig.add_subplot(gs[1, :])
        ax2.fill_between(drawdown_series.index, drawdown_series.values, 0, 
                        color='#ff9999', alpha=0.7, label='回撤')
        ax2.set_title('回撤曲线', fontsize=14)
        ax2.set_ylabel('回撤 (%)', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend(fontsize=10)
        
        # 收益率分布
        ax3 = fig.add_subplot(gs[2, 0])
        returns = analysis_result['returns_series'] * 100
        ax3.hist(returns, bins=50, alpha=0.7, color='#2ca02c', edgecolor='black')
        ax3.set_title('日收益率分布', fontsize=14)
        ax3.set_xlabel('日收益率 (%)', fontsize=12)
        ax3.set_ylabel('频次', fontsize=12)
        ax3.grid(True, linestyle='--', alpha=0.6)
        
        # 月度收益率热力图
        ax4 = fig.add_subplot(gs[2, 1])
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x/100).prod() - 1) * 100
        monthly_returns_pivot = monthly_returns.groupby([
            monthly_returns.index.year, 
            monthly_returns.index.month
        ]).first().unstack(fill_value=0)
        
        if not monthly_returns_pivot.empty:
            im = ax4.imshow(monthly_returns_pivot.values, cmap='RdYlGn', aspect='auto')
            ax4.set_title('月度收益率热力图', fontsize=14)
            ax4.set_xlabel('月份', fontsize=12)
            ax4.set_ylabel('年份', fontsize=12)
            
            # 设置坐标轴标签
            ax4.set_xticks(range(len(monthly_returns_pivot.columns)))
            ax4.set_xticklabels(monthly_returns_pivot.columns)
            ax4.set_yticks(range(len(monthly_returns_pivot.index)))
            ax4.set_yticklabels(monthly_returns_pivot.index)
            
            # 添加颜色条
            plt.colorbar(im, ax=ax4, label='收益率 (%)')
        
        # 保存图表
        plot_file = os.path.join(output_dir, f'performance_charts_{timestamp}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.logger:
            self.logger.info(f"图表已保存: {plot_file}")
        
        # 生成高级HTML业绩归因报告
        self._generate_advanced_html_report(analysis_result, output_dir, timestamp)
    
    def _generate_advanced_html_report(self, analysis_result: Dict[str, Any], 
                                      output_dir: str, timestamp: str):
        """
        生成高级HTML业绩归因报告
        
        参数:
            analysis_result: 分析结果
            output_dir: 输出目录
            timestamp: 时间戳
        """
        try:
            # 获取组合收益率序列
            portfolio_returns = analysis_result.get('returns_series')
            if portfolio_returns is None or len(portfolio_returns) == 0:
                if self.logger:
                    self.logger.warning("组合收益率数据为空，无法生成HTML报告")
                return
            
            # 获取基准收益率序列
            benchmark_returns = analysis_result.get('benchmark_returns')
            
            if self.logger:
                self.logger.info("开始生成高级HTML业绩归因报告...")
            
            # 使用高级HTML报告生成器
            html_file = os.path.join(output_dir, f'performance_report_{timestamp}.html')
            html_generator = AdvancedHTMLReportGenerator()
            html_generator.generate_report(portfolio_returns, benchmark_returns, 
                                         analysis_result, html_file)
            
            if self.logger:
                self.logger.info(f"高级HTML业绩归因报告已保存: {html_file}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"生成高级HTML报告失败: {str(e)}")
                import traceback
                self.logger.debug(traceback.format_exc())
                
            # 生成简化版HTML报告作为备选
            try:
                html_file = os.path.join(output_dir, f'performance_report_{timestamp}.html')
                self._generate_simple_html_report(portfolio_returns, benchmark_returns, 
                                                analysis_result, html_file)
                if self.logger:
                    self.logger.info(f"简化版HTML报告已保存: {html_file}")
            except Exception as fallback_error:
                if self.logger:
                    self.logger.error(f"简化版HTML报告也生成失败: {str(fallback_error)}")
    
    def _generate_simple_html_report(self, portfolio_returns: pd.Series, 
                                   benchmark_returns: pd.Series, 
                                   analysis_result: Dict[str, Any], 
                                   html_file: str):
        """
        生成简化版HTML报告
        
        参数:
            portfolio_returns: 组合收益率序列
            benchmark_returns: 基准收益率序列
            analysis_result: 分析结果
            html_file: HTML文件路径
        """
        # 计算累积净值曲线
        portfolio_cumulative = (1 + portfolio_returns).cumprod()
        
        # 准备图表数据
        chart_dates = [date.strftime('%Y-%m-%d') for date in portfolio_cumulative.index]
        portfolio_values = portfolio_cumulative.tolist()
        
        # 基准数据处理
        benchmark_values = []
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            # 对齐数据
            aligned_portfolio, aligned_benchmark = portfolio_returns.align(benchmark_returns, join='inner')
            if len(aligned_benchmark) > 0:
                benchmark_cumulative = (1 + aligned_benchmark).cumprod()
                # 重新对齐图表数据
                chart_dates = [date.strftime('%Y-%m-%d') for date in aligned_portfolio.index]
                portfolio_values = (1 + aligned_portfolio).cumprod().tolist()
                benchmark_values = benchmark_cumulative.tolist()
        
        # 转换为JavaScript数组字符串
        dates_js = str(chart_dates)
        portfolio_js = str(portfolio_values)
        benchmark_js = str(benchmark_values) if benchmark_values else "[]"
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>投资组合绩效分析报告</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; border-left: 4px solid #3498db; padding-left: 15px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; margin-bottom: 5px; }}
        .metric-label {{ font-size: 14px; opacity: 0.9; }}
        .summary-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .summary-table th, .summary-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        .summary-table th {{ background-color: #3498db; color: white; }}
        .summary-table tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .positive {{ color: #27ae60; font-weight: bold; }}
        .negative {{ color: #e74c3c; font-weight: bold; }}
        .warning-box {{ background-color: #fff3cd; border: 1px solid #ffeaa7; color: #856404; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .chart-container {{ position: relative; height: 400px; margin: 20px 0; background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .chart-title {{ text-align: center; margin-bottom: 20px; color: #2c3e50; font-size: 18px; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>投资组合绩效分析报告</h1>
        
        <div class="warning-box">
            <strong>注意：</strong>由于pandas版本兼容性问题，本报告为简化版本。完整的QuantStats专业报告因频率代码兼容性问题暂时无法生成。
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{analysis_result.get('total_return', 0):.2f}%</div>
                <div class="metric-label">总收益率</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{analysis_result.get('annual_return', 0):.2f}%</div>
                <div class="metric-label">年化收益率</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{analysis_result.get('annual_volatility', 0):.2f}%</div>
                <div class="metric-label">年化波动率</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{analysis_result.get('max_drawdown', 0):.2f}%</div>
                <div class="metric-label">最大回撤</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{analysis_result.get('sharpe_ratio', 0):.4f}</div>
                <div class="metric-label">夏普比率</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{analysis_result.get('total_trades', 0)}</div>
                <div class="metric-label">总交易次数</div>
            </div>
        </div>
        
        <h2>回测概要</h2>
        <table class="summary-table">
            <tr><th>指标</th><th>数值</th></tr>
            <tr><td>回测期间</td><td>{analysis_result.get('start_date', '').strftime('%Y-%m-%d') if analysis_result.get('start_date') else 'N/A'} 至 {analysis_result.get('end_date', '').strftime('%Y-%m-%d') if analysis_result.get('end_date') else 'N/A'}</td></tr>
            <tr><td>交易天数</td><td>{analysis_result.get('trading_days', 0)}</td></tr>
            <tr><td>初始资金</td><td>{analysis_result.get('start_value', 0):,.2f}</td></tr>
            <tr><td>期末资金</td><td>{analysis_result.get('end_value', 0):,.2f}</td></tr>
            <tr><td>总手续费</td><td>{analysis_result.get('total_commission', 0):,.2f}</td></tr>
            <tr><td>胜率</td><td>{analysis_result.get('win_rate', 0):.2f}%</td></tr>
        </table>
        
        <h2>净值曲线</h2>
        <div class="chart-container">
            <div class="chart-title">投资组合净值走势图</div>
            <canvas id="netValueChart"></canvas>
        </div>
"""

        # 如果有基准比较数据，添加基准比较部分
        if benchmark_returns is not None and 'benchmark_total_return' in analysis_result:
            excess_return = analysis_result.get('excess_return', 0)
            information_ratio = analysis_result.get('information_ratio', 0)
            alpha = analysis_result.get('alpha', 0)
            
            html_content += f"""
        <h2>基准比较分析（vs 沪深300）</h2>
        <table class="summary-table">
            <tr><th>指标</th><th>组合</th><th>基准</th><th>差异</th></tr>
            <tr><td>总收益率</td><td>{analysis_result.get('total_return', 0):.2f}%</td><td>{analysis_result.get('benchmark_total_return', 0):.2f}%</td><td class="{'positive' if excess_return > 0 else 'negative'}">{excess_return:.2f}%</td></tr>
            <tr><td>年化收益率</td><td>{analysis_result.get('annual_return', 0):.2f}%</td><td>{analysis_result.get('benchmark_annual_return', 0):.2f}%</td><td>-</td></tr>
            <tr><td>年化波动率</td><td>{analysis_result.get('annual_volatility', 0):.2f}%</td><td>{analysis_result.get('benchmark_volatility', 0):.2f}%</td><td>-</td></tr>
            <tr><td>最大回撤</td><td>{analysis_result.get('max_drawdown', 0):.2f}%</td><td>{analysis_result.get('benchmark_max_drawdown', 0):.2f}%</td><td>-</td></tr>
        </table>
        
        <table class="summary-table">
            <tr><th>相对指标</th><th>数值</th><th>说明</th></tr>
            <tr><td>信息比率</td><td class="{'positive' if information_ratio > 0 else 'negative'}">{information_ratio:.4f}</td><td>主动管理能力</td></tr>
            <tr><td>Beta系数</td><td>{analysis_result.get('beta', 0):.4f}</td><td>系统性风险</td></tr>
            <tr><td>Alpha系数</td><td class="{'positive' if alpha > 0 else 'negative'}">{alpha:.2f}%</td><td>超额收益</td></tr>
            <tr><td>跟踪误差</td><td>{analysis_result.get('tracking_error', 0):.2f}%</td><td>相对波动性</td></tr>
        </table>
"""

        html_content += f"""
        <h2>报告说明</h2>
        <p>本报告由回测框架自动生成。由于pandas 2.0版本兼容性问题，QuantStats的完整HTML报告暂时无法生成，但基准比较分析和统计指标计算正常工作。</p>
        <p>详细文件：</p>
        <ul>
            <li><strong>performance_charts_*.png</strong> - 绩效图表</li>
            <li><strong>quantstats_stats_*.txt</strong> - 统计摘要</li>
            <li><strong>trades_*.csv</strong> - 交易记录</li>
        </ul>
        
        <p style="text-align: center; margin-top: 40px; color: #7f8c8d;">
            报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </p>
    </div>

    <script>
        const ctx = document.getElementById('netValueChart').getContext('2d');
        
        const portfolioData = {portfolio_js};
        const benchmarkData = {benchmark_js};
        const dateLabels = {dates_js};
        
        const datasets = [
            {{
                label: '投资组合净值',
                data: portfolioData,
                borderColor: '#3498db',
                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.1
            }}
        ];
        
        if (benchmarkData.length > 0) {{
            datasets.push({{
                label: '沪深300基准',
                data: benchmarkData,
                borderColor: '#e74c3c',
                backgroundColor: 'rgba(231, 76, 60, 0.1)',
                borderWidth: 2,
                fill: false,
                tension: 0.1
            }});
        }}
        
        const chartData = {{
            labels: dateLabels,
            datasets: datasets
        }};
        
        const config = {{
            type: 'line',
            data: chartData,
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        display: true,
                        position: 'top'
                    }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                const value = context.parsed.y;
                                const percentage = ((value - 1) * 100).toFixed(2);
                                return context.dataset.label + ': ' + percentage + '%';
                            }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        title: {{
                            display: true,
                            text: '时间'
                        }},
                        ticks: {{
                            maxTicksLimit: 10
                        }}
                    }},
                    y: {{
                        title: {{
                            display: true,
                            text: '累积收益率'
                        }},
                        ticks: {{
                            callback: function(value) {{
                                return ((value - 1) * 100).toFixed(1) + '%';
                            }}
                        }}
                    }}
                }}
            }}
        }};
        
        document.addEventListener('DOMContentLoaded', function() {{
            try {{
                new Chart(ctx, config);
                console.log('图表创建成功');
            }} catch (error) {{
                console.error('图表创建失败:', error);
                document.getElementById('netValueChart').style.display = 'none';
                const errorDiv = document.createElement('div');
                errorDiv.innerHTML = '<p style="text-align: center; color: #e74c3c; padding: 40px;">图表加载失败: ' + error.message + '</p>';
                document.querySelector('.chart-container').appendChild(errorDiv);
            }}
        }});
    </script>
</body>
</html>
"""

        # 写入HTML文件
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
