"""
信号分析器 - 分析T日、T+1日、T+2日的胜率和盈亏比
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class SignalAnalyzer:
    """信号分析器"""
    
    def __init__(self, logger=None, config=None):
        self.logger = logger
        
        # 从配置中获取交易参数
        if config:
            self.trade_price_type = config.get('TRADE_PRICE_TYPE', 'close')
            self.position_value_price_type = config.get('POSITION_VALUE_PRICE_TYPE', 'close')
        else:
            # 默认配置
            self.trade_price_type = 'close'
            self.position_value_price_type = 'close'
        
        if self.logger:
            self.logger.info(f"信号分析器配置:")
            self.logger.info(f"  交易价格类型: {self.trade_price_type}")
            self.logger.info(f"  估值价格类型: {self.position_value_price_type}")
    
    def analyze_signal_performance(self, signal_df: pd.DataFrame, stock_data: Dict[str, pd.DataFrame], 
                                 output_dir: str) -> Dict[str, Any]:
        """
        分析信号绩效
        
        参数:
            signal_df: 信号数据DataFrame
            stock_data: 股票价格数据
            output_dir: 输出目录
            
        返回:
            Dict: 分析结果
        """
        if self.logger:
            self.logger.info("开始信号绩效分析")
        
        # 只分析买入信号
        buy_signals = signal_df[signal_df['trade_type'] == 'buy'].copy()
        
        if len(buy_signals) == 0:
            if self.logger:
                self.logger.warning("没有买入信号，跳过信号分析")
            return {}
        
        # 计算T日、T+1日、T+2日收益率
        results = {}
        for period in ['T', 'T+1', 'T+2']:
            period_results = self._analyze_period_performance(buy_signals, stock_data, period)
            results[period] = period_results
        
        # 生成分析报告和图表
        self._generate_signal_report(results, output_dir)
        
        if self.logger:
            self.logger.info("信号绩效分析完成")
        
        return results
    
    def _analyze_period_performance(self, buy_signals: pd.DataFrame, 
                                  stock_data: Dict[str, pd.DataFrame], 
                                  period: str) -> Dict[str, Any]:
        """分析特定周期的绩效"""
        
        # 计算偏移天数
        offset_days = {'T': 0, 'T+1': 1, 'T+2': 2}[period]
        
        signal_returns = []
        valid_signals = []
        
        for _, signal in buy_signals.iterrows():
            stock_code = signal['ts_code']
            signal_date = pd.to_datetime(signal['trade_date'])
            
            if stock_code not in stock_data:
                continue
                
            stock_df = stock_data[stock_code].copy()
            stock_df.index = pd.to_datetime(stock_df.index)
            
            # 找到信号日期在股票数据中的位置
            try:
                if signal_date not in stock_df.index:
                    # 找到最近的交易日
                    available_dates = stock_df.index[stock_df.index >= signal_date]
                    if len(available_dates) == 0:
                        continue
                    signal_date = available_dates[0]
                
                signal_idx = stock_df.index.get_loc(signal_date)
                
                # 计算目标日期
                target_idx = signal_idx + offset_days
                if target_idx >= len(stock_df):
                    continue
                
                target_date = stock_df.index[target_idx]
                
                # 根据配置确定买入价格和卖出价格
                buy_price = self._get_buy_price(stock_df, signal_idx, target_idx, period)
                sell_price = stock_df.iloc[target_idx]["close"]
                
                if buy_price is None or sell_price is None:
                    continue
                
                if pd.isna(buy_price) or pd.isna(sell_price) or buy_price <= 0:
                    continue
                
                # 计算收益率
                return_rate = (sell_price - buy_price) / buy_price
                
                signal_returns.append(return_rate)
                valid_signals.append({
                    'stock_code': stock_code,
                    'signal_date': signal_date,
                    'target_date': target_date,
                    'return_rate': return_rate,
                    'year': signal_date.year
                })
                
            except Exception as e:
                if self.logger:
                    self.logger.debug(f"处理信号失败 {stock_code} {signal_date}: {e}")
                continue
        
        if len(signal_returns) == 0:
            return {
                'total_signals': 0,
                'win_rate': 0,
                'profit_loss_ratio': 0,
                'avg_return': 0,
                'yearly_stats': {},
                'return_distribution': []
            }
        
        # 计算统计指标
        signal_returns = np.array(signal_returns)
        positive_returns = signal_returns[signal_returns > 0]
        negative_returns = signal_returns[signal_returns < 0]
        
        win_rate = len(positive_returns) / len(signal_returns) * 100
        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = abs(negative_returns.mean()) if len(negative_returns) > 0 else 0
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        avg_return = signal_returns.mean()
        
        # 按年度统计
        yearly_stats = {}
        valid_signals_df = pd.DataFrame(valid_signals)
        
        if len(valid_signals_df) > 0:
            for year in valid_signals_df['year'].unique():
                year_data = valid_signals_df[valid_signals_df['year'] == year]
                year_returns = year_data['return_rate'].values
                
                if len(year_returns) > 0:
                    year_positive = year_returns[year_returns > 0]
                    year_negative = year_returns[year_returns < 0]
                    
                    year_win_rate = len(year_positive) / len(year_returns) * 100
                    year_avg_win = year_positive.mean() if len(year_positive) > 0 else 0
                    year_avg_loss = abs(year_negative.mean()) if len(year_negative) > 0 else 0
                    year_profit_loss_ratio = year_avg_win / year_avg_loss if year_avg_loss > 0 else 0
                    
                    yearly_stats[year] = {
                        'signals': len(year_returns),
                        'win_rate': year_win_rate,
                        'profit_loss_ratio': year_profit_loss_ratio,
                        'avg_return': year_returns.mean()
                    }
        
        return {
            'total_signals': len(signal_returns),
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'avg_return': avg_return,
            'yearly_stats': yearly_stats,
            'return_distribution': signal_returns,
            'valid_signals': valid_signals
        }
    
    def _generate_signal_report(self, results: Dict[str, Any], output_dir: str):
        """生成信号分析报告"""
        
        # 创建HTML报告
        html_content = self._create_signal_html_report(results, output_dir)
        
        # 保存HTML文件
        html_file = os.path.join(output_dir, "signal_analysis_report.html")
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        if self.logger:
            self.logger.info(f"信号分析报告已保存: {html_file}")
    
    def _create_signal_html_report(self, results: Dict[str, Any], output_dir: str) -> str:
        """创建HTML报告"""
        
        html_parts = []
        
        # HTML头部
        html_parts.append("""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>信号分析报告</title>
            <style>
                body { font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
                h1, h2, h3 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
                .summary-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                .summary-table th, .summary-table td { border: 1px solid #ddd; padding: 12px; text-align: center; }
                .summary-table th { background-color: #4CAF50; color: white; }
                .summary-table tr:nth-child(even) { background-color: #f2f2f2; }
                .chart-container { margin: 20px 0; text-align: center; }
                .chart-container img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }
                .period-section { margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
                .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
                .stat-card { background-color: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; }
                .stat-value { font-size: 24px; font-weight: bold; color: #4CAF50; }
                .stat-label { font-size: 14px; color: #666; margin-top: 5px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 信号分析报告</h1>
                <p><strong>生成时间:</strong> """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
        """)
        
        # 总体统计表格
        html_parts.append("""
                <h2>📈 总体统计概览</h2>
                <table class="summary-table">
                    <thead>
                        <tr>
                            <th>周期</th>
                            <th>信号数量</th>
                            <th>胜率 (%)</th>
                            <th>盈亏比</th>
                            <th>平均收益率 (%)</th>
                        </tr>
                    </thead>
                    <tbody>
        """)
        
        for period in ['T', 'T+1', 'T+2']:
            if period in results:
                data = results[period]
                html_parts.append(f"""
                        <tr>
                            <td><strong>{period}日</strong></td>
                            <td>{data['total_signals']}</td>
                            <td>{data['win_rate']:.2f}%</td>
                            <td>{data['profit_loss_ratio']:.4f}</td>
                            <td>{data['avg_return']*100:.2f}%</td>
                        </tr>
                """)
        
        html_parts.append("""
                    </tbody>
                </table>
        """)
        
        # 为每个周期生成详细分析
        for period in ['T', 'T+1', 'T+2']:
            if period not in results or results[period]['total_signals'] == 0:
                continue
                
            data = results[period]
            html_parts.append(f"""
                <div class="period-section">
                    <h2>📊 {period}日详细分析</h2>
                    
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-value">{data['total_signals']}</div>
                            <div class="stat-label">信号总数</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{data['win_rate']:.2f}%</div>
                            <div class="stat-label">胜率</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{data['profit_loss_ratio']:.4f}</div>
                            <div class="stat-label">盈亏比</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{data['avg_return']*100:.2f}%</div>
                            <div class="stat-label">平均收益率</div>
                        </div>
                    </div>
            """)
            
            # 年度统计表格
            if data['yearly_stats']:
                html_parts.append("""
                    <h3>📅 年度统计</h3>
                    <table class="summary-table">
                        <thead>
                            <tr>
                                <th>年份</th>
                                <th>信号数量</th>
                                <th>胜率 (%)</th>
                                <th>盈亏比</th>
                                <th>平均收益率 (%)</th>
                            </tr>
                        </thead>
                        <tbody>
                """)
                
                for year in sorted(data['yearly_stats'].keys()):
                    year_data = data['yearly_stats'][year]
                    html_parts.append(f"""
                            <tr>
                                <td>{year}</td>
                                <td>{year_data['signals']}</td>
                                <td>{year_data['win_rate']:.2f}%</td>
                                <td>{year_data['profit_loss_ratio']:.4f}</td>
                                <td>{year_data['avg_return']*100:.2f}%</td>
                            </tr>
                    """)
                
                html_parts.append("""
                        </tbody>
                    </table>
                """)
            
            # 生成收益率分布直方图
            if len(data['return_distribution']) > 0:
                chart_file = self._create_return_distribution_chart(
                    data['return_distribution'], period, output_dir
                )
                if chart_file:
                    chart_name = os.path.basename(chart_file)
                    html_parts.append(f"""
                    <h3>📊 {period}日收益率分布</h3>
                    <div class="chart-container">
                        <img src="{chart_name}" alt="{period}日收益率分布直方图">
                    </div>
                    """)
            
            html_parts.append("</div>")
        
        # HTML尾部
        html_parts.append("""
            </div>
        </body>
        </html>
        """)
        
        return "".join(html_parts)
    
    def _create_return_distribution_chart(self, returns: np.ndarray, period: str, output_dir: str) -> str:
        """创建收益率分布直方图"""
        
        try:
            plt.figure(figsize=(10, 6))
            
            # 创建直方图
            n_bins = min(50, max(10, len(returns) // 10))
            plt.hist(returns * 100, bins=n_bins, alpha=0.7, color='skyblue', edgecolor='black')
            
            # 添加统计线
            mean_return = returns.mean() * 100
            plt.axvline(mean_return, color='red', linestyle='--', linewidth=2, label=f'平均收益率: {mean_return:.2f}%')
            plt.axvline(0, color='gray', linestyle='-', alpha=0.5, label='盈亏平衡线')
            
            plt.title(f'{period}日收益率分布直方图', fontsize=16, fontweight='bold')
            plt.xlabel('收益率 (%)', fontsize=12)
            plt.ylabel('频次', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 添加统计信息
            positive_pct = (returns > 0).sum() / len(returns) * 100
            plt.text(0.02, 0.98, f'正收益比例: {positive_pct:.1f}%\n样本数量: {len(returns)}', 
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            # 保存图片
            chart_file = os.path.join(output_dir, f"return_distribution_{period}.png")
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_file
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"创建{period}日收益率分布图失败: {e}")
            plt.close()
            return None
    
    def _get_buy_price(self, stock_df: pd.DataFrame, signal_idx: int, target_idx: int, period: str) -> float:
        """
        根据配置获取买入价格
        
        参数:
            stock_df: 股票数据
            signal_idx: 信号日索引
            target_idx: 目标日索引
            period: 分析周期 ('T', 'T+1', 'T+2')
            
        返回:
            float: 买入价格，失败返回None
        """
        try:
            # 买入日为信号日，无需再调整
            buy_idx = signal_idx
            
            # 根据交易价格类型获取买入价格
            buy_price = stock_df.iloc[buy_idx][self.trade_price_type]
            
            if pd.isna(buy_price) or buy_price <= 0:
                return None
            
            return float(buy_price)
            
        except (IndexError, KeyError, ValueError):
            return None
    
