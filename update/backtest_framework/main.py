"""
回测框架主程序入口
"""
import os
import sys
from datetime import datetime

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from backtest_framework import BacktestFramework
from config.config import Config

def run_hs300_equal_weight_backtest():
    """运行沪深300等权重策略回测"""
    
    # 初始化输出目录
    output_dir = Config.initialize_output_dir()
    
    print("=" * 80)
    print("五层架构回测框架 - 沪深300等权重策略")
    print("=" * 80)
    
    # 显示输出目录信息
    print(f"\n输出目录: {output_dir}")
    print(f"日志文件: {Config.get_log_file()}")
    
    # 创建回测框架实例
    framework = BacktestFramework()
    
    # 显示框架信息
    framework_info = framework.get_framework_info()
    print(f"\n框架名称: {framework_info['framework_name']}")
    print(f"版本: {framework_info['version']}")
    print("\n框架层级:")
    for layer, description in framework_info['layers'].items():
        print(f"  - {description}")
    
    print(f"\n回测配置:")
    print(f"  - 回测期间: {Config.START_DATE.strftime('%Y-%m-%d')} 到 {Config.END_DATE.strftime('%Y-%m-%d')}")
    print(f"  - 初始资金: {Config.INITIAL_CASH:,}")
    print(f"  - 调仓频率: {Config.REBALANCE_FREQ}")
    print(f"  - 手续费率: {Config.COMMISSION:.3f}")
    print(f"  - 最大股票数: {Config.MAX_STOCKS}")
    
    try:
        # 运行回测
        print(f"\n开始运行回测...")
        start_time = datetime.now()
        
        results = framework.run_backtest()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n回测完成！耗时: {duration:.2f} 秒")
        
        # 显示策略信息
        strategy_info = results['strategy_info']
        print(f"\n策略信息:")
        print(f"  - 策略名称: {strategy_info['strategy_name']}")
        print(f"  - 策略类型: {strategy_info['strategy_type']}")
        print(f"  - 描述: {strategy_info['description']}")
        
        # 显示关键绩效指标
        analysis = results['analysis_result']
        print(f"\n关键绩效指标:")
        print(f"  - 总收益率: {analysis['total_return']:.2f}%")
        print(f"  - 年化收益率: {analysis['annual_return']:.2f}%")
        print(f"  - 年化波动率: {analysis['annual_volatility']:.2f}%")
        print(f"  - 最大回撤: {analysis['max_drawdown']:.2f}%")
        print(f"  - 夏普比率: {analysis['sharpe_ratio']:.4f}")
        print(f"  - 总交易次数: {analysis['total_trades']}")
        
        print(f"\n详细结果已保存到: {Config.get_output_dir()}")
        print(f"  - 日志文件: backtest.log")
        print(f"  - 绩效指标: performance_metrics_*.csv")
        print(f"  - 净值序列: portfolio_values_*.csv")
        print(f"  - 交易记录: trades_*.csv")
        print(f"  - 持仓记录: positions_*.csv")
        print(f"  - 绩效图表: performance_charts_*.png")
        print(f"  - 专业报告: quantstats_report_*.html")
        print(f"  - 统计摘要: quantstats_stats_*.txt")
        print("=" * 80)
        
        return results
        
    except Exception as e:
        print(f"\n回测过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数"""
    # 运行沪深300等权重策略回测
    results = run_hs300_equal_weight_backtest()
    
    if results:
        print("\n回测成功完成！")
    else:
        print("\n回测失败！")

if __name__ == '__main__':
    main()
