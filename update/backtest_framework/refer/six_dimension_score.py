#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
六维评分表生成工具
基于portfolio_performance_analysis.xlsx生成组合的六维评分
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

def load_performance_data():
    """加载组合表现分析数据"""
    print("正在加载组合表现分析数据...")
    file_path = 'intern/portfolio_performance_analysis.xlsx'
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")
    
    try:
        df = pd.read_excel(file_path)
        print(f"数据形状: {df.shape}")
        print(f"数据列: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"读取数据时出错: {str(e)}")
        raise

def calculate_percentile_score(series, reverse=False, default_value=0.5):
    """
    计算分位数得分
    
    Parameters:
    series: 数据序列
    reverse: 是否反向计算（对于负向指标如最大回撤、换手率等）
    default_value: 缺失值的默认填充值
    """
    # 处理缺失值
    series_clean = series.fillna(default_value)
    
    if reverse:
        # 对于负向指标，先取负数再计算分位数
        series_clean = -series_clean
    
    # 计算分位数排名（0-1之间）
    percentile_scores = series_clean.rank(pct=True)
    
    return percentile_scores

def calculate_six_dimension_scores(df):
    """计算六维评分"""
    print("开始计算六维评分...")
    
    result_df = df[['symbol', 'start_date', 'end_date']].copy()
    
    # 1. 收益能力维度
    print("计算收益能力维度...")
    annual_return_score = calculate_percentile_score(df['annual_return'])
    excess_return_score = calculate_percentile_score(df['excess_return'])
    positive_months_score = calculate_percentile_score(df['positive_months_ratio'])
    
    result_df['收益能力'] = (annual_return_score + excess_return_score + positive_months_score) / 3
    
    # 2. 风险管理能力维度
    print("计算风险管理能力维度...")
    max_drawdown_score = calculate_percentile_score(df['max_drawdown'], reverse=True)  # 最大回撤取负数
    sharpe_score = calculate_percentile_score(df['sharpe_ratio'])
    calmar_score = calculate_percentile_score(df['calmar_ratio'])
    
    result_df['风险管理能力'] = (max_drawdown_score + sharpe_score + calmar_score) / 3
    
    # 3. 投资效率维度
    print("计算投资效率维度...")
    turnover_score = calculate_percentile_score(df['turnover_rate'], reverse=True)  # 换手率取负数
    win_rate_score = calculate_percentile_score(df['win_rate'])
    profit_loss_score = calculate_percentile_score(df['profit_loss_ratio'])
    
    result_df['投资效率'] = (turnover_score + win_rate_score + profit_loss_score) / 3
    
    # 4. 组合管理能力维度
    print("计算组合管理能力维度...")
    concentration_score = calculate_percentile_score(df['position_concentration'], reverse=True)  # 持仓集中度取负数
    disposal_score = calculate_percentile_score(df['disposal_efficiency'])
    
    result_df['组合管理能力'] = (concentration_score + disposal_score) / 2
    
    # 5. 投资风格维度
    print("计算投资风格维度...")
    result_df['投资风格'] = calculate_percentile_score(df['growth_score'])
    
    # 6. 持仓风格维度
    print("计算持仓风格维度...")
    result_df['持仓风格'] = calculate_percentile_score(df['avg_holding_days'])
    
    return result_df

def generate_summary_statistics(df):
    """生成汇总统计信息"""
    print("\n=== 六维评分汇总统计 ===")
    
    dimensions = ['收益能力', '风险管理能力', '投资效率', '组合管理能力', '投资风格', '持仓风格']
    
    for dim in dimensions:
        if dim in df.columns:
            mean_score = df[dim].mean()
            std_score = df[dim].std()
            min_score = df[dim].min()
            max_score = df[dim].max()
            print(f"{dim}: 均值={mean_score:.3f}, 标准差={std_score:.3f}, 最小值={min_score:.3f}, 最大值={max_score:.3f}")
    
    # 显示各维度相关性
    print(f"\n各维度相关性矩阵:")
    correlation_matrix = df[dimensions].corr()
    print(correlation_matrix.round(3))

def main():
    """主函数"""
    print("=== 六维评分表生成开始 ===")
    
    try:
        # 1. 加载数据
        performance_df = load_performance_data()
        
        # 2. 计算六维评分
        six_dim_df = calculate_six_dimension_scores(performance_df)
        
        # 3. 确保intern目录存在
        os.makedirs('intern', exist_ok=True)
        
        # 4. 保存结果
        output_file = 'intern/six_dimension_scores.xlsx'
        six_dim_df.to_excel(output_file, index=False)
        print(f"\n六维评分结果已保存到: {output_file}")
        print(f"结果数据形状: {six_dim_df.shape}")
        
        # 5. 生成汇总统计
        generate_summary_statistics(six_dim_df)
        
        # 6. 显示前几行结果
        print(f"\n前5行结果预览:")
        print(six_dim_df.head())
        
        # 7. 显示各维度得分分布
        print(f"\n各维度得分分布:")
        dimensions = ['收益能力', '风险管理能力', '投资效率', '组合管理能力', '投资风格', '持仓风格']
        for dim in dimensions:
            if dim in six_dim_df.columns:
                print(f"{dim}: {six_dim_df[dim].describe().round(3).to_dict()}")
        
        print("\n=== 六维评分表生成完成 ===")
        return six_dim_df
        
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    result = main()
