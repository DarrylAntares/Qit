#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
净值曲线统计分析工具
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
import warnings
import os
warnings.filterwarnings('ignore')

def load_nav_data():
    """加载净值表数据"""
    print("正在加载净值表数据...")
    file_path = 'source/净值表.xlsx'
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")
    
    try:
        df = pd.read_excel(file_path)
        print(f"净值数据形状: {df.shape}")
        print(f"净值数据列: {df.columns.tolist()}")
        
        # 检查必要的列
        required_cols = ['symbol', 'date', 'value']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必要的列: {missing_cols}")
        
        # 转换日期格式
        df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d').dt.date
        print(f"日期范围: {df['date'].min()} 到 {df['date'].max()}")
        print(f"涉及组合数: {df['symbol'].nunique()}")
        
        return df
    except Exception as e:
        print(f"读取净值表时出错: {str(e)}")
        raise

def load_benchmark_data():
    """加载基准指数数据"""
    print("正在加载基准指数数据...")
    file_path = 'benchmark/930950.CSI.xlsx'
    
    if not os.path.exists(file_path):
        print(f"警告: 找不到基准指数文件 {file_path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_excel(file_path)
        print(f"基准数据形状: {df.shape}")
        print(f"基准数据列: {df.columns.tolist()}")
        
        # 转换日期格式
        if 'trade_dt' in df.columns:
            df['trade_dt'] = pd.to_datetime(df['trade_dt']).dt.date
        
        return df
    except Exception as e:
        print(f"读取基准指数数据时出错: {str(e)}")
        return pd.DataFrame()

def load_trade_data():
    """加载交易分析结果数据"""
    print("正在加载交易分析结果数据...")
    file_path = 'intern/trade_analysis_result.xlsx'
    
    if not os.path.exists(file_path):
        print(f"警告: 找不到交易分析结果文件 {file_path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_excel(file_path)
        print(f"交易数据形状: {df.shape}")
        print(f"交易数据列: {df.columns.tolist()}")
        
        # 转换日期格式
        if 'trade_dt' in df.columns:
            df['trade_dt'] = pd.to_datetime(df['trade_dt']).dt.date
        
        return df
    except Exception as e:
        print(f"读取交易分析结果时出错: {str(e)}")
        return pd.DataFrame()

def load_industry_params():
    """加载行业参数数据"""
    print("正在加载行业参数数据...")
    file_path = 'Config/industry_param.xlsx'
    
    if not os.path.exists(file_path):
        print(f"警告: 找不到行业参数文件 {file_path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_excel(file_path)
        print(f"行业参数数据形状: {df.shape}")
        print(f"行业参数数据列: {df.columns.tolist()}")
        
        return df
    except Exception as e:
        print(f"读取行业参数数据时出错: {str(e)}")
        return pd.DataFrame()

def calculate_returns_metrics(nav_df, benchmark_df, symbol):
    """计算收益能力指标"""
    symbol_nav = nav_df[nav_df['symbol'] == symbol].copy()
    symbol_nav = symbol_nav.sort_values('date')
    
    if len(symbol_nav) < 2:
        return {
            'start_date': np.nan,
            'end_date': np.nan,
            'total_return': np.nan,
            'annual_return': np.nan,
            'excess_return': np.nan,
            'positive_months_ratio': np.nan
        }
    
    # 获取起始日期和截止日期
    start_date = symbol_nav['date'].min()
    end_date = symbol_nav['date'].max()
    
    # 计算日收益率
    symbol_nav['daily_return'] = symbol_nav['value'].pct_change()
    
    # 计算累计总收益率
    total_return = (symbol_nav['value'].iloc[-1] / 1) - 1
    
    # 计算年化收益率
    total_days = (end_date - start_date).days
    annual_return = (1 + total_return) ** (365 / total_days) - 1 if total_days > 0 else np.nan
    
    # 计算超额收益率
    excess_return = np.nan
    if not benchmark_df.empty and 'close' in benchmark_df.columns:
        # 合并基准数据
        benchmark_df_copy = benchmark_df.copy()
        benchmark_df_copy = benchmark_df_copy.rename(columns={'trade_dt': 'date'})
        merged = pd.merge(symbol_nav, benchmark_df_copy[['date', 'close']], on='date', how='inner')
        
        if len(merged) > 1:
            merged['benchmark_return'] = merged['close'].pct_change()
            portfolio_total_return = (merged['value'].iloc[-1] / 1 ) - 1
            
            # 找到merged的start_date，获取benchmark_df_copy在start_date前一条数据的close
            merged_start_date = merged['date'].min()
            benchmark_df_copy_sorted = benchmark_df_copy.sort_values('date')
            prev_data = benchmark_df_copy_sorted[benchmark_df_copy_sorted['date'] < merged_start_date]
            
            if not prev_data.empty:
                # 使用start_date前一条数据的close作为分母
                prev_close = prev_data['close'].iloc[-1]
                benchmark_total_return = (merged['close'].iloc[-1] / prev_close) - 1
            else:
                # 如果没有前一条数据，使用原来的逻辑
                benchmark_total_return = (merged['close'].iloc[-1] / merged['close'].iloc[0]) - 1
            
            excess_return = portfolio_total_return - benchmark_total_return
    
    # 计算绝对收益能力（正收益率月份数/总月份数）
    symbol_nav['year_month'] = pd.to_datetime(symbol_nav['date']).dt.to_period('M')
    monthly_returns = symbol_nav.groupby('year_month')['daily_return'].apply(lambda x: (1 + x).prod() - 1)
    positive_months = (monthly_returns > 0).sum()
    total_months = len(monthly_returns)
    positive_months_ratio = positive_months / total_months if total_months > 0 else np.nan
    
    return {
        'start_date': start_date,
        'end_date': end_date,
        'total_return': total_return * 100,  # 转换为百分比
        'annual_return': annual_return * 100,  # 转换为百分比
        'excess_return': excess_return * 100 if not pd.isna(excess_return) else np.nan,
        'positive_months_ratio': positive_months_ratio * 100 if not pd.isna(positive_months_ratio) else np.nan
    }

def calculate_risk_metrics(nav_df, benchmark_df, symbol):
    """计算风险控制能力指标"""
    symbol_nav = nav_df[nav_df['symbol'] == symbol].copy()
    symbol_nav = symbol_nav.sort_values('date')
    
    if len(symbol_nav) < 2:
        return {
            'max_drawdown': np.nan,
            'sharpe_ratio': np.nan,
            'calmar_ratio': np.nan
        }
    
    # 计算日收益率
    symbol_nav['daily_return'] = symbol_nav['value'].pct_change()
    
    # 计算最大回撤
    symbol_nav['cumulative'] = symbol_nav['value'] / 1
    symbol_nav['running_max'] = symbol_nav['cumulative'].expanding().max()
    symbol_nav['drawdown'] = (symbol_nav['cumulative'] - symbol_nav['running_max']) / symbol_nav['running_max']
    max_drawdown = symbol_nav['drawdown'].min() * 100  # 转换为百分比
    
    # 计算夏普比率
    daily_returns = symbol_nav['daily_return'].dropna()
    if len(daily_returns) > 1:
        annual_return = (symbol_nav['value'].iloc[-1] / 1) ** (252 / len(symbol_nav)) - 1
        annual_volatility = daily_returns.std() * (252 ** 0.5)
        sharpe_ratio = (annual_return - 0.02) / annual_volatility if annual_volatility != 0 else np.nan
    else:
        sharpe_ratio = np.nan
    
    # 计算卡玛比率
    #annual_return_for_calmar = calculate_returns_metrics(nav_df, benchmark_df, symbol)['annual_return'] / 100

    calmar_ratio = annual_return / abs(max_drawdown / 100) if max_drawdown != 0 else np.nan
    
    return {
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'calmar_ratio': calmar_ratio
    }

def calculate_efficiency_metrics(trade_df, nav_df, symbol):
    """计算投资效率指标"""
    symbol_trades = trade_df[trade_df['symbol'] == symbol].copy()
    symbol_nav = nav_df[nav_df['symbol'] == symbol].copy()
    
    if symbol_trades.empty or symbol_nav.empty:
        return {
            'turnover_rate': np.nan,
            'win_rate': np.nan,
            'profit_loss_ratio': np.nan
        }
    
    # 计算换手率
    total_vol = symbol_trades['vol'].sum()
    time_span_days = (symbol_nav['date'].max() - symbol_nav['date'].min()).days
    time_span_years = time_span_days / 365 if time_span_days > 0 else 1
    turnover_rate = (total_vol / 2) / time_span_years if time_span_years > 0 else np.nan
    
    # 计算胜率
    sale_trades = symbol_trades[symbol_trades['trade_type'] == 'sale'].copy()
    if not sale_trades.empty:
        winning_trades = sale_trades[sale_trades['yield'] >= 0]
        win_rate = len(winning_trades) / len(sale_trades) * 100
    else:
        win_rate = np.nan
    
    # 计算盈亏比
    if not sale_trades.empty:
        winning_yields = sale_trades[sale_trades['yield'] >= 0]['yield']
        losing_yields = sale_trades[sale_trades['yield'] < 0]['yield']
        
        if len(winning_yields) > 0 and len(losing_yields) > 0:
            avg_win = winning_yields.mean()
            avg_loss = abs(losing_yields.mean())
            profit_loss_ratio = avg_win / avg_loss if avg_loss != 0 else np.nan
        else:
            profit_loss_ratio = np.nan
    else:
        profit_loss_ratio = np.nan
    
    return {
        'turnover_rate': turnover_rate,
        'win_rate': win_rate,
        'profit_loss_ratio': profit_loss_ratio
    }

def calculate_management_metrics(trade_df, symbol):
    """计算组合管理能力指标"""
    symbol_trades = trade_df[trade_df['symbol'] == symbol].copy()
    
    if symbol_trades.empty:
        return {
            'position_concentration': np.nan,
            'disposal_efficiency': np.nan
        }
    
    # 计算持仓集中度
    buy_trades = symbol_trades[symbol_trades['trade_type'] == 'buy']
    if not buy_trades.empty:
        position_concentration = buy_trades['accu_vol'].mean()
    else:
        position_concentration = np.nan
    
    # 计算处置效率
    sale_trades = symbol_trades[symbol_trades['trade_type'] == 'sale'].copy()
    if not sale_trades.empty:
        winning_sales = sale_trades[sale_trades['yield'] >= 0]
        losing_sales = sale_trades[sale_trades['yield'] < 0]
        
        if len(winning_sales) > 0 and len(losing_sales) > 0:
            avg_win_days = winning_sales['holding_days'].mean()
            avg_loss_days = losing_sales['holding_days'].mean()
            disposal_efficiency = avg_win_days / avg_loss_days if avg_loss_days != 0 else np.nan
        else:
            disposal_efficiency = np.nan
    else:
        disposal_efficiency = np.nan
    
    return {
        'position_concentration': position_concentration,
        'disposal_efficiency': disposal_efficiency
    }

def calculate_style_metrics(trade_df, industry_params_df, symbol):
    """计算投资风格指标"""
    symbol_trades = trade_df[trade_df['symbol'] == symbol].copy()
    
    if symbol_trades.empty or industry_params_df.empty:
        return {'growth_score': np.nan}
    
    # 计算成长股得分
    buy_trades = symbol_trades[symbol_trades['trade_type'] == 'buy'].copy()
    if not buy_trades.empty and 'score' in industry_params_df.columns:
        # 创建行业得分映射
        industry_score_map = dict(zip(industry_params_df.iloc[:, 0], industry_params_df['score']))
        
        # 计算加权得分
        buy_trades['industry_score'] = buy_trades['sw'].map(industry_score_map)
        buy_trades = buy_trades.dropna(subset=['industry_score'])
        
        if not buy_trades.empty:
            total_vol = buy_trades['vol'].sum()
            weighted_score = (buy_trades['vol'] * buy_trades['industry_score']).sum() / total_vol if total_vol != 0 else np.nan
        else:
            weighted_score = np.nan
    else:
        weighted_score = np.nan
    
    return {'growth_score': weighted_score}

def calculate_holding_metrics(trade_df, symbol):
    """计算持仓周期指标"""
    symbol_trades = trade_df[trade_df['symbol'] == symbol].copy()
    
    if symbol_trades.empty:
        return {'avg_holding_days': np.nan}
    
    # 计算平均持有天数
    sale_trades = symbol_trades[symbol_trades['trade_type'] == 'sale']
    if not sale_trades.empty:
        avg_holding_days = sale_trades['holding_days'].mean()
    else:
        avg_holding_days = np.nan
    
    return {'avg_holding_days': avg_holding_days}

def calculate_trade_count(trade_df, symbol):
    """计算交易总笔数"""
    symbol_trades = trade_df[trade_df['symbol'] == symbol].copy()
    
    if symbol_trades.empty:
        return {'total_trades': 0}
    
    # 统计买入和卖出的总次数
    buy_count = len(symbol_trades[symbol_trades['trade_type'] == 'buy'])
    sale_count = len(symbol_trades[symbol_trades['trade_type'] == 'sale'])
    total_trades = buy_count + sale_count
    
    return {'total_trades': total_trades}

def analyze_portfolio_performance(nav_df, benchmark_df, trade_df, industry_params_df):
    """分析所有组合的表现"""
    print("开始分析组合表现...")
    
    symbols = nav_df['symbol'].unique()
    results = []
    
    for symbol in symbols:
        print(f"分析组合: {symbol}")
        
        # 计算各项指标
        returns_metrics = calculate_returns_metrics(nav_df, benchmark_df, symbol)
        risk_metrics = calculate_risk_metrics(nav_df, benchmark_df, symbol)
        efficiency_metrics = calculate_efficiency_metrics(trade_df, nav_df, symbol)
        management_metrics = calculate_management_metrics(trade_df, symbol)
        style_metrics = calculate_style_metrics(trade_df, industry_params_df, symbol)
        holding_metrics = calculate_holding_metrics(trade_df, symbol)
        trade_count_metrics = calculate_trade_count(trade_df, symbol)
        
        # 整合结果
        result = {
            'symbol': symbol,
            # 基本信息
            'start_date': returns_metrics['start_date'],
            'end_date': returns_metrics['end_date'],
            'total_return': returns_metrics['total_return'],
            'total_trades': trade_count_metrics['total_trades'],
            # 收益能力
            'annual_return': returns_metrics['annual_return'],
            'excess_return': returns_metrics['excess_return'],
            'positive_months_ratio': returns_metrics['positive_months_ratio'],
            # 风险控制
            'max_drawdown': risk_metrics['max_drawdown'],
            'sharpe_ratio': risk_metrics['sharpe_ratio'],
            'calmar_ratio': risk_metrics['calmar_ratio'],
            # 投资效率
            'turnover_rate': efficiency_metrics['turnover_rate'],
            'win_rate': efficiency_metrics['win_rate'],
            'profit_loss_ratio': efficiency_metrics['profit_loss_ratio'],
            # 组合管理
            'position_concentration': management_metrics['position_concentration'],
            'disposal_efficiency': management_metrics['disposal_efficiency'],
            # 投资风格
            'growth_score': style_metrics['growth_score'],
            # 持仓周期
            'avg_holding_days': holding_metrics['avg_holding_days']
        }
        
        results.append(result)
    
    return pd.DataFrame(results)

def main():
    """主函数"""
    print("=== 净值曲线统计分析开始 ===")
    
    try:
        # 1. 加载数据
        nav_df = load_nav_data()
        benchmark_df = load_benchmark_data()
        trade_df = load_trade_data()
        industry_params_df = load_industry_params()
        
        # 2. 分析组合表现
        results_df = analyze_portfolio_performance(nav_df, benchmark_df, trade_df, industry_params_df)
        
        # 3. 确保intern目录存在
        os.makedirs('intern', exist_ok=True)
        
        # 4. 保存结果
        output_file = 'intern/portfolio_performance_analysis.xlsx'
        results_df.to_excel(output_file, index=False)
        print(f"\n分析结果已保存到: {output_file}")
        print(f"结果数据形状: {results_df.shape}")
        
        # 5. 显示结果摘要
        print("\n=== 分析结果摘要 ===")
        print(f"分析组合数: {len(results_df)}")
        print(f"平均累计总收益率: {results_df['total_return'].mean():.2f}%")
        print(f"平均交易总笔数: {results_df['total_trades'].mean():.0f}笔")
        print(f"平均年化收益率: {results_df['annual_return'].mean():.2f}%")
        print(f"平均最大回撤: {results_df['max_drawdown'].mean():.2f}%")
        print(f"平均夏普比率: {results_df['sharpe_ratio'].mean():.2f}")
        
        # 显示日期范围信息
        if 'start_date' in results_df.columns and 'end_date' in results_df.columns:
            earliest_start = results_df['start_date'].min()
            latest_end = results_df['end_date'].max()
            print(f"组合日期范围: {earliest_start} 到 {latest_end}")
        
        # 显示交易笔数统计
        if 'total_trades' in results_df.columns:
            total_trades_sum = results_df['total_trades'].sum()
            print(f"所有组合交易总笔数: {total_trades_sum}笔")
        
        # 显示前几行结果
        print(f"\n前5行结果预览:")
        print(results_df.head())
        
        print("\n=== 分析完成 ===")
        return results_df
        
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    result = main()