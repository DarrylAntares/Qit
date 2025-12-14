import pandas as pd


performance_data = pd.read_excel("intern/portfolio_performance_analysis.xlsx")
portfolio_data = pd.read_excel("source/投顾组合映射.xlsx")

performance_data = performance_data.merge(portfolio_data, on="symbol", how="left")

data = performance_data.copy()

col = ["consultant","symbol","symbol_name","start_date",'end_date','total_return','annual_return','excess_return','positive_months_ratio','max_drawdown',
        'sharpe_ratio','turnover_rate','win_rate','profit_loss_ratio','growth_score','avg_holding_days']

data = data[col]
# 将start_date和end_date转换为datetime类型，再转为字符串yyyy-mm-dd
for i in col[3:5]:
    data[i] = pd.to_datetime(data[i]).dt.strftime('%Y-%m-%d')

# 将total_return, annual_return, excess_return, max_drawdown, sharpe_ratio, turnover_rate, win_rate, profit_loss_ratio, growth_score, avg_holding_days转换为float类型,保留两位小数
for i in col[5:]:
    data[i] = data[i].astype(float).round(2)

# 导出为csv文件
data.to_excel("portfolio_performance_analysis_clean.xlsx", index=False)
