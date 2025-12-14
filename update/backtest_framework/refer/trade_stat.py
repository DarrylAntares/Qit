import pandas as pd
import numpy as np
from datetime import datetime, date
import warnings
import os
warnings.filterwarnings('ignore')

def load_trade_data():
    """加载交易明细表数据"""
    print("正在加载交易明细表...")
    file_path = 'source/交易明细表.xlsx'
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")
    
    try:
        df = pd.read_excel(file_path)
        print(f"原始数据形状: {df.shape}")
        print(f"数据列: {df.columns.tolist()}")
        
        # 显示前几行数据以便调试
        print("\n前5行数据预览:")
        print(df.head())
        
        return df
    except Exception as e:
        print(f"读取交易明细表时出错: {str(e)}")
        raise

def load_industry_data():
    """加载行业分类数据"""
    print("正在加载行业分类数据...")
    file_path = 'swindustry/sw.xlsx'
    
    if not os.path.exists(file_path):
        print(f"警告: 找不到行业分类文件 {file_path}，将跳过行业信息")
        return pd.DataFrame()
    
    try:
        industry_df = pd.read_excel(file_path)
        print(f"行业数据形状: {industry_df.shape}")
        print(f"行业数据列: {industry_df.columns.tolist()}")
        
        # 显示前几行数据
        print("\n行业数据前5行预览:")
        print(industry_df.head())
        
        return industry_df
    except Exception as e:
        print(f"读取行业数据时出错: {str(e)}")
        return pd.DataFrame()

def preprocess_data(df):
    """数据预处理"""
    print("开始数据预处理...")
    
    # 检查必要的列是否存在
    required_cols = ['status', 'matched_at', 'stock_symbol', 'stock_market', 'symbol', 'weight']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要的列: {missing_cols}")
    
    # 1. 过滤status=1的记录
    print(f"原始数据中status的唯一值: {df['status'].unique()}")
    df_filtered = df[df['status'] == 1].copy()
    print(f"过滤后数据形状: {df_filtered.shape}")
    
    if len(df_filtered) == 0:
        raise ValueError("过滤后没有数据，请检查status列的值")
    
    # 2. 创建trade_dt列 - 将matched_at转换为日期格式
    try:
        df_filtered['trade_dt'] = pd.to_datetime(df_filtered['matched_at']).dt.date
        print(f"日期范围: {df_filtered['trade_dt'].min()} 到 {df_filtered['trade_dt'].max()}")
    except Exception as e:
        print(f"日期转换出错: {str(e)}")
        raise
    
    # 3. 创建sec_code列 - 拼接stock_symbol和stock_market，股票代码用0填充至6位
    df_filtered['sec_code'] = df_filtered['stock_symbol'].astype(str).str.zfill(6) + '.' + df_filtered['stock_market'].astype(str)
    print(f"生成了 {df_filtered['sec_code'].nunique()} 个唯一证券代码")
    
    # 显示一些统计信息
    print(f"涉及组合数: {df_filtered['symbol'].nunique()}")
    print(f"交易日期数: {df_filtered['trade_dt'].nunique()}")
    
    print("数据预处理完成")
    return df_filtered

def analyze_trades_by_symbol(df, industry_df):
    """按symbol分组进行交易分析"""
    print("开始按组合进行交易分析...")
    
    # 创建行业映射字典
    industry_map = dict(zip(industry_df.iloc[:, 0], industry_df['sw'])) if 'sw' in industry_df.columns else {}
    
    all_trades = []
    symbols = df['symbol'].unique()
    
    for symbol in symbols:
        print(f"处理组合: {symbol}")
        symbol_df = df[df['symbol'] == symbol].copy()
        
        # 按日期排序
        symbol_df = symbol_df.sort_values(['trade_dt', 'sec_code'])
        
        # 获取所有交易日期
        trade_dates = sorted(symbol_df['trade_dt'].unique())
        
        # 存储每个证券的历史买入记录
        buy_records = {}
        
        for trade_date in trade_dates:
            daily_data = symbol_df[symbol_df['trade_dt'] == trade_date]
            
            for _, row in daily_data.iterrows():
                sec_code = row['sec_code']
                current_weight = row['weight']
                prev_weight = row.get('prev_weight_adjusted', 0)
                
                # 判断交易类型
                if current_weight > prev_weight:
                    # 买入交易
                    trade_record = {
                        'symbol': symbol,
                        'trade_dt': trade_date,
                        'sec_code': sec_code,
                        'stock_name': row.get('stock_name', ''),
                        'trade_type': 'buy',
                        'vol': current_weight - prev_weight,
                        'accu_vol': current_weight,  # 添加累计权重
                        'yield': np.nan,
                        'holding_days': np.nan,
                        'sw': industry_map.get(sec_code, np.nan)  # 添加行业信息
                    }
                    
                    all_trades.append(trade_record)
                    
                    # 记录买入信息用于后续卖出时计算持有期
                    if sec_code not in buy_records:
                        buy_records[sec_code] = []
                    buy_records[sec_code].append({
                        'trade_dt': trade_date,
                        'weight': current_weight - prev_weight,
                        'price': row.get('price', np.nan)
                    })
                
                elif current_weight < prev_weight:
                    # 卖出交易
                    vol_sold = prev_weight - current_weight
                    
                    # 计算收益率 - 转换为百分比
                    current_price = row.get('price', np.nan)
                    prev_price = row.get('prev_price', np.nan)
                    yield_rate = ((current_price / prev_price - 1) * 100) if (pd.notna(current_price) and pd.notna(prev_price) and prev_price != 0) else np.nan
                    
                    # 计算持有期天数
                    holding_days = np.nan
                    if sec_code in buy_records and buy_records[sec_code]:
                        # 找到最近的买入记录
                        recent_buy = None
                        for buy_record in reversed(buy_records[sec_code]):
                            if buy_record['trade_dt'] <= trade_date:
                                recent_buy = buy_record
                                break
                        
                        if recent_buy:
                            holding_days = (trade_date - recent_buy['trade_dt']).days
                    
                    trade_record = {
                        'symbol': symbol,
                        'trade_dt': trade_date,
                        'sec_code': sec_code,
                        'stock_name': row.get('stock_name', ''),
                        'trade_type': 'sale',
                        'vol': vol_sold,
                        'accu_vol': current_weight,  # 添加累计权重
                        'yield': yield_rate,
                        'holding_days': holding_days,
                        'sw': industry_map.get(sec_code, np.nan)  # 添加行业信息
                    }
                    
                    all_trades.append(trade_record)
    
    return pd.DataFrame(all_trades)

def main():
    """主函数"""
    print("=== 交易明细统计分析开始 ===")
    
    try:
        # 1. 加载数据
        trade_df = load_trade_data()
        industry_df = load_industry_data()
        
        # 2. 数据预处理
        processed_df = preprocess_data(trade_df)
        
        # 3. 交易分析
        result_df = analyze_trades_by_symbol(processed_df, industry_df)
        
        # 4. 数据质量检查
        if len(result_df) == 0:
            print("警告: 没有生成任何交易记录，请检查数据和逻辑")
            return pd.DataFrame()
        
        # 5. 保存结果
        output_file = 'intern/trade_analysis_result.xlsx'
        
        # 确保输出列的顺序，包含累计权重和行业信息
        output_columns = ['symbol', 'trade_dt', 'sec_code', 'stock_name', 'trade_type', 'vol', 'accu_vol', 'yield', 'holding_days', 'sw']
        result_df = result_df.reindex(columns=output_columns)
        
        result_df.to_excel(output_file, index=False)
        print(f"\n分析结果已保存到: {output_file}")
        print(f"结果数据形状: {result_df.shape}")
        
        # 6. 显示结果摘要
        print("\n=== 分析结果摘要 ===")
        print(f"总交易记录数: {len(result_df)}")
        
        if 'trade_type' in result_df.columns:
            buy_count = len(result_df[result_df['trade_type'] == 'buy'])
            sale_count = len(result_df[result_df['trade_type'] == 'sale'])
            print(f"买入交易数: {buy_count}")
            print(f"卖出交易数: {sale_count}")
        
        print(f"涉及组合数: {result_df['symbol'].nunique()}")
        print(f"涉及证券数: {result_df['sec_code'].nunique()}")
        
        # 显示日期范围
        if 'trade_dt' in result_df.columns:
            print(f"交易日期范围: {result_df['trade_dt'].min()} 到 {result_df['trade_dt'].max()}")
        
        # 显示前几行结果
        print(f"\n前5行结果预览:")
        print(result_df.head())
        
        print("\n=== 分析完成 ===")
        return result_df
        
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    result = main()