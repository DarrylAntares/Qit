def convert_stock_code_format(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            # 分割每行的列
            parts = line.strip().split('\t')
            
            if len(parts) >= 1:
                # 转换证券代码格式
                old_code = parts[0]
                if old_code.startswith('SH'):
                    new_code = old_code[2:] + '.SH'
                elif old_code.startswith('SZ'):
                    new_code = old_code[2:] + '.SZ'
                else:
                    new_code = old_code  # 保持原样（如果有其他格式）
                
                # 替换第一列为新格式
                parts[0] = new_code
                
                # 重新组合行并写入输出文件
                new_line = '\t'.join(parts)
                outfile.write(new_line + '\n')
            else:
                # 如果行没有内容，直接写入
                outfile.write(line)

# 使用示例
input_filename = 'dataset/indice_weight/csi300.txt'
output_filename = 'dataset/indice_weight/hs300.txt'

convert_stock_code_format(input_filename, output_filename)
print(f"转换完成！输出文件: {output_filename}")