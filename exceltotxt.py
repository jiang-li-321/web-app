import pandas as pd

file_path = r'C:\Users\ljian\Desktop\116.xlsx'
df = pd.read_excel(file_path, usecols=['A', 'B'])
output_file_path = r'C:\Users\ljian\Desktop\output.txt'
with open(output_file_path, 'w', newline='', encoding='utf-8') as f:
    for index, row in df.iterrows():
        line = f"{row['A']},{row['B']}\n"
        f.write(line)


print(f"数据已成功导出到 {output_file_path}")