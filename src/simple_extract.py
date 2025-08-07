import json
import csv
from collections import Counter

# 读取JSON文件
with open('/home/user/zedongpeng/workspace/cpp-benchmark-deduplicator/data/data_of_designs_kernels.json', 'r') as f:
    designs = json.load(f)

# 统计各个source的数量
source_counts = Counter()
for design in designs:
    source_name = design.get('source_name', 'Unknown')
    source_counts[source_name] += 1

# 写入CSV文件
with open('selected_designs_summary.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    
    # 写入表头
    writer.writerow(['algo_name', 'source_name'])
    
    # 写入数据
    for design in designs:
        algo_name = design.get('algo_name', 'Unknown')
        source_name = design.get('source_name', 'Unknown')
        writer.writerow([algo_name, source_name])

print("CSV文件已生成完成！")

# 输出各个source的统计信息
print("\n各个source的数量统计：")
print("-" * 30)
for source_name, count in sorted(source_counts.items()):
    print(f"{source_name}: {count}个")

print(f"\n总共有 {len(source_counts)} 个不同的source")
print(f"总共有 {sum(source_counts.values())} 个设计")