import pandas as pd
import numpy as np
from datetime import datetime


# 读取数据
data_path='data/7303_all.csv'
# 提取 7303 或 7304
file_prefix = data_path.split('/')[-1].split('_')[0]

# ===== 第一步: 7303_all.csv -> 7303_in.csv =====
print(f"步骤1: {file_prefix}_all.csv -> {file_prefix}_in.csv")

# 读取原始数据（指定GB2312编码）
df_all = pd.read_csv(data_path, encoding='gb2312')

print(f"原始列名: {list(df_all.columns)}")

# 提取需要的列（根据实际列名调整）
df_in = df_all[['日期', '时间', 'X[m]', 'Y[m]', 'Z[m]', '能量[J]']].copy()

# 重命名列为英文
df_in.columns = ['date', 'time', 'x', 'y', 'z', 'e']

# 转换date格式: 2023/7/5 -> 2023-07-05
df_in['date'] = pd.to_datetime(df_in['date'], format='%Y/%m/%d').dt.strftime('%Y-%m-%d')

# 转换time格式: 2:08:47 -> 02:08:47
def format_time(time_str):
    parts = time_str.split(':')
    # 补齐小时和分钟为两位数
    return ':'.join([part.zfill(2) for part in parts])

df_in['time'] = df_in['time'].apply(format_time)

# 保存中间结果
df_in.to_csv(f'data/{file_prefix}_in.csv', index=False, encoding='utf-8')
print(f"✓ {file_prefix}_in.csv 已保存")

# ===== 第二步: 7303_in.csv -> 7303_xyzet.csv =====
print(f"\n步骤2: {file_prefix}_in.csv -> {file_prefix}_xyzet.csv")

# 合并日期和时间为datetime
df_in['datetime'] = pd.to_datetime(
    df_in['date'] + ' ' + df_in['time'], 
    format='%Y-%m-%d %H:%M:%S'
)

# 计算相对时间戳(秒) - 从最早时间开始计算
time_min = df_in['datetime'].min()
df_in['t'] = (df_in['datetime'] - time_min).dt.total_seconds().astype(int)

# 创建7303_xyzet.csv，只保留x,y,z,e,t五个属性
df_xyzet = df_in[['x', 'y', 'z', 'e', 't']].copy()

# 保存结果
df_xyzet.to_csv(f'data/{file_prefix}_xyzet.csv', index=False)
print(f"✓ {file_prefix}_xyzet.csv 已保存")

# 输出统计信息
print("\n" + "="*60)
print("转换完成统计信息")
print("="*60)
print(f"{file_prefix}_all.csv 原始行数: {len(df_all)}")
print(f"{file_prefix}_in.csv 行数: {len(df_in)}")
print(f"{file_prefix}_xyzet.csv 行数: {len(df_xyzet)}")

print(f"\n{file_prefix}_xyzet.csv 列名: {list(df_xyzet.columns)}")

print(f"\n数据范围统计:")
print(f"  x: [{df_xyzet['x'].min():.2f}, {df_xyzet['x'].max():.2f}]")
print(f"  y: [{df_xyzet['y'].min():.2f}, {df_xyzet['y'].max():.2f}]")
print(f"  z: [{df_xyzet['z'].min():.2f}, {df_xyzet['z'].max():.2f}]")
print(f"  e: [{df_xyzet['e'].min():.6f}, {df_xyzet['e'].max():.6f}]")
print(f"  t: [{df_xyzet['t'].min():.0f}, {df_xyzet['t'].max():.0f}] (秒)")

print(f"\n时间范围:")
print(f"  开始时间: {time_min.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  结束时间: {df_in['datetime'].max().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  总时长: {df_xyzet['t'].max()} 秒 ({df_xyzet['t'].max()/3600:.2f} 小时)")

print("\n前5行数据:")
print(df_xyzet.head())