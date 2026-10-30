import os
import datetime
import pandas as pd
import numpy as np

# 数据路径
data_path = 'data/6306_all.csv'
file_prefix = os.path.basename(data_path).split('_')[0]

# ===== 第一步: 6306_all.csv -> 6306_in.csv =====
print(f"步骤1: {file_prefix}_all.csv -> {file_prefix}_in.csv")

def load_data(file_path):
    """
    自适应读取数据:
    6306_all.csv 底层实际为 xlsx 二进制格式，若直接使用 read_csv 会报错。
    此处优先根据文件头特征或 read_excel 读取，兼顾常规 csv。
    """
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)
        if header.startswith(b'PK\x03\x04'):  # Zip/XLSX 文件头标识
            return pd.read_excel(file_path)
    except Exception:
        pass

    for enc in ['utf-8', 'gb2312', 'gbk', 'gb18030']:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except Exception:
            continue
    return pd.read_excel(file_path)

df_all = load_data(data_path)
print(f"原始列名: {list(df_all.columns)}")

# 兼容 6306（波兰 SOS 系统导出列名）与 7303（中文列名）
col_map = {
    'Data': 'date', 'CZAS': 'time', 'X': 'x', 'Y': 'y', 'Z': 'z', 'ENERGIA': 'e',
    '日期': 'date', '时间': 'time', 'X[m]': 'x', 'Y[m]': 'y', 'Z[m]': 'z', '能量[J]': 'e'
}

# 提取并重命名列
selected_cols = [c for c in df_all.columns if c in col_map]
df_in = df_all[selected_cols].copy()
df_in.rename(columns=col_map, inplace=True)

# 确保目标列存在且排序一致
df_in = df_in[['date', 'time', 'x', 'y', 'z', 'e']]

# 格式化日期为 YYYY-MM-DD
df_in['date'] = pd.to_datetime(df_in['date']).dt.strftime('%Y-%m-%d')

# 格式化时间为 HH:MM:SS（兼容 datetime.time 对象与时间字符串，并自动补齐两位）
def format_time(val):
    if pd.isna(val):
        return '00:00:00'
    if isinstance(val, (datetime.time, datetime.datetime)):
        return val.strftime('%H:%M:%S')
    val_str = str(val).strip()
    parts = val_str.split(':')
    if len(parts) == 2:
        parts.append('00')
    return ':'.join([p.zfill(2) for p in parts[:3]])

df_in['time'] = df_in['time'].apply(format_time)

# 保存 6306_in.csv
output_in_path = f'data/{file_prefix}_in.csv'
df_in.to_csv(output_in_path, index=False, encoding='utf-8')
print(f"✓ {file_prefix}_in.csv 已保存 ({len(df_in)} 行)")

# ===== 第二步: 6306_in.csv -> 6306_xyzet.csv =====
print(f"\n步骤2: {file_prefix}_in.csv -> {file_prefix}_xyzet.csv")

# 合并日期和时间解析为统一时间戳
df_in['datetime'] = pd.to_datetime(
    df_in['date'] + ' ' + df_in['time'], 
    format='%Y-%m-%d %H:%M:%S'
)

# 计算相对于第一条记录的时间戳 (秒)
time_min = df_in['datetime'].min()
df_in['t'] = (df_in['datetime'] - time_min).dt.total_seconds().astype(int)

# 保留 x, y, z, e, t 五个属性
df_xyzet = df_in[['x', 'y', 'z', 'e', 't']].copy()

# 保存 6306_xyzet.csv
output_xyzet_path = f'data/{file_prefix}_xyzet.csv'
df_xyzet.to_csv(output_xyzet_path, index=False)
print(f"✓ {file_prefix}_xyzet.csv 已保存 ({len(df_xyzet)} 行)")

# ===== 统计信息输出 =====
print("\n" + "=" * 60)
print(f"{file_prefix} 转换完成统计信息")
print("=" * 60)
print(f"{file_prefix}_all.csv 原始行数: {len(df_all)}")
print(f"{file_prefix}_in.csv 行数: {len(df_in)}")
print(f"{file_prefix}_xyzet.csv 行数: {len(df_xyzet)}")
print(f"\n{file_prefix}_xyzet.csv 列名: {list(df_xyzet.columns)}")

print("\n数据范围统计:")
print(f"  x: [{df_xyzet['x'].min():.2f}, {df_xyzet['x'].max():.2f}]")
print(f"  y: [{df_xyzet['y'].min():.2f}, {df_xyzet['y'].max():.2f}]")
print(f"  z: [{df_xyzet['z'].min():.2f}, {df_xyzet['z'].max():.2f}]")
print(f"  e: [{df_xyzet['e'].min():.6f}, {df_xyzet['e'].max():.6f}]")
print(f"  t: [{df_xyzet['t'].min():.0f}, {df_xyzet['t'].max():.0f}] (秒)")

print("\n时间范围:")
print(f"  开始时间: {time_min.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  结束时间: {df_in['datetime'].max().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  总时长: {df_xyzet['t'].max()} 秒 ({df_xyzet['t'].max()/86400:.2f} 天 / {df_xyzet['t'].max()/3600:.2f} 小时)")

print("\n前5行数据:")
print(df_xyzet.head())