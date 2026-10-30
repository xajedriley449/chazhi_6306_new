import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# 1. 字体与全局设置 (满足硕士论文标准：中文宋体，英文和数字 Times New Roman)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['SimSun'] # 全局默认使用宋体
plt.rcParams['mathtext.fontset'] = 'stix' # 数学公式和数字使用标准衬线体 (类似 Times New Roman)
plt.rcParams['axes.formatter.use_mathtext'] = True # 强制坐标轴刻度使用 mathtext
plt.rcParams['axes.unicode_minus'] = False

size = 10
plt.rcParams.update({
    'svg.fonttype': 'path',
    'font.size': 14+size,
    'axes.titlesize': 16+size,
    'axes.labelsize': 15+size,
    'xtick.labelsize': 13+size,
    'ytick.labelsize': 13+size,
})

# 2. 读取与预处理数据
data_path = 'data/7303_all.csv'
file_prefix = data_path.split('/')[-1].split('_')[0]

# 读取原始数据
df = pd.read_csv(data_path, encoding='gb2312')

# 提取需要的列并重命名
df = df[['日期', '时间', '能量[J]']].copy()
df.columns = ['date', 'time', 'energy']

# 转换日期格式并设置为 datetime 类型
df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d')

# 3. 按天聚合统计
# 统计每天的频次 (count) 和总能量 (sum)
daily_stats = df.groupby('date').agg(
    frequency=('energy', 'count'),
    total_energy=('energy', 'sum')
).reset_index()

# 4. 绘图
fig, ax1 = plt.subplots(figsize=(14, 7))

# 绘制频次 (柱状图，左 Y 轴)
color1 = '#1f77b4' # 蓝色
ax1.bar(daily_stats['date'], daily_stats['frequency'], color=color1, alpha=0.7, label='日频次')
ax1.set_xlabel('日期 (Date)')
ax1.set_ylabel('微震频次 (次)', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)

# 设置 X 轴日期格式
ax1.xaxis.set_major_locator(mdates.MonthLocator()) # 按月显示主刻度
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 创建共享 X 轴的右 Y 轴
ax2 = ax1.twinx()

# 绘制总能量 (折线图，右 Y 轴)
color2 = '#d62728' # 红色
# 为了图表美观，能量通常取对数或缩放，这里我们直接绘制原始总能量，但使用科学计数法
ax2.plot(daily_stats['date'], daily_stats['total_energy'], color=color2, marker='o', markersize=4, linewidth=2, label='日总能量')
ax2.set_ylabel('总释放能量 (J)', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)
# 强制右侧 Y 轴也使用 mathtext
ax2.yaxis.get_offset_text().set_fontfamily('serif')
ax2.yaxis.get_offset_text().set_fontname('SimSun')

# 添加图例 (合并两个轴的图例)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

#plt.title(f'{file_prefix} 工作面微震事件频次与能量释放时序演化')
plt.tight_layout()

# 5. 保存图片
output_dir = 'data/img/211'
os.makedirs(output_dir, exist_ok=True)

svg_path = os.path.join(output_dir, f'{file_prefix}_time_series.svg')
png_path = os.path.join(output_dir, f'{file_prefix}_time_series.png')

plt.savefig(svg_path, format='svg', bbox_inches='tight')
plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')

print(f"图表已成功保存至: {output_dir}")
