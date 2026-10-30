import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. 字体与全局设置 (满足硕士论文标准：中文宋体，英文和数字 Times New Roman)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['SimSun'] # 全局默认使用宋体
plt.rcParams['mathtext.fontset'] = 'stix' # 数学公式和数字使用标准衬线体
plt.rcParams['axes.formatter.use_mathtext'] = True 
plt.rcParams['axes.unicode_minus'] = False

size = 10
plt.rcParams.update({
    'svg.fonttype': 'path',
    'font.size': 14+size,
    'axes.labelsize': 15+size,
    'xtick.labelsize': 13+size,
    'ytick.labelsize': 13+size,
})

# 2. 读取数据
data_path = 'data/7303_all.csv'
file_prefix = data_path.split('/')[-1].split('_')[0]

df = pd.read_csv(data_path, encoding='gb2312')
energy = df['能量[J]'].dropna().values

# 能量对数变换
energy_log = np.log10(1.0 + energy)

output_dir = 'data/img/213'
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# 3. 绘图 - 图a: 原始能量分布
# ==========================================
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(energy, bins=50, kde=True, ax=ax, color='#1f77b4', stat='density', alpha=0.6)
ax.set_xlabel('原始能量 $E$ (J)')
ax.set_ylabel('概率密度')

# 强制使用科学计数法
ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax.xaxis.get_offset_text().set_fontfamily('serif')
ax.xaxis.get_offset_text().set_fontname('SimSun')

formatter = plt.ScalarFormatter(useOffset=False)
formatter.set_useMathText(True)
ax.yaxis.set_major_formatter(formatter)

fig.canvas.draw()
tick_labels = ax.get_xticklabels() + ax.get_yticklabels()
for label_obj in tick_labels:
    label_obj.set_fontname('Times New Roman')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'{file_prefix}_energy_dist_original.svg'), format='svg', bbox_inches='tight')
plt.savefig(os.path.join(output_dir, f'{file_prefix}_energy_dist_original.png'), format='png', dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# 4. 绘图 - 图b: 对数变换后的能量分布
# ==========================================
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(energy_log, bins=50, kde=True, ax=ax, color='#d62728', stat='density', alpha=0.6)
ax.set_xlabel('对数能量 $\log_{10}(1+E)$')
ax.set_ylabel('概率密度')

formatter = plt.ScalarFormatter(useOffset=False)
formatter.set_useMathText(True)
ax.yaxis.set_major_formatter(formatter)

fig.canvas.draw()
tick_labels = ax.get_xticklabels() + ax.get_yticklabels()
for label_obj in tick_labels:
    label_obj.set_fontname('Times New Roman')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'{file_prefix}_energy_dist_log.svg'), format='svg', bbox_inches='tight')
plt.savefig(os.path.join(output_dir, f'{file_prefix}_energy_dist_log.png'), format='png', dpi=300, bbox_inches='tight')
plt.close()

print(f"图表已成功分别保存至: {output_dir}")
