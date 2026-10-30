import os
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# ==================== 1. 参数与底图配置 ====================
file_prefix = '6306'
data_path = f'data/{file_prefix}_xyzet.csv'
output_csv_path = f'data/{file_prefix}_xyzet_rotated.csv'

# 旋转参数 (CAD 测定)
theta_deg = 10.28178            # 倾斜夹角
center_x = 39489249.8160        # 切眼左下角基点 X
center_y = 3921105.0496         # 切眼左下角基点 Y
direction = 'clockwise'         # 顺时针旋转使工作面水平

# 底图路径与精确四至坐标范围 (左下到右上)
bg_img_path = 'data/img/GongZuoMian6306.png'
# 如果图片放在当前根目录，自动做个后备查找
if not os.path.exists(bg_img_path) and os.path.exists('GongZuoMian6306.png'):
    bg_img_path = 'GongZuoMian6306.png'

bg_xmin = 39489242
bg_ymin = 3921098
bg_xmax = 39490713
bg_ymax = 3921376
img_extent = [bg_xmin, bg_xmax, bg_ymin, bg_ymax]

# 视图范围模式:
# True: 严格聚焦在底图工作面切片范围内 (类似原 7303 效果)
# False: 自动扩展坐标轴，完整展示工作面周边顶底板的所有外围微震事件
crop_to_seam = False  

# ==================== 2. 读取微震数据 ====================
print(f"正在读取微震数据: {data_path} ...")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"未找到输入文件: {data_path}，请确认已生成 {file_prefix}_xyzet.csv")

df = pd.read_csv(data_path)
print(f"读取成功，共 {len(df)} 个微震事件。")

data_x = df['x'].values
data_y = df['y'].values
data_z = df['z'].values
data_e = df['e'].values
data_t = df['t'].values

# ==================== 3. 严格真实物理坐标旋转 ====================
theta_rad = math.radians(theta_deg)
dx = data_x - center_x
dy = data_y - center_y

if direction == 'clockwise':
    # 顺时针旋转公式 (将东北倾斜转至水平)
    x_rotated = center_x + dx * math.cos(theta_rad) + dy * math.sin(theta_rad)
    y_rotated = center_y - dx * math.sin(theta_rad) + dy * math.cos(theta_rad)
else:
    # 逆时针旋转公式
    x_rotated = center_x + dx * math.cos(theta_rad) - dy * math.sin(theta_rad)
    y_rotated = center_y + dx * math.sin(theta_rad) + dy * math.cos(theta_rad)

# 验证走向斜率
slope_orig = np.polyfit(data_x, data_y, 1)[0]
slope_rot = np.polyfit(x_rotated, y_rotated, 1)[0]
print(f"旋转前轴向夹角: {math.degrees(math.atan(slope_orig)):.2f}°")
print(f"旋转后轴向夹角: {math.degrees(math.atan(slope_rot)):.2f}° (已拉平)")

# ==================== 4. 保存旋转数据 ====================
df_rotated = pd.DataFrame({
    'x': np.round(x_rotated, 4),
    'y': np.round(y_rotated, 4),
    'z': data_z,
    'e': data_e,
    't': data_t
})
os.makedirs('data', exist_ok=True)
df_rotated.to_csv(output_csv_path, index=False)
print(f"✓ 旋转后数据已保存至: {output_csv_path}")

# ==================== 5. 论文标准可视化与底图叠置 ====================
os.makedirs('data/img', exist_ok=True)

# 硕士论文规范字体
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['SimSun', 'Songti SC', 'STSong'],
    'mathtext.fontset': 'stix',
    'axes.formatter.use_mathtext': True,
    'axes.unicode_minus': False,
    'svg.fonttype': 'path',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
})

has_bg = os.path.exists(bg_img_path)
bg_img = plt.imread(bg_img_path) if has_bg else None
if has_bg:
    print(f"✓ 成功加载底图: {bg_img_path}，将进行真实坐标叠置显示。")
else:
    print(f"提示: 未找到底图 {bg_img_path}，将直接绘制散点。")

# --- 图 1: 旋转后工作面底图与微震叠置主图 ---
plt.figure(figsize=(13, 6))
ax = plt.gca()

# 1. 绘制底层 CAD 工作面底图
if has_bg:
    ax.imshow(bg_img, extent=img_extent, origin='upper', aspect='equal', zorder=1)

# 2. 叠置微震事件点 (红色点加白色细边框与透明度，提高辨识度)
scatter = ax.scatter(
    x_rotated, y_rotated,
    s=12,
    c='red',
    edgecolors='white',
    linewidths=0.3,
    alpha=0.65,
    zorder=2,
    label='微震事件'
)

# 3. 标记切眼基准点
ax.scatter([center_x], [center_y], c='blue', marker='^', s=60, zorder=3, label='切眼旋转基点')

# 4. 控制显示视界
if crop_to_seam:
    ax.set_xlim(bg_xmin, bg_xmax)
    ax.set_ylim(bg_ymin, bg_ymax)
else:
    # 适当留白，既能清晰看清底图，也能看全外围顶板事件
    margin_x = 80
    margin_y = 120
    ax.set_xlim(min(bg_xmin, x_rotated.min()) - margin_x, max(bg_xmax, x_rotated.max()) + margin_x)
    ax.set_ylim(min(bg_ymin, y_rotated.min()) - margin_y, max(bg_ymax, y_rotated.max()) + margin_y)

ax.set_title(f'6306工作面微震事件分布图 (坐标校正后，顺时针旋转 {theta_deg:.2f}°)', fontweight='bold')
ax.set_xlabel('X坐标 (m)')
ax.set_ylabel('Y坐标 (m)')
ax.legend(loc='upper right')
ax.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

# 保存主图
plt.savefig(f"data/img/{file_prefix}_rotated_overlay.png", dpi=300, bbox_inches='tight')
plt.savefig(f"data/img/{file_prefix}_rotated_overlay.svg", format='svg', bbox_inches='tight')
plt.close()

# --- 图 2: 旋转前后对比图 (用于论文答辩/展示) ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# 子图 1: 原始未旋转 (大地经纬网坐标系)
ax1.scatter(data_x, data_y, s=8, c='#1f77b4', alpha=0.6, label='原始微震点')
ax1.scatter([center_x], [center_y], c='black', marker='^', s=50, zorder=3, label='切眼顶点')
ax1.set_title('原始微震事件分布 (经纬网坐标系，轴向倾斜)')
ax1.set_xlabel('X坐标 (m)')
ax1.set_ylabel('Y坐标 (m)')
ax1.axis('equal')
ax1.grid(True, linestyle='--', alpha=0.3)
ax1.legend(loc='upper left')

# 子图 2: 旋转后叠置底图
if has_bg:
    ax2.imshow(bg_img, extent=img_extent, origin='upper', aspect='equal', zorder=1)
ax2.scatter(x_rotated, y_rotated, s=8, c='red', alpha=0.65, edgecolors='white', linewidths=0.2, zorder=2, label='旋转后微震点')
ax2.scatter([center_x], [center_y], c='blue', marker='^', s=50, zorder=3, label='切眼顶点')
ax2.set_title(f'旋转后微震事件分布 (工作面水平坐标系，叠置底图)')
ax2.set_xlabel('X坐标 (m)')
ax2.set_ylabel('Y坐标 (m)')
if crop_to_seam:
    ax2.set_xlim(bg_xmin, bg_xmax)
    ax2.set_ylim(bg_ymin, bg_ymax)
else:
    ax2.axis('equal')
ax2.grid(True, linestyle='--', alpha=0.3)
ax2.legend(loc='upper left')

plt.tight_layout()
plt.savefig(f"data/img/{file_prefix}_comparison.png", dpi=300, bbox_inches='tight')
plt.savefig(f"data/img/{file_prefix}_comparison.svg", format='svg', bbox_inches='tight')
plt.close()

print(f"✓ 图像输出完毕: data/img/{file_prefix}_rotated_overlay.png 及 .svg")