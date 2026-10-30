import pickle
import cv2
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import KDTree
import math

# 【修改】：满足硕士论文标准：中文宋体，英文和数字 Times New Roman (通过 mathtext 实现)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['SimSun'] # 全局默认使用宋体
plt.rcParams['mathtext.fontset'] = 'stix' # 数学公式和数字使用标准衬线体 (类似 Times New Roman)
plt.rcParams['axes.formatter.use_mathtext'] = True # 强制坐标轴刻度使用 mathtext
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
data_path='data/7303_xyzet.csv'
df = pd.read_csv(data_path)
# 提取 7303 或 7304
file_prefix = data_path.split('/')[-1].split('_')[0]
#df = pd.read_csv('D:/workspace/python2/zhaolou/data/data_xyzet.csv')
data_x, data_y, data_z, data_e, data_t = df[['x']].values, df[['y']].values, df[['z']].values, df[['e']].values, df[['t']].values

# 计算倾斜角度
tan_theta = 879/1618
theta = math.atan(tan_theta)  # 弧度
theta_deg = math.degrees(theta)  # 角度
print(f"工作面倾斜角度: {theta_deg:.2f}°")

# 将坐标归一化到图像尺寸
scaler_x = MinMaxScaler(feature_range=(0, 3188))
scaler_y = MinMaxScaler(feature_range=(0, 2363))
workplace_x, workplace_y = [20400911, 20402820], [3915809, 3917226]
wp_x = scaler_x.fit_transform(np.array(workplace_x).reshape(-1, 1))
wp_y = scaler_y.fit_transform(np.array(workplace_y).reshape(-1, 1))
x = scaler_x.transform(np.array(data_x).reshape(-1, 1))
y = scaler_y.transform(np.array(data_y).reshape(-1, 1))
e = data_e

# 创建旋转矩阵（顺时针旋转θ角度）
rotation_matrix = np.array([
    [math.cos(theta), math.sin(theta)],
    [-math.sin(theta), math.cos(theta)]
])

# 坐标转换函数 - 只旋转微震事件坐标
def rotate_coordinates(coords, rotation_matrix, center):
    """
    旋转坐标
    
    参数:
    coords: 原始坐标数组，形状为 (n, 2)
    rotation_matrix: 旋转矩阵
    center: 旋转中心点
    
    返回:
    旋转后的坐标数组
    """
    # 将坐标平移到以中心点为原点
    translated_coords = coords - center
    
    # 应用旋转
    rotated_coords = np.dot(translated_coords, rotation_matrix.T)
    
    # 将坐标平移回原位置
    corrected_coords = rotated_coords + center
    
    return corrected_coords

# 选择旋转中心（图像中心）
center = np.array([3188/2, 2363/2])

# 准备坐标数据
original_coords = np.column_stack((x.flatten(), y.flatten()))

# 应用坐标旋转
rotated_coords = rotate_coordinates(original_coords, rotation_matrix, center)

# 分离旋转后的坐标
x_rotated = rotated_coords[:, 0].reshape(-1, 1)
y_rotated = rotated_coords[:, 1].reshape(-1, 1)


# 创建可视化
plt.figure(figsize=(12, 9))
#img = plt.imread("D:/workspace/python2/zhaolou/GongZuoMian.png")
img = plt.imread("data/img/GongZuoMian33.png")

width, height = img.shape[:2]
plt.imshow(img, origin='upper', extent=[0, height, 0, width])

# 绘制旋转后的微震事件
s = 15
plt.scatter(x_rotated, y_rotated, 
            s=s,               
            c='red',            
            edgecolors='white', 
            alpha=0.6,          
            zorder=2)

# 添加图例和标题
plt.legend()
plt.title(f'微震事件坐标旋转校正 - 顺时针旋转 {theta_deg:.2f}°')
plt.xlabel('X坐标 (像素)')
plt.ylabel('Y坐标 (像素)')

# 添加网格
plt.grid(True, alpha=0.3)

plt.tight_layout()
# plt.show() # 注释掉，防止阻塞

# 创建对比图 - 旋转前后的微震事件
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# 原始微震事件
ax1.imshow(img, origin='upper', extent=[0, height, 0, width])
ax1.scatter(
    x, y,
    s=5, c='red', alpha=0.6
)
ax1.set_title('原始微震事件分布')
ax1.set_xlabel('X坐标')
ax1.set_ylabel('Y坐标')

# 旋转后的微震事件
ax2.imshow(img, origin='upper', extent=[0, height, 0, width])
ax2.scatter(x_rotated, y_rotated, s=5, c='red', alpha=0.6)
ax2.set_title(f'旋转后的微震事件')
ax2.set_xlabel('X坐标')
ax2.set_ylabel('Y坐标')

plt.tight_layout()

# 【新增修改】：按照硕士毕业论文要求将对比图保存至 data/img 路径下（输出精美 SVG 矢量图给 Word 使用）
# 确保输出目录存在
import os
import matplotlib.pyplot as plt

os.makedirs('data/img', exist_ok=True)

size = 10

# 全局设置
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['SimSun'],
    'mathtext.fontset': 'stix',
    'axes.formatter.use_mathtext': True,
    'axes.unicode_minus': False,
    'svg.fonttype': 'path',
    'font.size': 14+size,
    'axes.titlesize': 16+size,
    'axes.labelsize': 15+size,
    'xtick.labelsize': 13+size,
    'ytick.labelsize': 13+size,
})

show_title = False

# 图1
plt.figure(figsize=(8, 8))
plt.imshow(img, origin='upper', extent=[0, height, 0, width])
plt.scatter(x, y, s=5, c='red', alpha=0.6)
if show_title:
    plt.title('原始微震事件分布', fontweight='bold')
plt.xlabel('X坐标 (m)')
plt.ylabel('Y坐标 (m)')
plt.tight_layout()
plt.savefig(f"data/img/{file_prefix}_original.svg", format='svg', bbox_inches='tight')
# 新增 PNG 保存
plt.savefig(f"data/img/{file_prefix}_original.png", format='png', dpi=300, bbox_inches='tight')

# 图2
plt.figure(figsize=(8, 8))
plt.imshow(img, origin='upper', extent=[0, height, 0, width])
plt.scatter(x_rotated, y_rotated, s=5, c='red', alpha=0.6)
if show_title:
    plt.title('旋转后微震事件分布', fontweight='bold')
plt.xlabel('X坐标 (m)')
plt.ylabel('Y坐标 (m)')
plt.tight_layout()
plt.savefig(f"data/img/{file_prefix}_rotated.svg", format='svg', bbox_inches='tight')
# 新增 PNG 保存
plt.savefig(f"data/img/{file_prefix}_rotated.png", format='png', dpi=300, bbox_inches='tight')

# plt.show() # 注释掉，防止阻塞

# 输出旋转参数
print("\n旋转参数:")
print(f"旋转角度: {theta_deg:.2f}° (顺时针)")
print(f"旋转中心: ({center[0]}, {center[1]})")
print(f"旋转矩阵: [[{rotation_matrix[0,0]:.4f}, {rotation_matrix[0,1]:.4f}],")
print(f"           [{rotation_matrix[1,0]:.4f}, {rotation_matrix[1,1]:.4f}]]")

# 将旋转后的坐标转换回原始坐标系
# 注意：这里需要将归一化后的坐标转换回原始经纬度坐标
x_rotated_original = scaler_x.inverse_transform(x_rotated.reshape(-1, 1))
y_rotated_original = scaler_y.inverse_transform(y_rotated.reshape(-1, 1))

# 创建旋转后的数据DataFrame
rotated_data = pd.DataFrame({
    'x': x_rotated_original.flatten(),
    'y': y_rotated_original.flatten(),
    'z': data_z.flatten(),
    'e': data_e.flatten(),
    't': data_t.flatten()
})

# 输出CSV格式的数据
print("\n旋转后的数据示例:")
print(rotated_data.head().to_string(index=False))

# 保存为CSV文件
#rotated_data.to_csv('D:\workspace\python3\cluster-pred-main\data/7303_xyzet_rotated.csv', index=False)
rotated_data.to_csv(f'data/{file_prefix}_xyzet_rotated.csv', index=False)
