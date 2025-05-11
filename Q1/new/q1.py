import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Polygon
from scipy.optimize import minimize
from colormath.color_objects import XYZColor, sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
# 中文显示设置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 定义BT2020和显示屏RGB的三基色坐标(CIE 1931坐标)
# 参考文献[4]中提到的BT2020标准
BT2020_RED = (0.708, 0.292)
BT2020_GREEN = (0.170, 0.797)
BT2020_BLUE = (0.131, 0.046)

# 假设显示屏的RGB三基色(实际上需要根据具体显示器校准获得)
DISPLAY_RED = (0.6946, 0.3047)
DISPLAY_GREEN = (0.2612, 0.7076)
DISPLAY_BLUE = (0.1418, 0.0417)


def plot_color_spaces():
    """绘制CIE 1931色彩空间，以及BT2020和显示屏的色域"""
    # 加载CIE 1931色度图数据(简化版，实际应使用精确数据)
    # 这是马蹄形曲线的x,y坐标近似
    wavelengths = np.arange(380, 700, 5)
    x = np.array([0.1741, 0.1740, 0.1738, 0.1736, 0.1733, 0.1730, 0.1726, 0.1721,
                  0.1714, 0.1703, 0.1689, 0.1669, 0.1644, 0.1611, 0.1566, 0.1510,
                  0.1440, 0.1355, 0.1241, 0.1096, 0.0913, 0.0687, 0.0454, 0.0235,
                  0.0082, 0.0039, 0.0139, 0.0389, 0.0743, 0.1142, 0.1547, 0.1929,
                  0.2296, 0.2658, 0.3016, 0.3373, 0.3731, 0.4087, 0.4441, 0.4788,
                  0.5125, 0.5448, 0.5752, 0.6029, 0.6270, 0.6482, 0.6658, 0.6801,
                  0.6915, 0.7006, 0.7079, 0.7140, 0.7190, 0.7230, 0.7260, 0.7283,
                  0.7300, 0.7311, 0.7320, 0.7327, 0.7334, 0.7340, 0.7344, 0.7346])
    y = np.array([0.0050, 0.0050, 0.0049, 0.0049, 0.0048, 0.0048, 0.0048, 0.0048,
                  0.0051, 0.0058, 0.0069, 0.0086, 0.0109, 0.0138, 0.0177, 0.0227,
                  0.0297, 0.0399, 0.0578, 0.0868, 0.1327, 0.2007, 0.2950, 0.4127,
                  0.5384, 0.6548, 0.7502, 0.8120, 0.8338, 0.8262, 0.8059, 0.7816,
                  0.7543, 0.7243, 0.6923, 0.6589, 0.6245, 0.5896, 0.5547, 0.5202,
                  0.4866, 0.4544, 0.4242, 0.3965, 0.3725, 0.3514, 0.3340, 0.3197,
                  0.3083, 0.2993, 0.2920, 0.2859, 0.2809, 0.2770, 0.2740, 0.2717,
                  0.2700, 0.2689, 0.2680, 0.2673, 0.2666, 0.2660, 0.2656, 0.2654])

    # 闭合马蹄形曲线
    x = np.append(x, [0.1741])
    y = np.append(y, [0.0050])

    # 添加从蓝点到红点的直线(紫色线)
    purple_line_x = np.linspace(x[0], x[-2], 20)
    purple_line_y = np.linspace(y[0], y[-2], 20)

    plt.figure(figsize=(10, 8))

    # 绘制马蹄形曲线
    plt.plot(x, y, '-', color='black', label='光谱轨迹')
    plt.plot(purple_line_x, purple_line_y, '-', color='purple', label='紫线')

    # 绘制BT2020色域
    bt2020_x = [BT2020_RED[0], BT2020_GREEN[0], BT2020_BLUE[0], BT2020_RED[0]]
    bt2020_y = [BT2020_RED[1], BT2020_GREEN[1], BT2020_BLUE[1], BT2020_RED[1]]
    plt.plot(bt2020_x, bt2020_y, '-', color='brown', linewidth=2, label='BT2020')
    bt2020_poly = Polygon(np.column_stack([bt2020_x, bt2020_y]),
                          alpha=0.2, color='brown')
    plt.gca().add_patch(bt2020_poly)

    # 绘制显示屏RGB色域
    display_x = [DISPLAY_RED[0], DISPLAY_GREEN[0], DISPLAY_BLUE[0], DISPLAY_RED[0]]
    display_y = [DISPLAY_RED[1], DISPLAY_GREEN[1], DISPLAY_BLUE[1], DISPLAY_RED[1]]
    plt.plot(display_x, display_y, '-', color='red', linewidth=2, label='显示屏RGB')
    display_poly = Polygon(np.column_stack([display_x, display_y]),
                           alpha=0.2, color='red')
    plt.gca().add_patch(display_poly)

    # 标记RGB点
    plt.scatter([BT2020_RED[0], BT2020_GREEN[0], BT2020_BLUE[0]],
                [BT2020_RED[1], BT2020_GREEN[1], BT2020_BLUE[1]],
                color='brown', s=50)
    plt.scatter([DISPLAY_RED[0], DISPLAY_GREEN[0], DISPLAY_BLUE[0]],
                [DISPLAY_RED[1], DISPLAY_GREEN[1], DISPLAY_BLUE[1]],
                color='red', s=50)

    # 添加白点
    plt.scatter([0.3127, 0.3127], [0.3290, 0.3290], color='black', s=50, label='D65白点')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('CIE 1931色彩空间及BT2020与显示屏RGB色域对比')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 0.8)
    plt.ylim(0, 0.9)
    plt.tight_layout()

    # 计算色域面积
    def calculate_area(x, y):
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    bt2020_area = calculate_area(bt2020_x[:-1], bt2020_y[:-1])
    display_area = calculate_area(display_x[:-1], display_y[:-1])
    
    print("\n色域面积分析:")
    print(f"BT2020色域面积: {bt2020_area:.6f}")
    print(f"显示屏RGB色域面积: {display_area:.6f}")
    print(f"色域覆盖率: {(display_area/bt2020_area*100):.2f}%")
    
    return plt.gcf()


def xy_to_XYZ(x, y, Y=1.0):
    """将CIE xy坐标转换为XYZ值"""
    X = (x * Y) / y
    Z = ((1 - x - y) * Y) / y
    return X, Y, Z


def XYZ_to_xy(X, Y, Z):
    """将XYZ值转换为CIE xy坐标"""
    sum_XYZ = X + Y + Z
    if sum_XYZ == 0:
        return 0, 0
    x = X / sum_XYZ
    y = Y / sum_XYZ
    return x, y


def compute_primary_matrix():
    """计算BT2020和显示屏RGB原色的转换矩阵"""
    # BT2020原色的XYZ值
    BT2020_R_XYZ = xy_to_XYZ(*BT2020_RED)
    BT2020_G_XYZ = xy_to_XYZ(*BT2020_GREEN)
    BT2020_B_XYZ = xy_to_XYZ(*BT2020_BLUE)

    # 显示屏原色的XYZ值
    DISPLAY_R_XYZ = xy_to_XYZ(*DISPLAY_RED)
    DISPLAY_G_XYZ = xy_to_XYZ(*DISPLAY_GREEN)
    DISPLAY_B_XYZ = xy_to_XYZ(*DISPLAY_BLUE)

    # 构建原色矩阵
    BT2020_matrix = np.array([
        [BT2020_R_XYZ[0], BT2020_G_XYZ[0], BT2020_B_XYZ[0]],
        [BT2020_R_XYZ[1], BT2020_G_XYZ[1], BT2020_B_XYZ[1]],
        [BT2020_R_XYZ[2], BT2020_G_XYZ[2], BT2020_B_XYZ[2]]
    ])

    DISPLAY_matrix = np.array([
        [DISPLAY_R_XYZ[0], DISPLAY_G_XYZ[0], DISPLAY_B_XYZ[0]],
        [DISPLAY_R_XYZ[1], DISPLAY_G_XYZ[1], DISPLAY_B_XYZ[1]],
        [DISPLAY_R_XYZ[2], DISPLAY_G_XYZ[2], DISPLAY_B_XYZ[2]]
    ])

    # 计算转换矩阵 (从BT2020到显示屏RGB)
    conversion_matrix = np.linalg.inv(DISPLAY_matrix) @ BT2020_matrix

    # print("\n原色矩阵分析:")
    # print("BT2020原色矩阵:")
    # print(BT2020_matrix)
    # print("\n显示屏RGB原色矩阵:")
    # print(DISPLAY_matrix)
    # print("\n转换矩阵 (BT2020 -> 显示屏RGB):")
    # print(conversion_matrix)

    return BT2020_matrix, DISPLAY_matrix, conversion_matrix


def convert_color_bt2020_to_display(bt2020_rgb):
    """使用转换矩阵将BT2020的RGB值转换为显示屏的RGB值"""
    _, _, conversion_matrix = compute_primary_matrix()
    display_rgb = np.dot(conversion_matrix, bt2020_rgb)

    # 处理超出范围的颜色值
    return np.clip(display_rgb, 0, 1)


# 定义损失函数：使转换前后的色差最小
def color_difference(bt2020_xyz, display_xyz):
    """计算两个XYZ颜色空间点之间的欧氏距离"""
    return np.sqrt(np.sum((bt2020_xyz - display_xyz) ** 2))


def optimize_conversion(bt2020_rgb):
    """优化转换，使色差最小"""
    # 转换BT2020 RGB到XYZ
    BT2020_matrix, _, _ = compute_primary_matrix()
    bt2020_xyz = np.dot(BT2020_matrix, bt2020_rgb)

    # 定义目标函数：最小化色差
    def objective(display_rgb):
        # 确保RGB值在[0,1]范围内
        display_rgb_clipped = np.clip(display_rgb, 0, 1)
        # 转换显示屏RGB到XYZ
        _, DISPLAY_matrix, _ = compute_primary_matrix()
        display_xyz = np.dot(DISPLAY_matrix, display_rgb_clipped)
        # 返回色差
        return color_difference(bt2020_xyz, display_xyz)

    # 初始猜测（使用简单转换）
    initial_guess = convert_color_bt2020_to_display(bt2020_rgb)

    # 优化
    bounds = [(0, 1), (0, 1), (0, 1)]  # RGB值范围为[0,1]
    result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')

    # 计算优化前后的色差
    _, DISPLAY_matrix, _ = compute_primary_matrix()
    initial_xyz = np.dot(DISPLAY_matrix, initial_guess)
    final_xyz = np.dot(DISPLAY_matrix, result.x)
    initial_diff = color_difference(bt2020_xyz, initial_xyz)
    final_diff = color_difference(bt2020_xyz, final_xyz)
    
    print(f"\n颜色优化分析 (RGB={bt2020_rgb}):")
    print(f"初始色差: {initial_diff:.6f}")
    print(f"优化后色差: {final_diff:.6f}")
    print(f"色差改善: {((initial_diff-final_diff)/initial_diff*100):.2f}%")

    return np.clip(result.x, 0, 1)  # 确保结果在[0,1]范围内

def analyze_conversion_accuracy():
    """分析4通道到5通道转换的精度"""
    # 创建测试颜色集
    test_colors = [
        np.array([1.0, 0.0, 0.0]),  # 纯R
        np.array([0.0, 1.0, 0.0]),  # 纯G
        np.array([0.0, 0.0, 1.0]),  # 纯B
        np.array([0.5, 0.5, 0.0]),  # RG混合
        np.array([0.5, 0.0, 0.5]),  # RB混合
        np.array([0.0, 0.5, 0.5]),  # GB混合
        np.array([0.0, 0.0, 0.5]),  # BV混合
        np.array([0.33, 0.33, 0.33]),  # 均匀混合
        np.array([0.5, 0.3, 0.2]),  # 复杂混合1
        np.array([0.2, 0.4, 0.4]),  # 复杂混合2
        np.array([0.1, 0.1, 0.8]),  # 复杂混合3
        np.array([0.3, 0.2, 0.5]),  # 复杂混合4
    ]

    # 获取颜色矩阵
    BT2020_matrix, DISPLAY_matrix, _ = compute_primary_matrix()

    # 存储结果
    results = []
    total_error = 0
    max_error = 0
    min_error = float('inf')

    print("\n转换精度分析结果：")
    print("=" * 50)
    print("颜色\t\t\t源RGBV\t\t\t目标rgb\t\t\t色差(ΔE)")
    print("-" * 100)

    for i, color in enumerate(test_colors):
        # 转换到5通道
        display_rgb = convert_bt2020_to_display(color)

        # 计算原始颜色和目标颜色的XYZ值
        BT2020_xyz = np.dot(BT2020_matrix, color)
        display_xyz = np.dot(DISPLAY_matrix, display_rgb)

        # 计算色差(ΔE)
        error = np.sqrt(np.sum((BT2020_xyz - display_xyz) ** 2))
        total_error += error
        max_error = max(max_error, error)
        min_error = min(min_error, error)

        # 格式化输出
        color_name = f"颜色{i+1}"
        BT2020_str = f"({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})"
        target_str = f"({display_rgb[0]:.2f}, {display_rgb[1]:.2f}, {display_rgb[2]:.2f})"
        print(f"{color_name}\t{BT2020_str}\t{target_str}\t{error:.4f}")

        results.append({
            'BT2020': color,
            'target': display_rgb,
            'error': error
        })

    # 计算统计信息
    avg_error = total_error / len(test_colors)
    
    print("\n统计信息：")
    print("=" * 50)
    print(f"平均色差(ΔE): {avg_error:.4f}")
    print(f"最大色差(ΔE): {max_error:.4f}")
    print(f"最小色差(ΔE): {min_error:.4f}")
    print(f"色差标准差: {np.std([r['error'] for r in results]):.4f}")

    return results


def visualize_color_mapping():
    """可视化色彩映射效果"""
    # 创建BT2020色域内的采样点
    samples = 10  # 每个维度的采样数
    r_values = np.linspace(0, 1, samples)
    g_values = np.linspace(0, 1, samples)
    b_values = np.linspace(0, 1, samples)

    # 存储转换前后的颜色
    original_colors = []
    mapped_colors = []

    # 对于每个采样点
    for r in r_values:
        for g in g_values:
            for b in b_values:
                bt2020_rgb = np.array([r, g, b])
                # 简单转换
                simple_display_rgb = convert_color_bt2020_to_display(bt2020_rgb)
                # 优化转换
                optimized_display_rgb = convert_bt2020_to_display(bt2020_rgb)

                # 添加到颜色列表
                original_colors.append(bt2020_rgb)
                # 使用优化的结果
                mapped_colors.append(optimized_display_rgb)

    # 转换为数组
    original_colors = np.array(original_colors)
    mapped_colors = np.array(mapped_colors)

    # 计算转换前后的xy坐标
    BT2020_matrix, DISPLAY_matrix, _ = compute_primary_matrix()

    original_xyz = np.array([np.dot(BT2020_matrix, rgb) for rgb in original_colors])
    mapped_xyz = np.array([np.dot(DISPLAY_matrix, rgb) for rgb in mapped_colors])

    original_xy = np.array([XYZ_to_xy(*xyz) for xyz in original_xyz])
    mapped_xy = np.array([XYZ_to_xy(*xyz) for xyz in mapped_xyz])

    # 绘制结果
    plt.figure(figsize=(12, 10))

    # 绘制CIE 1931马蹄形曲线(简化)
    wavelengths = np.arange(380, 700, 5)
    x = np.array([0.1741, 0.1740, 0.1738, 0.1736, 0.1733, 0.1730, 0.1726, 0.1721,
                  0.1714, 0.1703, 0.1689, 0.1669, 0.1644, 0.1611, 0.1566, 0.1510,
                  0.1440, 0.1355, 0.1241, 0.1096, 0.0913, 0.0687, 0.0454, 0.0235,
                  0.0082, 0.0039, 0.0139, 0.0389, 0.0743, 0.1142, 0.1547, 0.1929,
                  0.2296, 0.2658, 0.3016, 0.3373, 0.3731, 0.4087, 0.4441, 0.4788,
                  0.5125, 0.5448, 0.5752, 0.6029, 0.6270, 0.6482, 0.6658, 0.6801,
                  0.6915, 0.7006, 0.7079, 0.7140, 0.7190, 0.7230, 0.7260, 0.7283,
                  0.7300, 0.7311, 0.7320, 0.7327, 0.7334, 0.7340, 0.7344, 0.7346])
    y = np.array([0.0050, 0.0050, 0.0049, 0.0049, 0.0048, 0.0048, 0.0048, 0.0048,
                  0.0051, 0.0058, 0.0069, 0.0086, 0.0109, 0.0138, 0.0177, 0.0227,
                  0.0297, 0.0399, 0.0578, 0.0868, 0.1327, 0.2007, 0.2950, 0.4127,
                  0.5384, 0.6548, 0.7502, 0.8120, 0.8338, 0.8262, 0.8059, 0.7816,
                  0.7543, 0.7243, 0.6923, 0.6589, 0.6245, 0.5896, 0.5547, 0.5202,
                  0.4866, 0.4544, 0.4242, 0.3965, 0.3725, 0.3514, 0.3340, 0.3197,
                  0.3083, 0.2993, 0.2920, 0.2859, 0.2809, 0.2770, 0.2740, 0.2717,
                  0.2700, 0.2689, 0.2680, 0.2673, 0.2666, 0.2660, 0.2656, 0.2654])

    purple_line_x = np.linspace(x[0], x[-1], 20)
    purple_line_y = np.linspace(y[0], y[-1], 20)

    plt.plot(x, y, '-', color='black', label='光谱轨迹')
    plt.plot(purple_line_x, purple_line_y, '-', color='purple', label='紫线')

    # 绘制BT2020色域
    bt2020_x = [BT2020_RED[0], BT2020_GREEN[0], BT2020_BLUE[0], BT2020_RED[0]]
    bt2020_y = [BT2020_RED[1], BT2020_GREEN[1], BT2020_BLUE[1], BT2020_RED[1]]
    plt.plot(bt2020_x, bt2020_y, '-', color='brown', linewidth=2, label='BT2020')
    bt2020_poly = Polygon(np.column_stack([bt2020_x, bt2020_y]),
                          alpha=0.2, color='brown')
    plt.gca().add_patch(bt2020_poly)

    # 绘制显示屏RGB色域
    display_x = [DISPLAY_RED[0], DISPLAY_GREEN[0], DISPLAY_BLUE[0], DISPLAY_RED[0]]
    display_y = [DISPLAY_RED[1], DISPLAY_GREEN[1], DISPLAY_BLUE[1], DISPLAY_RED[1]]
    plt.plot(display_x, display_y, '-', color='red', linewidth=2, label='显示屏RGB')
    display_poly = Polygon(np.column_stack([display_x, display_y]),
                           alpha=0.2, color='red')
    plt.gca().add_patch(display_poly)

    # 绘制转换前后的点
    plt.scatter(original_xy[:, 0], original_xy[:, 1], c='blue', alpha=0.5, s=20, label='BT2020 原始颜色')
    plt.scatter(mapped_xy[:, 0], mapped_xy[:, 1], c='green', alpha=0.5, s=20, label='映射后颜色')

    # 绘制转换对应关系(绘制一些样本点的对应关系，避免过于拥挤)
    for i in range(0, len(original_xy), len(original_xy) // 20):
        plt.plot([original_xy[i, 0], mapped_xy[i, 0]],
                 [original_xy[i, 1], mapped_xy[i, 1]],
                 'k-', alpha=0.2)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('BT2020到显示屏RGB的颜色映射可视化')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 0.8)
    plt.ylim(0, 0.9)
    plt.tight_layout()

    return plt.gcf()


def compare_color_patches():
    """比较原始颜色和映射后颜色的视觉效果"""
    # 创建一些测试颜色
    test_colors = [
        (1.0, 0.0, 0.0),  # 纯红
        (0.0, 1.0, 0.0),  # 纯绿
        (0.0, 0.0, 1.0),  # 纯蓝
        (1.0, 1.0, 0.0),  # 黄
        (0.0, 1.0, 1.0),  # 青
        (1.0, 0.0, 1.0),  # 洋红
        (0.5, 0.0, 0.0),  # 暗红
        (0.0, 0.5, 0.0),  # 暗绿
        (0.0, 0.0, 0.5),  # 暗蓝
        (0.5, 0.5, 0.0),  # 橄榄
        (0.0, 0.5, 0.5),  # 蓝绿
        (0.5, 0.0, 0.5),  # 紫
        (0.5, 0.5, 0.5),  # 灰
        (1.0, 0.5, 0.0),  # 橙
        (0.0, 0.5, 1.0),  # 天蓝
        (1.0, 0.0, 0.5),  # 粉红
    ]

    # 进行颜色转换
    mapped_colors = [convert_bt2020_to_display(np.array(color)) for color in test_colors]

    # 创建显示图
    fig, axes = plt.subplots(len(test_colors), 2, figsize=(8, 2 * len(test_colors)))
    fig.suptitle('BT2020与映射到显示屏RGB的颜色对比', fontsize=16)

    for i, (original, mapped) in enumerate(zip(test_colors, mapped_colors)):
        # 显示原始颜色
        axes[i, 0].add_patch(plt.Rectangle((0, 0), 1, 1, color=original))
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        axes[i, 0].set_title(f'原始 ({original[0]:.1f}, {original[1]:.1f}, {original[2]:.1f})')

        # 显示映射后的颜色
        axes[i, 1].add_patch(plt.Rectangle((0, 0), 1, 1, color=mapped))
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])
        axes[i, 1].set_title(f'映射 ({mapped[0]:.1f}, {mapped[1]:.1f}, {mapped[2]:.1f})')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    return plt.gcf()

BT2020_matrix, DISPLAY_matrix, BT2020_to_display_matrix = compute_primary_matrix()

# 定义 MLP 模型，使用 Hardtanh 激活函数, 保证输出为[0,1]
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 3)
        #self.dropout = nn.Dropout(p=0.5)  # 添加 Dropout 层
        self.hardtanh = nn.Hardtanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # 将 Hardtanh 的输出从 [-1, 1] 映射到 [0, 1]
        #x = self.dropout(x)
        x = (self.hardtanh(x) + 1) / 2
        return x


# def BT2020rgb_to_xy(rgb):
#     if len(rgb.shape) == 1:
#         rgb = np.expand_dims(rgb, axis=0)
#     xy_list = []
#     for r, g, b in rgb:
#         xyz  = np.dot(BT2020_matrix, np.array([r, g, b]))
#         x, y = XYZ_to_xy(*xyz)
#         xy_list.append([x, y])
#     return np.array(xy_list)

# def displayrgb_to_xy(rgb):
#     if len(rgb.shape) == 1:
#         rgb = np.expand_dims(rgb, axis=0)
#     xy_list = []
#     for r, g, b in rgb:
#         xyz  = np.dot(DISPLAY_matrix, np.array([r, g, b]))
#         x, y = XYZ_to_xy(*xyz)
#         xy_list.append([x, y])
#     return np.array(xy_list)
def BT2020rgb_to_xy(rgb: torch.Tensor) -> torch.Tensor:
    """
    将BT2020 RGB颜色值转换为CIE xy色度坐标
    参数:
        rgb: 形状为 (..., 3) 的PyTorch张量，表示RGB颜色值
    返回:
        xy: 形状为 (..., 2) 的PyTorch张量，表示xy色度坐标
    """
    # 确保输入是正确的形状
    if rgb.dim() == 1:
        rgb = rgb.unsqueeze(0)  # 添加批次维度
    
    # 将矩阵转换为PyTorch张量
    BT2020_matrix_tensor = torch.tensor(BT2020_matrix, dtype=rgb.dtype, device=rgb.device)
    
    # 执行矩阵乘法：RGB到XYZ
    # 假设BT2020_matrix是3x3矩阵，rgb是(..., 3)张量
    xyz = torch.matmul(rgb, BT2020_matrix_tensor.T)  # (..., 3)
    
    # 计算xy坐标
    sum_xyz = torch.sum(xyz, dim=-1, keepdim=True)  # (..., 1)
    x = xyz[..., 0:1] / (sum_xyz + 1e-8)  # (..., 1)
    y = xyz[..., 1:2] / (sum_xyz + 1e-8)  # (..., 1)
    
    return torch.cat([x, y], dim=-1)  # (..., 2)

def displayrgb_to_xy(rgb: torch.Tensor) -> torch.Tensor:
    """
    将显示屏RGB颜色值转换为CIE xy色度坐标
    参数:
        rgb: 形状为 (..., 3) 的PyTorch张量，表示RGB颜色值
    返回:
        xy: 形状为 (..., 2) 的PyTorch张量，表示xy色度坐标
    """
    # 确保输入是正确的形状
    if rgb.dim() == 1:
        rgb = rgb.unsqueeze(0)  # 添加批次维度
    
    # 将矩阵转换为PyTorch张量
    DISPLAY_matrix_tensor = torch.tensor(DISPLAY_matrix, dtype=rgb.dtype, device=rgb.device)
    
    # 执行矩阵乘法：RGB到XYZ
    xyz = torch.matmul(rgb, DISPLAY_matrix_tensor.T)  # (..., 3)
    
    # 计算xy坐标
    sum_xyz = torch.sum(xyz, dim=-1, keepdim=True)  # (..., 1)
    x = xyz[..., 0:1] / (sum_xyz + 1e-8)  # (..., 1)
    y = xyz[..., 1:2] / (sum_xyz + 1e-8)  # (..., 1)
    
    return torch.cat([x, y], dim=-1)  # (..., 2)

# 定义基于 PyTorch 的色差计算函数
def color_difference_torch(rgb1, rgb2):
    # 确保输入是张量
    if not isinstance(rgb1, torch.Tensor):
        rgb1 = torch.tensor(rgb1, dtype=torch.float32)
    if not isinstance(rgb2, torch.Tensor):
        rgb2 = torch.tensor(rgb2, dtype=torch.float32)
        
    # 计算欧氏距离（简化版，实际应使用 CIEDE2000 的 PyTorch 实现）
    return torch.sqrt(torch.sum((rgb1 - rgb2) ** 2))

def XYZ_to_Lab(XYZ: torch.Tensor) -> torch.Tensor:
    """
    将 XYZ 颜色空间转换为 Lab 颜色空间（PyTorch 实现）
    """
    # CIE 标准照明体 D65
    Xn, Yn, Zn = 95.047, 100.000, 108.883
    
    # 归一化
    XYZ_norm = XYZ / torch.tensor([Xn, Yn, Zn], device=XYZ.device, dtype=XYZ.dtype)
    
    # 非线性变换
    def f(t):
        delta = 6/29
        return torch.where(t > delta**3, 
                          torch.pow(t, 1/3), 
                          t/(3 * delta**2) + 4/29)
    
    f_XYZ = f(XYZ_norm)
    
    # 计算 Lab
    L = 116 * f_XYZ[..., 1] - 16
    a = 500 * (f_XYZ[..., 0] - f_XYZ[..., 1])
    b = 200 * (f_XYZ[..., 1] - f_XYZ[..., 2])
    
    return torch.stack([L, a, b], dim=-1)

def delta_e_cie2000_torch(Lab1: torch.Tensor, Lab2: torch.Tensor) -> torch.Tensor:
    """
    计算两个 Lab 颜色之间的 CIEDE2000 色差（PyTorch 实现）
    """
    # 提取 L, a, b 分量
    L1, a1, b1 = Lab1[..., 0], Lab1[..., 1], Lab1[..., 2]
    L2, a2, b2 = Lab2[..., 0], Lab2[..., 1], Lab2[..., 2]
    
    # 计算 CIE94 中的 C*ab
    C1_ab = torch.sqrt(a1**2 + b1**2)
    C2_ab = torch.sqrt(a2**2 + b2**2)
    
    # 计算平均 C*ab
    C_ab_mean = (C1_ab + C2_ab) / 2
    
    # 创建常量张量而不使用 torch.tensor
    const_25 = C_ab_mean.new_full((), 25.0)
    const_1e10 = C_ab_mean.new_full((), 1e-10)
    
    # 安全计算 G 参数
    C_ab_mean_p7 = torch.pow(C_ab_mean, 7)
    denominator = C_ab_mean_p7 + torch.pow(const_25, 7) + const_1e10
    G = 0.5 * (1 - torch.sqrt(C_ab_mean_p7 / denominator))
    
    # 计算调整后的 a'
    a1_prime = (1 + G) * a1
    a2_prime = (1 + G) * a2
    
    # 计算 C'
    C1_prime = torch.sqrt(a1_prime**2 + b1**2)
    C2_prime = torch.sqrt(a2_prime**2 + b2**2)
    
    # 计算平均 C'
    C_prime_mean = (C1_prime + C2_prime) / 2
    
    # 计算 h'
    def calculate_h_prime(a_prime, b):
        h_prime = torch.atan2(b, a_prime) * (180 / np.pi)
        h_prime = h_prime % 360
        return torch.where(h_prime < 0, h_prime + 360, h_prime)
    
    h1_prime = calculate_h_prime(a1_prime, b1)
    h2_prime = calculate_h_prime(a2_prime, b2)
    
    # 计算 Δh'
    delta_h_prime = torch.zeros_like(h1_prime)
    
    mask1 = (torch.abs(h1_prime - h2_prime) <= 180)
    delta_h_prime[mask1] = h2_prime[mask1] - h1_prime[mask1]
    
    mask2 = (h1_prime - h2_prime > 180)
    delta_h_prime[mask2] = h2_prime[mask2] - h1_prime[mask2] + 360
    
    mask3 = (h1_prime - h2_prime < -180)
    delta_h_prime[mask3] = h2_prime[mask3] - h1_prime[mask3] - 360
    
    # 计算 ΔL', ΔC', ΔH'
    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime
    delta_H_prime = 2 * torch.sqrt(C1_prime * C2_prime) * torch.sin(delta_h_prime * (np.pi / 180) / 2)
    
    # 计算平均 L', C', h'
    L_prime_mean = (L1 + L2) / 2
    C_prime_mean = (C1_prime + C2_prime) / 2
    
    # 计算平均 h'
    h_prime_mean = torch.zeros_like(h1_prime)
    
    mask4 = (torch.abs(h1_prime - h2_prime) <= 180)
    h_prime_mean[mask4] = (h1_prime[mask4] + h2_prime[mask4]) / 2
    
    mask5 = ((h1_prime + h2_prime) < 360) & (torch.abs(h1_prime - h2_prime) > 180)
    h_prime_mean[mask5] = (h1_prime[mask5] + h2_prime[mask5] + 360) / 2
    
    mask6 = ((h1_prime + h2_prime) >= 360) & (torch.abs(h1_prime - h2_prime) > 180)
    h_prime_mean[mask6] = (h1_prime[mask6] + h2_prime[mask6] - 360) / 2
    
    # 计算各项修正因子
    T = 1 - 0.17 * torch.cos((h_prime_mean - 30) * (np.pi / 180)) + \
        0.24 * torch.cos((2 * h_prime_mean) * (np.pi / 180)) + \
        0.32 * torch.cos((3 * h_prime_mean + 6) * (np.pi / 180)) - \
        0.20 * torch.cos((4 * h_prime_mean - 63) * (np.pi / 180))
    
    delta_theta = 30 * torch.exp(-torch.pow((h_prime_mean - 275) / 25, 2))
    
    # 安全计算 RC
    C_prime_mean_p7 = torch.pow(C_prime_mean, 7)
    RC_denominator = C_prime_mean_p7 + torch.pow(const_25, 7) + const_1e10
    RC = 2 * torch.sqrt(C_prime_mean_p7 / RC_denominator)
    
    # 安全计算 SL
    L_diff_squared = torch.pow(L_prime_mean - 50, 2)
    SL_denominator = torch.sqrt(20 + L_diff_squared) + const_1e10
    SL = 1 + (0.015 * L_diff_squared) / SL_denominator
    
    SC = 1 + 0.045 * C_prime_mean
    SH = 1 + 0.015 * C_prime_mean * T
    
    RT = -RC * torch.sin(2 * delta_theta * (np.pi / 180))
    
    # 计算最终色差
    delta_E = torch.sqrt(
        torch.pow(delta_L_prime / SL, 2) +
        torch.pow(delta_C_prime / SC, 2) +
        torch.pow(delta_H_prime / SH, 2) +
        RT * (delta_C_prime / SC) * (delta_H_prime / SH)
    )
    
    return delta_E

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 生成训练数据
num_samples = 1000
bt2020_rgb_train = np.random.uniform(0, 1, (num_samples, 3))
bt2020_rgb_train = torch.tensor(bt2020_rgb_train, dtype=torch.float32).to(device)

# 创建数据集和数据加载器
batch_size = 32  # 小批量的大小
dataset = TensorDataset(bt2020_rgb_train)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化 MLP 模型并移动到 GPU
model = MLP().to(device)

# 定义优化器为 SGD
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 存储每个 epoch 的损失
losses = []

# 训练模型
num_epochs = 200
for epoch in tqdm(range(num_epochs)):
    epoch_loss = 0
    for batch in dataloader:
        bt2020_rgb_batch = batch[0]
        outputs = model(bt2020_rgb_batch)

        # 计算输入和输出的 xy 值
        bt2020_xy = BT2020rgb_to_xy(bt2020_rgb_batch)
        output_xy = displayrgb_to_xy(outputs)

        # 计算 CIEDE2000 色差
        # 将xy转换为XYZ (假设Y=1)
        bt2020_XYZ = torch.stack([
            bt2020_xy[..., 0] * 1 / bt2020_xy[..., 1],
            torch.ones_like(bt2020_xy[..., 0]),
            (1 - bt2020_xy[..., 0] - bt2020_xy[..., 1]) * 1 / bt2020_xy[..., 1]
        ], dim=-1)
        
        output_XYZ = torch.stack([
            output_xy[..., 0] * 1 / output_xy[..., 1],
            torch.ones_like(output_xy[..., 0]),
            (1 - output_xy[..., 0] - output_xy[..., 1]) * 1 / output_xy[..., 1]
        ], dim=-1)
        
        # 将XYZ转换为Lab
        bt2020_Lab = XYZ_to_Lab(bt2020_XYZ)
        output_Lab = XYZ_to_Lab(output_XYZ)
        
        # 计算CIEDE2000色差
        batch_loss = color_difference_torch(bt2020_xy,output_xy)#delta_e_cie2000_torch(bt2020_Lab, output_Lab).mean()
        # batch_loss = 0
        # for i in range(len(bt2020_rgb_batch)):
        #     color1 = XYZColor(bt2020_xy[i, 0], bt2020_xy[i, 1], 1 - bt2020_xy[i, 0] - bt2020_xy[i, 1])
        #     color2 = XYZColor(output_xy[i, 0], output_xy[i, 1], 1 - output_xy[i, 0] - output_xy[i, 1])
        #     # 将 XYZColor 转换为 LabColor
        #     lab_color1 = convert_color(color1, LabColor)
        #     lab_color2 = convert_color(color2, LabColor)
        #     batch_loss += delta_e_cie2000(lab_color1, lab_color2)
        # batch_loss = batch_loss / len(bt2020_rgb_batch)
        #batch_loss = color_difference_torch(bt2020_xy, output_xy)

        optimizer.zero_grad()
        batch_loss.backward()

        # 检查是否有梯度更新
        # with torch.no_grad():
        #     has_grad = False
        #     for name, param in model.named_parameters():
        #         if param.grad is not None and torch.norm(param.grad) > 1e-8:
        #             has_grad = True
        #             print(f"参数 {name} 有梯度，范数: {torch.norm(param.grad):.6f}")
            
        #     if not has_grad:
        #         print("警告：所有参数梯度为零！")
    
        optimizer.step()

        epoch_loss += batch_loss.item()

    epoch_loss = epoch_loss / len(dataloader)
    losses.append(epoch_loss)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# 保存模型参数
torch.save(model.state_dict(), 'mlp_model.pth')
print("模型参数已保存到 mlp_model.pth")

# 绘制损失曲线
import matplotlib.pyplot as plt
plt.plot(range(1, num_epochs + 1), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.savefig('loss_curve.png')
print("损失曲线已保存为 loss_curve.png")

# 定义颜色转换函数
def convert_bt2020_to_display(bt2020_rgb):
    display_rgb = np.dot(BT2020_to_display_matrix, bt2020_rgb)
    # return display_rgb.clip(0, 1)

    if np.all((display_rgb >= 0) & (display_rgb <= 1)): #如果在范围内，说明得到合理的显示器RGB值，在重合区域，直接返回
        display_rgb = display_rgb
    else:
        bt2020_rgb_tensor = torch.tensor(bt2020_rgb, dtype=torch.float32).to(device)
        display_rgb_tensor = model(bt2020_rgb_tensor)
        # 将 GPU 上的张量移动到 CPU 上，再转换为 NumPy 数组
        display_rgb = display_rgb_tensor.cpu().detach().numpy()

    # 确保输出在0 - 1之间
    display_rgb = np.clip(display_rgb, 0, 1)
    return display_rgb

def test_mlp_model(rgb_values, model_path='mlp_model.pth'):
    """
    加载保存的模型参数并处理输入的 RGB 值。
    :param rgb_values: 输入的 RGB 值，形状为 (n, 3) 的 numpy 数组或列表
    :param model_path: 保存的模型参数文件路径
    :return: 处理后的 RGB 值，形状为 (n, 3) 的 numpy 数组
    """
    def convert(bt2020_rgb, mlpmodel):
        display_rgb = np.dot(BT2020_to_display_matrix, bt2020_rgb)

        if np.all((display_rgb >= 0) & (display_rgb <= 1)): #如果在范围内，说明得到合理的显示器RGB值，在重合区域，直接返回
            display_rgb = display_rgb
        else:
            bt2020_rgb_tensor = torch.tensor(bt2020_rgb, dtype=torch.float32)
            display_rgb_tensor = mlpmodel(bt2020_rgb_tensor)
            display_rgb = display_rgb_tensor.detach().numpy()

        # 确保输出在0 - 1之间
        display_rgb = np.clip(display_rgb, 0, 1)
        return display_rgb
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except FileNotFoundError:
        print(f"错误: 未找到模型文件 {model_path}")
        return None
    return convert(rgb_values, model)

def main():
    # 绘制色彩空间
    color_space_fig = plot_color_spaces()
    color_space_fig.savefig('color_spaces.png')

    # 计算转换矩阵
    BT2020_matrix, DISPLAY_matrix, conversion_matrix = compute_primary_matrix()
    
    # 测试一些关键颜色
    test_colors = [
        np.array([1.0, 0.0, 0.0]),  # 纯红
        np.array([0.0, 1.0, 0.0]),  # 纯绿
        np.array([0.0, 0.0, 1.0]),  # 纯蓝
        np.array([1.0, 1.0, 1.0]),  # 白色
    ]
    
    # 创建结果文件
    with open('color_conversion_results.txt', 'w', encoding='utf-8') as f:
        f.write("色彩空间转换结果\n")
        f.write("================\n\n")
        
        # 写入转换矩阵
        f.write("1. 转换矩阵\n")
        f.write("BT2020原色矩阵:\n")
        f.write(str(BT2020_matrix))
        f.write("\n\n显示屏RGB原色矩阵:\n")
        f.write(str(DISPLAY_matrix))
        f.write("\n\n转换矩阵 (BT2020 -> 显示屏RGB):\n")
        f.write(str(conversion_matrix))
        f.write("\n\n")
        
        # 写入关键颜色转换结果
        f.write("2. 关键颜色转换测试\n")
        for color in test_colors:
            display_rgb = convert_color_bt2020_to_display(color)
            optimized_rgb = convert_bt2020_to_display(color)
            f.write(f"\nBT2020 RGB: {color}\n")
            f.write(f"简单转换后RGB: {display_rgb}\n")
            f.write(f"优化后RGB: {optimized_rgb}\n")
        
        # 写入色域面积分析
        f.write("\n3. 色域面积分析\n")
        bt2020_x = [BT2020_RED[0], BT2020_GREEN[0], BT2020_BLUE[0], BT2020_RED[0]]
        bt2020_y = [BT2020_RED[1], BT2020_GREEN[1], BT2020_BLUE[1], BT2020_RED[1]]
        display_x = [DISPLAY_RED[0], DISPLAY_GREEN[0], DISPLAY_BLUE[0], DISPLAY_RED[0]]
        display_y = [DISPLAY_RED[1], DISPLAY_GREEN[1], DISPLAY_BLUE[1], DISPLAY_RED[1]]
        
        def calculate_area(x, y):
            return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        
        bt2020_area = calculate_area(bt2020_x[:-1], bt2020_y[:-1])
        display_area = calculate_area(display_x[:-1], display_y[:-1])
        
        f.write(f"BT2020色域面积: {bt2020_area:.6f}\n")
        f.write(f"显示屏RGB色域面积: {display_area:.6f}\n")
        f.write(f"色域覆盖率: {(display_area/bt2020_area*100):.2f}%\n")

    print("\n色彩空间转换完成！结果已保存到 color_conversion_results.txt")

    # 分析转换精度
    results = analyze_conversion_accuracy()
    
    # 将结果保存到txt文件
    with open('accuracy_results_mlp.txt', 'w', encoding='utf-8') as f:
        
        f.write("颜色转换精度分析:\n")
        f.write("=" * 50 + "\n")
        f.write("颜色\t\t\t源RGBV\t\t\t目标rgb\t\t\t色差(ΔE)\n")
        f.write("-" * 100 + "\n")
        
        for i, result in enumerate(results):
            color = result['BT2020']
            target = result['target']
            error = result['error']
            f.write(f"颜色{i+1}\t({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})\t")
            f.write(f"({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})\t")
            f.write(f"{error:.4f}\n")
        
        # 计算统计信息
        errors = [r['error'] for r in results]
        avg_error = sum(errors) / len(errors)
        max_error = max(errors)
        min_error = min(errors)
        std_error = np.std(errors)
        
        f.write("\n统计信息:\n")
        f.write("=" * 50 + "\n")
        f.write(f"平均色差(ΔE): {avg_error:.4f}\n")
        f.write(f"最大色差(ΔE): {max_error:.4f}\n")
        f.write(f"最小色差(ΔE): {min_error:.4f}\n")
        f.write(f"色差标准差: {std_error:.4f}\n")

    # 可视化色彩映射
    mapping_fig = visualize_color_mapping()
    mapping_fig.savefig('color_mapping.png')

    # 比较颜色块
    patches_fig = compare_color_patches()
    patches_fig.savefig('color_patches.png')


if __name__ == "__main__":
    main()