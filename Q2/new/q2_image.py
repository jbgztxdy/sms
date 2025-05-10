import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Polygon
from scipy.optimize import minimize, linprog
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon as Polygon2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib as mpl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置是否是测试模式
TEST_MODE = True

###########################################################

# 定义4通道(RGBV)视频源的三基色坐标(CIE 1931坐标)
SOURCE_RED = (0.708, 0.292)
SOURCE_GREEN = (0.170, 0.797)
SOURCE_BLUE = (0.14, 0.046)
SOURCE_VIOLET = (0.03, 0.6)  # V通道(假设为紫色)

# 定义5通道(RGBCX)LED显示屏的三基色坐标
DISPLAY_RED = (0.6942, 0.3052)  # R通道
DISPLAY_GREEN = (0.2368, 0.7281)  # G通道
DISPLAY_BLUE = (0.1316, 0.0712)  # B通道
DISPLAY_CYAN = (0.04, 0.4)  # C通道(假设为青色)
DISPLAY_YELLOW = (0.1478, 0.7326)  # X通道(假设为黄色)

##########################################################
# 此处的三基色坐标根据自己电脑设备的实际情况进行调整，但是一般情况下不需要调整，因为这个已经是标准的了


def xy_to_XYZ(x, y, Y=1.0):
    """将CIE xy坐标转换为XYZ值"""
    if y == 0:
        return 0, 0, 0
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


def plot_color_spaces_4to5():
    """绘制CIE 1931色彩空间，以及4通道源和5通道显示的色域"""
    # 加载CIE 1931色度图数据(简化版)
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

    # 创建紫线(闭合马蹄形曲线)
    x = np.append(x, [0.1741])
    y = np.append(y, [0.0050])

    # 添加从蓝点到红点的直线(紫色线)
    purple_line_x = np.linspace(x[0], x[-2], 20)
    purple_line_y = np.linspace(y[0], y[-2], 20)

    plt.figure(figsize=(10, 8))

    # 绘制马蹄形曲线
    plt.plot(x, y, '-', color='black', label='光谱轨迹')
    plt.plot(purple_line_x, purple_line_y, '-', color='purple', label='紫线')

    # 绘制源4通道RGBV色域
    source_x = [SOURCE_RED[0], SOURCE_GREEN[0], SOURCE_VIOLET[0], SOURCE_BLUE[0], SOURCE_RED[0]]
    source_y = [SOURCE_RED[1], SOURCE_GREEN[1], SOURCE_VIOLET[1], SOURCE_BLUE[1], SOURCE_RED[1]]
    plt.plot(source_x, source_y, '-', color='blue', linewidth=2, label='4通道源RGBV')
    source_poly = Polygon(np.column_stack([source_x[:4], source_y[:4]]), alpha=0.2, color='blue')
    plt.gca().add_patch(source_poly)

    # 绘制显示屏5通道RGBCX色域
    display_x = [DISPLAY_RED[0], DISPLAY_GREEN[0], DISPLAY_YELLOW[0], DISPLAY_CYAN[0], DISPLAY_BLUE[0], DISPLAY_RED[0]]
    display_y = [DISPLAY_RED[1], DISPLAY_GREEN[1], DISPLAY_YELLOW[1], DISPLAY_CYAN[1], DISPLAY_BLUE[1], DISPLAY_RED[1]]
    plt.plot(display_x, display_y, '-', color='red', linewidth=2, label='5通道显示RGBCX')
    display_poly = Polygon(np.column_stack([display_x[:5], display_y[:5]]), alpha=0.2, color='red')
    plt.gca().add_patch(display_poly)

    # 标记RGB点
    plt.scatter([SOURCE_RED[0], SOURCE_GREEN[0], SOURCE_BLUE[0], SOURCE_VIOLET[0]],
                [SOURCE_RED[1], SOURCE_GREEN[1], SOURCE_BLUE[1], SOURCE_VIOLET[1]],
                color='blue', s=50)
    plt.text(SOURCE_RED[0], SOURCE_RED[1], 'R', fontsize=12)
    plt.text(SOURCE_GREEN[0], SOURCE_GREEN[1], 'G', fontsize=12)
    plt.text(SOURCE_BLUE[0], SOURCE_BLUE[1], 'B', fontsize=12)
    plt.text(SOURCE_VIOLET[0], SOURCE_VIOLET[1], 'V', fontsize=12)

    plt.scatter([DISPLAY_RED[0], DISPLAY_GREEN[0], DISPLAY_BLUE[0], DISPLAY_CYAN[0], DISPLAY_YELLOW[0]],
                [DISPLAY_RED[1], DISPLAY_GREEN[1], DISPLAY_BLUE[1], DISPLAY_CYAN[1], DISPLAY_YELLOW[1]],
                color='red', s=50)
    plt.text(DISPLAY_RED[0], DISPLAY_RED[1], 'R', fontsize=12)
    plt.text(DISPLAY_GREEN[0], DISPLAY_GREEN[1], 'G', fontsize=12)
    plt.text(DISPLAY_BLUE[0], DISPLAY_BLUE[1], 'B', fontsize=12)
    plt.text(DISPLAY_CYAN[0], DISPLAY_CYAN[1], 'C', fontsize=12)
    plt.text(DISPLAY_YELLOW[0], DISPLAY_YELLOW[1], 'X', fontsize=12)

    # 添加白点
    plt.scatter([0.3127], [0.3290], color='black', s=50, label='D65白点')
    plt.text(0.3127, 0.3290, 'D65', fontsize=12)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('CIE 1931色彩空间及4通道源与5通道显示色域对比')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 0.8)
    plt.ylim(0, 0.9)
    plt.tight_layout()

    return plt.gcf()


def compute_color_matrices():
    """计算4通道源和5通道显示的色彩转换矩阵"""
    # 4通道源原色的XYZ值
    SOURCE_R_XYZ = xy_to_XYZ(*SOURCE_RED)
    SOURCE_G_XYZ = xy_to_XYZ(*SOURCE_GREEN)
    SOURCE_B_XYZ = xy_to_XYZ(*SOURCE_BLUE)
    SOURCE_V_XYZ = xy_to_XYZ(*SOURCE_VIOLET)

    # 5通道显示原色的XYZ值
    DISPLAY_R_XYZ = xy_to_XYZ(*DISPLAY_RED)
    DISPLAY_G_XYZ = xy_to_XYZ(*DISPLAY_GREEN)
    DISPLAY_B_XYZ = xy_to_XYZ(*DISPLAY_BLUE)
    DISPLAY_C_XYZ = xy_to_XYZ(*DISPLAY_CYAN)
    DISPLAY_X_XYZ = xy_to_XYZ(*DISPLAY_YELLOW)

    # 构建4通道源原色矩阵
    SOURCE_matrix = np.array([
        [SOURCE_R_XYZ[0], SOURCE_G_XYZ[0], SOURCE_B_XYZ[0], SOURCE_V_XYZ[0]],
        [SOURCE_R_XYZ[1], SOURCE_G_XYZ[1], SOURCE_B_XYZ[1], SOURCE_V_XYZ[1]],
        [SOURCE_R_XYZ[2], SOURCE_G_XYZ[2], SOURCE_B_XYZ[2], SOURCE_V_XYZ[2]]
    ])

    # 构建5通道显示原色矩阵
    DISPLAY_matrix = np.array([
        [DISPLAY_R_XYZ[0], DISPLAY_G_XYZ[0], DISPLAY_B_XYZ[0], DISPLAY_C_XYZ[0], DISPLAY_X_XYZ[0]],
        [DISPLAY_R_XYZ[1], DISPLAY_G_XYZ[1], DISPLAY_B_XYZ[1], DISPLAY_C_XYZ[1], DISPLAY_X_XYZ[1]],
        [DISPLAY_R_XYZ[2], DISPLAY_G_XYZ[2], DISPLAY_B_XYZ[2], DISPLAY_C_XYZ[2], DISPLAY_X_XYZ[2]]
    ])

    return SOURCE_matrix, DISPLAY_matrix


def gamma_correction(linearRGB, gamma=2.2):
    """应用伽马校正，将线性RGB转换为sRGB"""
    return np.power(linearRGB, 1.0 / gamma)


def inverse_gamma_correction(sRGB, gamma=2.2):
    """应用反伽马校正，将sRGB转换为线性RGB"""
    return np.power(sRGB, gamma)


def optimize_4to5_conversion(source_rgbv):
    """优化4通道RGBV到5通道RGBCX的转换"""
    # 确保输入是线性RGB值
    source_rgbv_linear = source_rgbv.copy()

    # 转换4通道RGBV到XYZ
    SOURCE_matrix, DISPLAY_matrix = compute_color_matrices()
    source_xyz = np.dot(SOURCE_matrix, source_rgbv_linear)

    # 定义目标函数：最小化在XYZ空间中的欧氏距离
    def objective(display_rgbcx):
        # 确保RGBCX值在[0,1]范围内
        display_rgbcx_clipped = np.clip(display_rgbcx, 0, 1)
        # 转换5通道RGBCX到XYZ
        display_xyz = np.dot(DISPLAY_matrix, display_rgbcx_clipped)
        # 计算色差(欧氏距离)
        diff = np.sqrt(np.sum((source_xyz - display_xyz) ** 2))
        return diff

    # 初始猜测
    initial_guess = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # 平均分配

    # 优化
    bounds = [(0, 1)] * 5  # RGBCX值范围为[0,1]
    result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')

    # 返回优化后的5通道RGBCX值
    return np.clip(result.x, 0, 1)


def alternative_4to5_conversion(source_rgbv):
    """另一种4通道RGBV到5通道RGBCX的转换方法(使用线性规划)"""
    # 确保输入是线性RGB值
    source_rgbv_linear = source_rgbv.copy()

    # 转换4通道RGBV到XYZ
    SOURCE_matrix, DISPLAY_matrix = compute_color_matrices()
    source_xyz = np.dot(SOURCE_matrix, source_rgbv_linear)

    # 线性规划问题定义
    # 目标：最小化总RGB强度(以获得更好的能效)
    c = np.ones(5)  # 目标函数系数

    # 约束条件：DISPLAY_matrix * x = source_xyz
    A_eq = DISPLAY_matrix
    b_eq = source_xyz

    # 变量边界：0 <= x <= 1
    bounds = [(0, 1)] * 5

    # 求解线性规划问题
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    # 返回优化后的5通道RGBCX值
    if result.success:
        return np.clip(result.x, 0, 1)
    else:
        # 如果线性规划失败，回退到欧氏距离优化
        return optimize_4to5_conversion(source_rgbv)


def visualize_4to5_mapping():
    """可视化4通道到5通道的色彩映射"""
    # 创建一些测试颜色
    test_colors = [
        (1.0, 0.0, 0.0, 0.0),  # 纯R
        (0.0, 1.0, 0.0, 0.0),  # 纯G
        (0.0, 0.0, 1.0, 0.0),  # 纯B
        (0.0, 0.0, 0.0, 1.0),  # 纯V
        (0.5, 0.5, 0.0, 0.0),  # RG混合
        (0.5, 0.0, 0.5, 0.0),  # RB混合
        (0.0, 0.5, 0.5, 0.0),  # GB混合
        (0.0, 0.0, 0.5, 0.5),  # BV混合
        (0.5, 0.0, 0.0, 0.5),  # RV混合
        (0.0, 0.5, 0.0, 0.5),  # GV混合
        (0.25, 0.25, 0.25, 0.25),  # 均匀混合
        (0.5, 0.3, 0.1, 0.1),  # 复杂混合1
        (0.2, 0.4, 0.2, 0.2),  # 复杂混合2
        (0.1, 0.1, 0.4, 0.4),  # 复杂混合3
        (0.3, 0.2, 0.3, 0.2),  # 复杂混合4
    ]

    nn_mapped_colors = [convert_SOURCE_to_display(np.array(color)) for color in test_colors]
    
    # 其他映射方法
    mapped_colors = [optimize_4to5_conversion(np.array(color)) for color in test_colors]
    alt_mapped_colors = [alternative_4to5_conversion(np.array(color)) for color in test_colors]

    # 将4通道颜色映射到RGB显示空间(为了可视化)
    SOURCE_matrix, _ = compute_color_matrices()

    # 定义显示矩阵(简单RGB显示,用于可视化)
    DISPLAY_RGB_matrix = np.array([
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570]
    ])

    # 计算每个源颜色的XYZ值
    source_xyz_values = [np.dot(SOURCE_matrix, color) for color in test_colors]

    # 将XYZ转换为RGB(用于显示)
    source_rgb_values = [np.dot(np.linalg.inv(DISPLAY_RGB_matrix), xyz) for xyz in source_xyz_values]
    source_rgb_values = [np.clip(rgb, 0, 1) for rgb in source_rgb_values]

    # 应用伽马校正(为了正确显示)
    source_rgb_values = [gamma_correction(rgb) for rgb in source_rgb_values]

    # 创建显示图 - 修改为4列以包含神经网络映射
    fig, axes = plt.subplots(len(test_colors), 4, figsize=(20, 2 * len(test_colors)))
    fig.suptitle('4通道RGBV源到5通道RGBCX显示的映射对比', fontsize=16)

    for i, (original_rgbv, nn_rgbcx, mapped_rgbcx, alt_mapped_rgbcx, display_rgb) in enumerate(
            zip(test_colors, nn_mapped_colors, mapped_colors, alt_mapped_colors, source_rgb_values)):
        # 显示原始4通道颜色(用RGB近似显示)
        axes[i, 0].add_patch(plt.Rectangle((0, 0), 1, 1, color=display_rgb))
        axes[i, 0].set_xlim(0, 1)
        axes[i, 0].set_ylim(0, 1)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        axes[i, 0].set_title(
            f'源RGBV ({original_rgbv[0]:.1f}, {original_rgbv[1]:.1f}, {original_rgbv[2]:.1f}, {original_rgbv[3]:.1f})')

        # 显示神经网络映射的5通道颜色
        axes[i, 1].bar(['R', 'G', 'B', 'C', 'X'], nn_rgbcx, color=['red', 'green', 'blue', 'cyan', 'yellow'])
        axes[i, 1].set_ylim(0, 1)
        axes[i, 1].set_title('神经网络映射')

        # 显示欧氏距离优化映射的5通道颜色
        axes[i, 2].bar(['R', 'G', 'B', 'C', 'X'], mapped_rgbcx, color=['red', 'green', 'blue', 'cyan', 'yellow'])
        axes[i, 2].set_ylim(0, 1)
        axes[i, 2].set_title('欧氏距离优化映射')

        # 显示线性规划映射的5通道颜色
        axes[i, 3].bar(['R', 'G', 'B', 'C', 'X'], alt_mapped_rgbcx, color=['red', 'green', 'blue', 'cyan', 'yellow'])
        axes[i, 3].set_ylim(0, 1)
        axes[i, 3].set_title('线性规划映射')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return plt.gcf()


def visualize_conversion_accuracy():
    """可视化转换精度"""
    # 创建随机测试颜色
    np.random.seed(42)  # 为了重复性
    n_samples = 100
    test_colors = np.random.random((n_samples, 4))

    nn_mapped_colors = [convert_SOURCE_to_display(color) for color in test_colors]
    mapped_colors = [optimize_4to5_conversion(color) for color in test_colors]
    alt_mapped_colors = [alternative_4to5_conversion(color) for color in test_colors]

    # 计算转换误差
    SOURCE_matrix, DISPLAY_matrix = compute_color_matrices()

    errors_nn = []
    errors_opt = []
    errors_alt = []

    for i in range(n_samples):
        source_xyz = np.dot(SOURCE_matrix, test_colors[i])

        # 神经网络方法
        nn_mapped_xyz = np.dot(DISPLAY_matrix, nn_mapped_colors[i])
        error_nn = np.sqrt(np.sum((source_xyz - nn_mapped_xyz) ** 2))
        errors_nn.append(error_nn)

        # 欧氏距离优化方法
        mapped_xyz = np.dot(DISPLAY_matrix, mapped_colors[i])
        error_opt = np.sqrt(np.sum((source_xyz - mapped_xyz) ** 2))
        errors_opt.append(error_opt)

        # 线性规划方法
        alt_mapped_xyz = np.dot(DISPLAY_matrix, alt_mapped_colors[i])
        error_alt = np.sqrt(np.sum((source_xyz - alt_mapped_xyz) ** 2))
        errors_alt.append(error_alt)

    # 绘制误差分布 - 修改为3列
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 3, 1)
    plt.hist(errors_nn, bins=20, alpha=0.7, color='purple')
    plt.axvline(np.mean(errors_nn), color='red', linestyle='dashed', linewidth=2)
    plt.title(f'神经网络方法误差分布\n平均误差: {np.mean(errors_nn):.4f}')
    plt.xlabel('XYZ空间中的欧氏距离误差')
    plt.ylabel('频数')

    plt.subplot(1, 3, 2)
    plt.hist(errors_opt, bins=20, alpha=0.7, color='blue')
    plt.axvline(np.mean(errors_opt), color='red', linestyle='dashed', linewidth=2)
    plt.title(f'欧氏距离优化方法误差分布\n平均误差: {np.mean(errors_opt):.4f}')
    plt.xlabel('XYZ空间中的欧氏距离误差')
    plt.ylabel('频数')

    plt.subplot(1, 3, 3)
    plt.hist(errors_alt, bins=20, alpha=0.7, color='green')
    plt.axvline(np.mean(errors_alt), color='red', linestyle='dashed', linewidth=2)
    plt.title(f'线性规划方法误差分布\n平均误差: {np.mean(errors_alt):.4f}')
    plt.xlabel('XYZ空间中的欧氏距离误差')
    plt.ylabel('频数')

    plt.tight_layout()
    return plt.gcf()


def visualize_3d_color_spaces():
    """以3D方式可视化XYZ色彩空间中的4通道和5通道色域"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 计算4通道源的XYZ顶点
    SOURCE_R_XYZ = xy_to_XYZ(*SOURCE_RED)
    SOURCE_G_XYZ = xy_to_XYZ(*SOURCE_GREEN)
    SOURCE_B_XYZ = xy_to_XYZ(*SOURCE_BLUE)
    SOURCE_V_XYZ = xy_to_XYZ(*SOURCE_VIOLET)

    # 计算5通道显示的XYZ顶点
    DISPLAY_R_XYZ = xy_to_XYZ(*DISPLAY_RED)
    DISPLAY_G_XYZ = xy_to_XYZ(*DISPLAY_GREEN)
    DISPLAY_B_XYZ = xy_to_XYZ(*DISPLAY_BLUE)
    DISPLAY_C_XYZ = xy_to_XYZ(*DISPLAY_CYAN)
    DISPLAY_X_XYZ = xy_to_XYZ(*DISPLAY_YELLOW)

    # 绘制4通道源的色域四面体
    source_vertices = np.array([
        [SOURCE_R_XYZ[0], SOURCE_R_XYZ[1], SOURCE_R_XYZ[2]],
        [SOURCE_G_XYZ[0], SOURCE_G_XYZ[1], SOURCE_G_XYZ[2]],
        [SOURCE_B_XYZ[0], SOURCE_B_XYZ[1], SOURCE_B_XYZ[2]],
        [SOURCE_V_XYZ[0], SOURCE_V_XYZ[1], SOURCE_V_XYZ[2]]
    ])

    # 定义四面体的面
    source_faces = [
        [0, 1, 2],  # R-G-B
        [0, 1, 3],  # R-G-V
        [0, 2, 3],  # R-B-V
        [1, 2, 3]  # G-B-V
    ]

    # 绘制四面体
    for face in source_faces:
        vertices = source_vertices[face]
        ax.add_collection3d(Poly3DCollection(
            [vertices], alpha=0.3, color='blue'))

    # 绘制4通道源的边
    edges = [
        [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]
    ]
    for edge in edges:
        line = np.array([source_vertices[edge[0]], source_vertices[edge[1]]])
        ax.plot3D(line[:, 0], line[:, 1], line[:, 2], 'blue')

    # 绘制4通道源的顶点
    ax.scatter(source_vertices[:, 0], source_vertices[:, 1], source_vertices[:, 2],
               color='blue', s=50, label='4通道源RGBV')

    # 添加标签
    ax.text(SOURCE_R_XYZ[0], SOURCE_R_XYZ[1], SOURCE_R_XYZ[2], 'R', color='blue')
    ax.text(SOURCE_G_XYZ[0], SOURCE_G_XYZ[1], SOURCE_G_XYZ[2], 'G', color='blue')
    ax.text(SOURCE_B_XYZ[0], SOURCE_B_XYZ[1], SOURCE_B_XYZ[2], 'B', color='blue')
    ax.text(SOURCE_V_XYZ[0], SOURCE_V_XYZ[1], SOURCE_V_XYZ[2], 'V', color='blue')

    # 绘制5通道显示的色域
    display_vertices = np.array([
        [DISPLAY_R_XYZ[0], DISPLAY_R_XYZ[1], DISPLAY_R_XYZ[2]],
        [DISPLAY_G_XYZ[0], DISPLAY_G_XYZ[1], DISPLAY_G_XYZ[2]],
        [DISPLAY_B_XYZ[0], DISPLAY_B_XYZ[1], DISPLAY_B_XYZ[2]],
        [DISPLAY_C_XYZ[0], DISPLAY_C_XYZ[1], DISPLAY_C_XYZ[2]],
        [DISPLAY_X_XYZ[0], DISPLAY_X_XYZ[1], DISPLAY_X_XYZ[2]]
    ])

    # 绘制5通道显示的顶点
    ax.scatter(display_vertices[:, 0], display_vertices[:, 1], display_vertices[:, 2],
               color='red', s=50, label='5通道显示RGBCX')

    # 添加标签
    ax.text(DISPLAY_R_XYZ[0], DISPLAY_R_XYZ[1], DISPLAY_R_XYZ[2], 'R', color='red')
    ax.text(DISPLAY_G_XYZ[0], DISPLAY_G_XYZ[1], DISPLAY_G_XYZ[2], 'G', color='red')
    ax.text(DISPLAY_B_XYZ[0], DISPLAY_B_XYZ[1], DISPLAY_B_XYZ[2], 'B', color='red')
    ax.text(DISPLAY_C_XYZ[0], DISPLAY_C_XYZ[1], DISPLAY_C_XYZ[2], 'C', color='red')
    ax.text(DISPLAY_X_XYZ[0], DISPLAY_X_XYZ[1], DISPLAY_X_XYZ[2], 'X', color='red')

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D XYZ色彩空间中的4通道和5通道色域')

    # 添加图例
    ax.legend()

    # 设置视角
    ax.view_init(elev=20, azim=45)

    return plt.gcf()

def calculate_gamut_coverage():
    """计算5通道显示对4通道源的色域覆盖率"""
    # 计算4通道源的色域面积
    source_points = np.array([
        [SOURCE_RED[0], SOURCE_RED[1]],
        [SOURCE_GREEN[0], SOURCE_GREEN[1]],
        [SOURCE_BLUE[0], SOURCE_BLUE[1]],
        [SOURCE_VIOLET[0], SOURCE_VIOLET[1]]
    ])
    
    # 计算5通道显示的色域面积
    display_points = np.array([
        [DISPLAY_RED[0], DISPLAY_RED[1]],
        [DISPLAY_GREEN[0], DISPLAY_GREEN[1]],
        [DISPLAY_BLUE[0], DISPLAY_BLUE[1]],
        [DISPLAY_CYAN[0], DISPLAY_CYAN[1]],
        [DISPLAY_YELLOW[0], DISPLAY_YELLOW[1]]
    ])
    
    # 计算多边形面积
    def polygon_area(points):
        x = points[:, 0]
        y = points[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    source_area = polygon_area(source_points)
    display_area = polygon_area(display_points)
    
    # 计算覆盖率
    coverage = min(display_area / source_area, 1.0) * 100
    
    return coverage, source_area, display_area

def analyze_conversion_accuracy():
    """分析4通道到5通道转换的精度"""
    # 创建测试颜色集
    test_colors = [
        np.array([1.0, 0.0, 0.0, 0.0]),  # 纯R
        np.array([0.0, 1.0, 0.0, 0.0]),  # 纯G
        np.array([0.0, 0.0, 1.0, 0.0]),  # 纯B
        np.array([0.0, 0.0, 0.0, 1.0]),  # 纯V
        np.array([0.5, 0.5, 0.0, 0.0]),  # RG混合
        np.array([0.5, 0.0, 0.5, 0.0]),  # RB混合
        np.array([0.0, 0.5, 0.5, 0.0]),  # GB混合
        np.array([0.0, 0.0, 0.5, 0.5]),  # BV混合
        np.array([0.5, 0.0, 0.0, 0.5]),  # RV混合
        np.array([0.0, 0.5, 0.0, 0.5]),  # GV混合
        np.array([0.25, 0.25, 0.25, 0.25]),  # 均匀混合
        np.array([0.5, 0.3, 0.1, 0.1]),  # 复杂混合1
        np.array([0.2, 0.4, 0.2, 0.2]),  # 复杂混合2
        np.array([0.1, 0.1, 0.4, 0.4]),  # 复杂混合3
        np.array([0.3, 0.2, 0.3, 0.2]),  # 复杂混合4
    ]

    # 获取颜色矩阵
    SOURCE_matrix, DISPLAY_matrix = compute_color_matrices()

    # 存储结果
    results = []
    total_error = 0
    max_error = 0
    min_error = float('inf')

    print("\n转换精度分析结果：")
    print("=" * 50)
    print("颜色\t\t\t源RGBV\t\t\t目标RGBCX\t\t\t色差(ΔE)")
    print("-" * 100)

    for i, color in enumerate(test_colors):
        # 转换到5通道
        display_rgbcx = convert_SOURCE_to_display(color)

        # 计算原始颜色和目标颜色的XYZ值
        source_xyz = np.dot(SOURCE_matrix, color)
        display_xyz = np.dot(DISPLAY_matrix, display_rgbcx)

        # 计算色差(ΔE)
        error = np.sqrt(np.sum((source_xyz - display_xyz) ** 2))
        total_error += error
        max_error = max(max_error, error)
        min_error = min(min_error, error)

        # 格式化输出
        color_name = f"颜色{i+1}"
        source_str = f"({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f}, {color[3]:.2f})"
        target_str = f"({display_rgbcx[0]:.2f}, {display_rgbcx[1]:.2f}, {display_rgbcx[2]:.2f}, {display_rgbcx[3]:.2f}, {display_rgbcx[4]:.2f})"
        print(f"{color_name}\t{source_str}\t{target_str}\t{error:.4f}")

        results.append({
            'source': color,
            'target': display_rgbcx,
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

SOURCE_matrix, DISPLAY_matrix = compute_color_matrices()

# # 定义 MLP 模型，使用 Hardtanh 激活函数, 保证输出为[0,1]
# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         # 增加第一个隐藏层神经元数量
#         self.fc1 = nn.Linear(4, 32)
#         self.relu1 = nn.ReLU()
#         # 添加第二个隐藏层
#         self.fc2 = nn.Linear(32, 64)
#         self.relu2 = nn.ReLU()
#         # 添加第三个隐藏层
#         self.fc3 = nn.Linear(64, 32)
#         self.relu3 = nn.ReLU()
#         self.fc4 = nn.Linear(32, 5)
#         self.hardtanh = nn.Hardtanh()

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu1(x)
#         x = self.fc2(x)
#         x = self.relu2(x)
#         x = self.fc3(x)
#         x = self.relu3(x)
#         x = self.fc4(x)
#         # 将 Hardtanh 的输出从 [-1, 1] 映射到 [0, 1]
#         x = (self.hardtanh(x) + 1) / 2
#         return x

class ChromaticAttention(nn.Module):
    """光谱注意力机制"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c = x.size()
        y = self.avg_pool(x.unsqueeze(-1)).view(b, c)
        y = self.fc(y).view(b, c)
        return x * y.expand_as(x)

class GamutProjection(nn.Module):
    """可微分色域投影层（最终正确版）"""
    def __init__(self):
        super().__init__()
        # 可学习的5x5投影矩阵
        self.proj_matrix = nn.Parameter(torch.eye(5))  # 初始化为单位矩阵
        
    def forward(self, x):
        # 保持5通道维度
        x = torch.mm(x, self.proj_matrix)  # [batch,5] x [5,5] → [batch,5]
        return torch.sigmoid(x)

class MLP(nn.Module):
    def __init__(self, input_dim=4, output_dim=5, hidden_dim=512):
        super().__init__()
        # 输入编码层
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 特征转换层
        self.transformer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # 输出解码层
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 色域投影层（显示设备矩阵应为3x5）
        self.gamut_layer = GamutProjection()  # 直接传入原始矩阵
        
    def forward(self, x):
        # 输入标准化
        x = x.clamp(0, 1)
        
        # 特征编码
        x = self.encoder(x)
        
        # 特征变换
        x = self.transformer(x) + x  # 残差连接
        
        # 解码输出（5通道）
        x = self.decoder(x)
        
        # 色域投影（保持5通道）
        return self.gamut_layer(x).clamp(0, 1)

class ChromaticLoss(nn.Module):
    """混合感知损失函数"""
    def __init__(self, alpha=0.6, beta=0.4):
        super().__init__()
        self.alpha = alpha  # XYZ空间权重
        self.beta = beta    # CIEDE2000权重
        
    def forward(self, source, output):
        # 计算输入和输出的 xy 值
        source_xy = SOURCErgb_to_xy(source)
        output_xy = displayrgb_to_xy(output)

        # 计算 CIEDE2000 色差
        # 将xy转换为XYZ (假设Y=1)
        source_XYZ = torch.stack([
            source_xy[..., 0] * 1 / source_xy[..., 1],
            torch.ones_like(source_xy[..., 0]),
            (1 - source_xy[..., 0] - source_xy[..., 1]) * 1 / source_xy[..., 1]
        ], dim=-1)
        
        output_XYZ = torch.stack([
            output_xy[..., 0] * 1 / output_xy[..., 1],
            torch.ones_like(output_xy[..., 0]),
            (1 - output_xy[..., 0] - output_xy[..., 1]) * 1 / output_xy[..., 1]
        ], dim=-1)
        
        # 将XYZ转换为Lab
        source_Lab = XYZ_to_Lab(source_XYZ)
        output_Lab = XYZ_to_Lab(output_XYZ)
        xyz_loss = F.mse_loss(source_xy, output_xy)
        
        de_loss = delta_e_cie2000_torch(source_Lab, output_Lab).mean()
        
        # 混合损失
        return self.alpha*xyz_loss + self.beta*de_loss
def SOURCErgb_to_xy(rgb: torch.Tensor) -> torch.Tensor:
    """
    将SOURCE RGB颜色值转换为CIE xy色度坐标
    参数:
        rgb: 形状为 (..., 4) 的PyTorch张量，表示RGB颜色值
    返回:
        xy: 形状为 (..., 2) 的PyTorch张量，表示xy色度坐标
    """
    # 确保输入是正确的形状
    if rgb.dim() == 1:
        rgb = rgb.unsqueeze(0)  # 添加批次维度
    
    # 将矩阵转换为PyTorch张量
    SOURCE_matrix_tensor = torch.tensor(SOURCE_matrix, dtype=rgb.dtype, device=rgb.device)
    xyz = torch.matmul(rgb, SOURCE_matrix_tensor.T)  # (..., 3)
    
    # 计算xy坐标
    sum_xyz = torch.sum(xyz, dim=-1, keepdim=True)  # (..., 1)
    x = xyz[..., 0:1] / (sum_xyz + 1e-8)  # (..., 1)
    y = xyz[..., 1:2] / (sum_xyz + 1e-8)  # (..., 1)
    
    return torch.cat([x, y], dim=-1)  # (..., 2)

def displayrgb_to_xy(rgb: torch.Tensor) -> torch.Tensor:
    """
    将显示屏RGB颜色值转换为CIE xy色度坐标
    参数:
        rgb: 形状为 (..., 5) 的PyTorch张量，表示RGB颜色值
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

def generate_training_data(num_samples=10000):
    """改进的训练数据生成策略"""
    samples = []
    
    # 策略1：色域边界采样
    for _ in range(num_samples//4):
        base = np.random.rand(3)
        base /= np.sum(base)
        samples.append(np.append(base, np.random.rand()))
    
    # 策略2：通道极值采样
    for _ in range(num_samples//4):
        vec = np.zeros(4)
        vec[np.random.choice(4)] = 1.0
        samples.append(vec)
    
    # 策略3：色域内均匀采样
    samples.extend(np.random.dirichlet(np.ones(4), size=num_samples//2))
    
    # 添加噪声增强
    samples = np.clip(samples + np.random.normal(0, 0.05, (num_samples,4)), 0, 1)
    return torch.tensor(samples, dtype=torch.float32)
# # 生成训练数据
# num_samples = 1000
# SOURCE_rgb_train = np.random.uniform(0, 1, (num_samples, 4))
# SOURCE_rgb_train = torch.tensor(SOURCE_rgb_train, dtype=torch.float32).to(device)

# # 创建数据集和数据加载器
# batch_size = 100  # 小批量的大小
# dataset = TensorDataset(SOURCE_rgb_train)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 定义颜色转换函数
def convert_SOURCE_to_display(source_rgb):
    # 初始化 MLP 模型并移动到 GPU
    model = MLP().to(device)
    model.load_state_dict(torch.load('mlp_model.pth', map_location=device))
    model.eval()
    
    # 将输入转换为2D张量 (添加batch维度)
    source_rgb_tensor = torch.tensor(source_rgb, dtype=torch.float32).to(device)
    
    # 确保输入是2D张量：(batch_size=1, num_features)
    if source_rgb_tensor.dim() == 1:
        source_rgb_tensor = source_rgb_tensor.unsqueeze(0)  # 添加batch维度
    
    with torch.no_grad():  # 禁用梯度计算，提高推理速度
        display_rgb_tensor = model(source_rgb_tensor)
    
    # 将输出转换回NumPy数组
    display_rgb = display_rgb_tensor.cpu().detach().numpy()
    
    # 如果是单个样本，移除batch维度
    if display_rgb.shape[0] == 1:
        display_rgb = display_rgb[0]
    
    # 确保输出在0-1之间
    display_rgb = np.clip(display_rgb, 0, 1)
    return display_rgb

def test_mlp_model(rgb_values, model_path='mlp_model.pth'):
    """
    加载保存的模型参数并处理输入的 RGB 值。
    :param rgb_values: 输入的 RGB 值，形状为 (n, 3) 的 numpy 数组或列表
    :param model_path: 保存的模型参数文件路径
    :return: 处理后的 RGB 值，形状为 (n, 3) 的 numpy 数组
    """
    def convert(source_rgb, mlpmodel):
        source_rgb_tensor = torch.tensor(source_rgb, dtype=torch.float32)
        display_rgb_tensor = mlpmodel(source_rgb_tensor)
        display_rgb = display_rgb_tensor.detach().numpy()
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

def visualize_color_mapping():
    """可视化色彩映射效果"""
    # 创建SOURCE色域内的采样点
    samples = 10  # 每个维度的采样数
    r_values = np.linspace(0, 1, samples)
    g_values = np.linspace(0, 1, samples)
    b_values = np.linspace(0, 1, samples)
    v_values = np.linspace(0, 1, samples)

    # 存储转换前后的颜色
    original_colors = []
    mapped_colors = []

    # 对于每个采样点
    for r in r_values:
        for g in g_values:
            for b in b_values:
                for v in v_values:
                    SOURCE_rgbv = np.array([r, g, b, v])
                    # 优化转换
                    optimized_display_rgbv = convert_SOURCE_to_display(SOURCE_rgbv[:4])

                    # 添加到颜色列表
                    original_colors.append(SOURCE_rgbv)
                    # 使用优化的结果
                    mapped_colors.append(optimized_display_rgbv)

    # 转换为数组
    original_colors = np.array(original_colors)
    mapped_colors = np.array(mapped_colors)

    # 计算转换前后的xy坐标
    SOURCE_matrix, DISPLAY_matrix = compute_color_matrices()

    original_xyz = np.array([np.dot(SOURCE_matrix, rgb) for rgb in original_colors[:, :4]])
    mapped_xyz = np.array([np.dot(DISPLAY_matrix, rgb) for rgb in mapped_colors[:, :5]])

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

    # 绘制源4通道RGBV色域
    source_x = [SOURCE_RED[0], SOURCE_GREEN[0], SOURCE_VIOLET[0], SOURCE_BLUE[0], SOURCE_RED[0]]
    source_y = [SOURCE_RED[1], SOURCE_GREEN[1], SOURCE_VIOLET[1], SOURCE_BLUE[1], SOURCE_RED[1]]
    plt.plot(source_x, source_y, '-', color='brown', linewidth=2, label='4通道源RGBV')
    source_poly = Polygon(np.column_stack([source_x[:4], source_y[:4]]), alpha=0.2, color='blue')
    plt.gca().add_patch(source_poly)

    # 绘制显示屏5通道RGBCX色域
    display_x = [DISPLAY_RED[0], DISPLAY_GREEN[0], DISPLAY_YELLOW[0], DISPLAY_CYAN[0], DISPLAY_BLUE[0], DISPLAY_RED[0]]
    display_y = [DISPLAY_RED[1], DISPLAY_GREEN[1], DISPLAY_YELLOW[1], DISPLAY_CYAN[1], DISPLAY_BLUE[1], DISPLAY_RED[1]]
    plt.plot(display_x, display_y, '-', color='red', linewidth=2, label='5通道显示RGBCX')
    display_poly = Polygon(np.column_stack([display_x[:5], display_y[:5]]), alpha=0.2, color='red')
    plt.gca().add_patch(display_poly)

    # 绘制转换前后的点
    plt.scatter(original_xy[:, 0], original_xy[:, 1], c='blue', alpha=0.5, s=20, label='SOURCE 原始颜色')
    plt.scatter(mapped_xy[:, 0], mapped_xy[:, 1], c='green', alpha=0.5, s=20, label='映射后颜色')

    # 绘制转换对应关系(绘制一些样本点的对应关系，避免过于拥挤)
    for i in range(0, len(original_xy), len(original_xy) // 20):
        plt.plot([original_xy[i, 0], mapped_xy[i, 0]],
                 [original_xy[i, 1], mapped_xy[i, 1]],
                 'k-', alpha=0.2)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('SOURCE到显示屏RGB的颜色映射可视化')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 0.8)
    plt.ylim(0, 0.9)
    plt.tight_layout()

    return plt.gcf()

def main():
    """主函数，运行所有可视化和分析"""
    print("开始分析...")
    
    # 计算色域覆盖率
    coverage, source_area, display_area = calculate_gamut_coverage()
    
    # 分析转换精度
    results = analyze_conversion_accuracy()
    
    # 将结果保存到txt文件
    with open('analysis_results_mlp.txt', 'w', encoding='utf-8') as f:
        f.write("色域分析结果:\n")
        f.write("=" * 50 + "\n")
        f.write(f"4通道源色域面积: {source_area:.4f}\n")
        f.write(f"5通道显示色域面积: {display_area:.4f}\n")
        f.write(f"色域覆盖率: {coverage:.2f}%\n\n")
        
        f.write("颜色转换精度分析:\n")
        f.write("=" * 50 + "\n")
        f.write("颜色\t\t\t源RGBV\t\t\t目标RGBCX\t\t\t色差(ΔE)\n")
        f.write("-" * 100 + "\n")
        
        for i, result in enumerate(results):
            color = result['source']
            target = result['target']
            error = result['error']
            f.write(f"颜色{i+1}\t({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f}, {color[3]:.2f})\t")
            f.write(f"({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}, {target[3]:.2f}, {target[4]:.2f})\t")
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
    
    print("\n数值结果已保存到 analysis_results_mlp.txt")
    
    # 绘制CIE 1931色彩空间和色域对比
    print("\n生成可视化图表...")
    fig1 = plot_color_spaces_4to5()
    fig1.savefig('color_spaces_4to5.png')
    plt.close(fig1)

    # 可视化4通道到5通道的映射
    fig2 = visualize_4to5_mapping()
    fig2.savefig('color_mapping_4to5.png')
    plt.close(fig2)

    # 可视化转换精度
    fig3 = visualize_conversion_accuracy()
    fig3.savefig('conversion_accuracy.png')
    plt.close(fig3)

    # 3D可视化色彩空间
    fig4 = visualize_3d_color_spaces()
    fig4.savefig('3d_color_spaces.png')
    plt.close(fig4)

    # 可视化色彩映射
    mapping_fig = visualize_color_mapping()
    mapping_fig.savefig('color_mapping_mlp.png')

    print("\n所有分析已完成，结果已保存为PNG文件。")

if __name__ == "__main__":
    main()