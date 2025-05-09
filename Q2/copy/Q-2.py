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

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

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

    # 进行颜色转换
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

    # 创建显示图
    fig, axes = plt.subplots(len(test_colors), 3, figsize=(15, 2 * len(test_colors)))
    fig.suptitle('4通道RGBV源到5通道RGBCX显示的映射对比', fontsize=16)

    for i, (original_rgbv, mapped_rgbcx, alt_mapped_rgbcx, display_rgb) in enumerate(
            zip(test_colors, mapped_colors, alt_mapped_colors, source_rgb_values)):
        # 显示原始4通道颜色(用RGB近似显示)
        axes[i, 0].add_patch(plt.Rectangle((0, 0), 1, 1, color=display_rgb))
        axes[i, 0].set_xlim(0, 1)
        axes[i, 0].set_ylim(0, 1)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        axes[i, 0].set_title(
            f'源RGBV ({original_rgbv[0]:.1f}, {original_rgbv[1]:.1f}, {original_rgbv[2]:.1f}, {original_rgbv[3]:.1f})')

        # 显示欧氏距离优化映射的5通道颜色
        axes[i, 1].bar(['R', 'G', 'B', 'C', 'X'], mapped_rgbcx, color=['red', 'green', 'blue', 'cyan', 'yellow'])
        axes[i, 1].set_ylim(0, 1)
        axes[i, 1].set_title('欧氏距离优化映射')

        # 显示线性规划映射的5通道颜色
        axes[i, 2].bar(['R', 'G', 'B', 'C', 'X'], alt_mapped_rgbcx, color=['red', 'green', 'blue', 'cyan', 'yellow'])
        axes[i, 2].set_ylim(0, 1)
        axes[i, 2].set_title('线性规划映射')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return plt.gcf()


def visualize_conversion_accuracy():
    """可视化转换精度"""
    # 创建随机测试颜色
    np.random.seed(42)  # 为了重复性
    n_samples = 100
    test_colors = np.random.random((n_samples, 4))

    # 进行两种方法的颜色转换
    mapped_colors = [optimize_4to5_conversion(color) for color in test_colors]
    alt_mapped_colors = [alternative_4to5_conversion(color) for color in test_colors]

    # 计算转换误差
    SOURCE_matrix, DISPLAY_matrix = compute_color_matrices()

    errors_opt = []
    errors_alt = []

    for i in range(n_samples):
        source_xyz = np.dot(SOURCE_matrix, test_colors[i])

        # 欧氏距离优化方法
        mapped_xyz = np.dot(DISPLAY_matrix, mapped_colors[i])
        error_opt = np.sqrt(np.sum((source_xyz - mapped_xyz) ** 2))
        errors_opt.append(error_opt)

        # 线性规划方法
        alt_mapped_xyz = np.dot(DISPLAY_matrix, alt_mapped_colors[i])
        error_alt = np.sqrt(np.sum((source_xyz - alt_mapped_xyz) ** 2))
        errors_alt.append(error_alt)

    # 绘制误差分布
    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    plt.hist(errors_opt, bins=20, alpha=0.7, color='blue')
    plt.axvline(np.mean(errors_opt), color='red', linestyle='dashed', linewidth=2)
    plt.title(f'欧氏距离优化方法误差分布\n平均误差: {np.mean(errors_opt):.4f}')
    plt.xlabel('XYZ空间中的欧氏距离误差')
    plt.ylabel('频数')

    plt.subplot(1, 2, 2)
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
        display_rgbcx = optimize_4to5_conversion(color)

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

def main():
    """主函数，运行所有可视化和分析"""
    print("开始分析...")
    
    # 计算色域覆盖率
    coverage, source_area, display_area = calculate_gamut_coverage()
    
    # 分析转换精度
    results = analyze_conversion_accuracy()
    
    # 将结果保存到txt文件
    with open('analysis_results.txt', 'w', encoding='utf-8') as f:
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
    
    print("\n数值结果已保存到 analysis_results.txt")
    
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

    print("\n所有分析已完成，结果已保存为PNG文件。")

if __name__ == "__main__":
    main()