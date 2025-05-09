import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Polygon
from scipy.optimize import minimize
from colormath.color_objects import XYZColor, sRGBColor
from colormath.color_conversions import convert_color
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

    # # 添加白点
    # plt.scatter([0.3127, 0.3127], [0.3290, 0.3290], color='black', s=50, label='D65白点')

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

    print("\n原色矩阵分析:")
    print("BT2020原色矩阵:")
    print(BT2020_matrix)
    print("\n显示屏RGB原色矩阵:")
    print(DISPLAY_matrix)
    print("\n转换矩阵 (BT2020 -> 显示屏RGB):")
    print(conversion_matrix)

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
                optimized_display_rgb = optimize_conversion(bt2020_rgb)

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
    mapped_colors = [optimize_conversion(np.array(color)) for color in test_colors]

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
            optimized_rgb = optimize_conversion(color)
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

    # 可视化色彩映射
    mapping_fig = visualize_color_mapping()
    mapping_fig.savefig('color_mapping.png')

    # 比较颜色块
    patches_fig = compare_color_patches()
    patches_fig.savefig('color_patches.png')


if __name__ == "__main__":
    main()