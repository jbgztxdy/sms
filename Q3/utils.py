import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号


# 函数：计算两个RGB颜色之间的色度差异（简化为欧氏距离）
def color_difference(rgb1, rgb2):
    return np.linalg.norm(np.array(rgb1) - np.array(rgb2))


# 计算感知色差 ΔE，传入两个 RGB 颜色
def calculate_delta_e(rgb1, rgb2):
    lab1 = convert_color(sRGBColor(*rgb1, is_upscaled=True), LabColor)
    lab2 = convert_color(sRGBColor(*rgb2, is_upscaled=True), LabColor)
    return delta_e_cie2000(lab1, lab2)


# 加载LED显示屏数据：64*64*9，data[i, j, 0] = R_R
def load_led_data(file_path) -> np.ndarray:
    sheet_name_list = ["R_R", "R_G", "R_B", "G_R", "G_G", "G_B", "B_R", "B_G", "B_B"]
    sheet_matrixs = [
        pd.read_excel(
            file_path, sheet_name, header=None, usecols=range(64), nrows=64
        ).values
        for sheet_name in sheet_name_list
    ]
    return np.stack(sheet_matrixs, axis=-1)


# 计算校正前和校正后的颜色偏差的标准差、平均值和最大值
def calculate_uniformity(
    original_data: np.ndarray, corrected_data: np.ndarray, target_color: np.ndarray
) -> tuple:
    original_color = original_data.reshape(-1, 3)
    corrected_color = corrected_data.reshape(-1, 3)

    # 计算原始数据和校正数据的颜色差异
    # 欧氏距离：np.linalg.norm(original_color - target_color, axis=1)
    # CIEDE2000：np.array([calculate_delta_e(orig, target_color) for orig in original_color])
    original_differences = np.array(
        [calculate_delta_e(orig, target_color) for orig in original_color]
    )
    corrected_differences = np.array(
        [calculate_delta_e(corr, target_color) for corr in corrected_color]
    )

    # 计算标准差、平均值和最大值
    original_std = np.std(original_differences)
    corrected_std = np.std(corrected_differences)

    original_mean = np.mean(original_differences)
    corrected_mean = np.mean(corrected_differences)

    original_max = np.max(original_differences)
    corrected_max = np.max(corrected_differences)

    return (
        original_std,
        corrected_std,
        original_mean,
        corrected_mean,
        original_max,
        corrected_max,
    )


# 可视化校正前和校正后的结果
def visualize_results(
    original_red,
    original_green,
    original_blue,
    corrected_red,
    corrected_green,
    corrected_blue,
    optim_corrected_red,
    optim_corrected_green,
    optim_corrected_blue,
):
    _, axes = plt.subplots(3, 3, figsize=(15, 10))

    # 定义标题和数据
    titles = [
        "原始红色",
        "原始绿色",
        "原始蓝色",
        "矩阵校正后红色",
        "矩阵校正后绿色",
        "矩阵校正后蓝色",
        "优化校正后红色",
        "优化校正后绿色",
        "优化校正后蓝色",
    ]
    images = [
        original_red,
        original_green,
        original_blue,
        corrected_red,
        corrected_green,
        corrected_blue,
        optim_corrected_red,
        optim_corrected_green,
        optim_corrected_blue,
    ]

    # 遍历绘制图像
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.set_title(titles[i])
        ax.axis("off")  # 隐藏坐标轴

    plt.tight_layout()
    plt.savefig("led_color_correction_results.png")
    plt.show()


# 保存校正后的RGB值到Excel文件
def save_corrected_data(corrected_red, corrected_green, corrected_blue, file_path):
    with pd.ExcelWriter(file_path) as writer:
        # 定义通道名称和对应数据
        channels = {"R": corrected_red, "G": corrected_green, "B": corrected_blue}
        components = ["R", "G", "B"]

        # 遍历每个通道和颜色分量
        for color, corrected in channels.items():
            for i, component in enumerate(components):
                pd.DataFrame(corrected[:, :, i]).to_excel(
                    writer, sheet_name=f"校正后_{color}_{component}", index=False
                )

    print(f"校正后的RGB值已保存到 {file_path}")


# 保存颜色校正的详细数值结果到文本文件
def save_numerical_results(
    original_red,
    original_green,
    original_blue,
    corrected_red,
    corrected_green,
    corrected_blue,
    target_red,
    target_green,
    target_blue,
    filename,
):
    # 定义通道名称和对应数据
    channels = {
        "红色": (original_red, corrected_red, target_red),
        "绿色": (original_green, corrected_green, target_green),
        "蓝色": (original_blue, corrected_blue, target_blue),
    }

    # 计算每个通道的均匀性指标并存储结果
    channel_results = {}
    for color_name, (original, corrected, target) in channels.items():
        channel_results[color_name] = calculate_uniformity(original, corrected, target)

    # 计算总体均匀性指标
    total_orig_std = np.mean([result[0] for result in channel_results.values()])
    total_corr_std = np.mean([result[1] for result in channel_results.values()])
    total_improvement = (total_orig_std - total_corr_std) / total_orig_std * 100

    # 写入结果到文件
    with open(filename, "w", encoding="utf-8") as f:
        f.write("颜色校正结果分析报告\n")
        f.write("=" * 50 + "\n\n")

        # 写入每个通道的分析结果
        for color_name, (
            orig_std,
            corr_std,
            orig_mean,
            corr_mean,
            orig_max,
            corr_max,
        ) in channel_results.items():
            f.write(f"{color_name}通道分析:\n")
            f.write("-" * 30 + "\n")
            f.write(f"原始标准差: {orig_std:.2f}\n")
            f.write(f"校正后标准差: {corr_std:.2f}\n")
            f.write(f"改善百分比: {((orig_std - corr_std) / orig_std * 100):.2f}%\n")
            f.write(f"原始平均偏差: {orig_mean:.2f}\n")
            f.write(f"校正后平均偏差: {corr_mean:.2f}\n")
            f.write(f"原始最大偏差: {orig_max:.2f}\n")
            f.write(f"校正后最大偏差: {corr_max:.2f}\n\n")

        # 写入总体分析结果
        f.write("总体分析:\n")
        f.write("-" * 30 + "\n")
        f.write(f"总体原始标准差: {total_orig_std:.2f}\n")
        f.write(f"总体校正后标准差: {total_corr_std:.2f}\n")
        f.write(f"总体改善百分比: {total_improvement:.2f}%\n")
