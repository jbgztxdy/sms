import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from utils import (
    load_led_data,
    save_corrected_data,
    save_numerical_results,
    visualize_results,
)


# 函数：将RGB值转换为CIE XYZ颜色空间
def rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb / 255.0  # 归一化到0-1范围

    rgb = np.where(rgb > 0.04045, np.power(rgb, 2.2), rgb / 12.92)  # 应用伽马校正

    # sRGB到XYZ的转换矩阵
    matrix = np.array(
        [
            [0.4124, 0.3576, 0.1805],
            [0.2126, 0.7152, 0.0722],
            [0.0193, 0.1192, 0.9505],
        ]
    )

    xyz = np.dot(matrix, rgb)

    return xyz


# 函数：将XYZ值转换回RGB颜色空间
def xyz_to_rgb(xyz: np.ndarray) -> np.ndarray:
    # XYZ到sRGB的转换矩阵
    matrix = np.array(
        [
            [3.2406, -1.5372, -0.4986],
            [-0.9689, 1.8758, 0.0415],
            [0.0557, -0.2040, 1.0570],
        ]
    )

    # 矩阵乘法计算RGB值
    rgb = np.dot(matrix, xyz)

    # 限制RGB值在合理范围内，避免负值或NaN
    rgb = np.clip(rgb, 0, None)

    # 应用逆伽马校正
    rgb = np.where(rgb > 0.0031308, 1.055 * np.power(rgb, 1 / 2.4) - 0.055, 12.92 * rgb)

    # 限制在0-1范围并缩放到0-255
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


# 拟合响应矩阵 A 和偏置向量 b
def fit_response_matrix(C: np.ndarray, F) -> tuple[np.ndarray, np.ndarray]:
    """
    拟合响应矩阵 A 和偏置向量 b
    C: 输入颜色矩阵 (n, 3)
    F: 实际输出颜色矩阵 (n, 3)
    返回: A (3, 3), b (3,)
    """
    # 添加偏置列
    C_augmented = np.hstack([C, np.ones((C.shape[0], 1))])  # (n, 4)

    # 最小二乘法拟合
    params, _, _, _ = np.linalg.lstsq(C_augmented, F, rcond=None)  # (4, 3)
    A = params[:3, :]  # (3, 3)
    b = params[3, :]  # (3,)
    return A.T, b


# 优化目标函数
def optimization_target(C_corr, A, b, T, C_0, lambda_):
    F_corr = np.dot(A, C_corr) + b
    term1 = np.linalg.norm(F_corr - T) ** 2  # 输出误差
    term2 = lambda_ * np.linalg.norm(C_corr - C_0) ** 2  # 偏离原始输入的惩罚
    return term1 + term2


# 优化校正值
def optimize_color(A, b, T, C_0, lambda_):
    result = minimize(
        fun=lambda C_corr: optimization_target(C_corr, A, b, T, C_0, lambda_),
        x0=C_0,  # 初始值为原始输入值
        method="L-BFGS-B",  # 优化方法
        bounds=[(0, 255)] * 3,  # 限制校正值在 [0, 255] 范围内
    )
    return result.x


# 校正函数
def color_correction(data: np.ndarray, use_lstsq=False, lambda_=0.01, xyz=False):
    """
    使用伪逆或最小二乘优化方法进行颜色校正
    :param data: 输入数据，形状为 (rows, cols, 9)
    :param method: 校正方法，"lstsq" 表示伪逆，"optimize" 表示最小二乘优化
    :param lambda_: 平衡系数，仅在最小二乘优化时使用
    :return: 校正后的红、绿、蓝通道数据
    """
    # 获取数据维度
    rows, cols, _ = data.shape

    # 初始化校正后的 RGB 通道
    corrected_red = np.zeros((rows, cols, 3), dtype=np.uint8)
    corrected_green = np.zeros((rows, cols, 3), dtype=np.uint8)
    corrected_blue = np.zeros((rows, cols, 3), dtype=np.uint8)

    # 目标 RGB 值
    target_output = np.array([[220, 0, 0], [0, 220, 0], [0, 0, 220]])

    # 遍历每个像素
    for i in range(rows):
        for j in range(cols):
            # 当前像素的输入颜色矩阵 (3x3)
            current_output = data[i, j].reshape(3, 3)

            if not use_lstsq:
                # 使用伪逆方法校正
                corrected_pixel = pinv_correction(current_output, target_output, xyz)
            else:
                # 使用最小二乘优化方法校正
                corrected_pixel = optim_correction(
                    current_output, target_output, lambda_
                )

            # 保存校正后的结果
            corrected_pixel = np.clip(corrected_pixel, 0, 255).astype(np.uint8)
            corrected_red[i, j] = corrected_pixel[0]
            corrected_green[i, j] = corrected_pixel[1]
            corrected_blue[i, j] = corrected_pixel[2]

    return corrected_red, corrected_green, corrected_blue


def pinv_correction(current_output: np.ndarray, target_output: np.ndarray, xyz=False):
    if xyz:
        current_output = rgb_to_xyz(current_output)
        target_output = rgb_to_xyz(target_output)

    A, b = fit_response_matrix(current_output, target_output)

    target_input = np.linalg.pinv(A) @ (target_output - b)

    corrected_output = np.dot(A, target_input) + b

    if xyz:
        corrected_output = xyz_to_rgb(corrected_output)

    return corrected_output


def optim_correction(
    current_output: np.ndarray, target_output: np.ndarray, lambda_, xyz=False
):
    """
    使用最小二乘优化方法进行颜色校正
    :param current_rgb: 当前像素的输入颜色矩阵 (3x3)
    :param target_rgb: 目标 RGB 值 (3x3)
    :param lambda_: 平衡系数
    :return: 校正后的 RGB 值
    """
    if xyz:
        current_output = rgb_to_xyz(current_output)
        target_output = rgb_to_xyz(target_output)

    # 拟合响应矩阵 A 和偏置向量 b
    A, b = fit_response_matrix(current_output, target_output)  # 假设当前输出等于输入

    # 对每个目标颜色进行优化
    corrected_pixel = []
    for k in range(3):
        # 初始输入颜色
        target_input = np.linalg.pinv(A) @ (target_output - b)
        C_0 = target_input[k]
        # C_0 = target_output[k]

        # 目标颜色
        T = target_output[k]

        # 优化校正值
        C_corr = optimize_color(A, b, T, C_0, lambda_)
        corrected_pixel.append(C_corr)

    corrected_pixel = np.array(corrected_pixel)

    corrected_output = np.dot(A, corrected_pixel.T).T + b.T

    if xyz:
        corrected_output = xyz_to_rgb(corrected_output)

    return corrected_output


# 主程序
def main():
    # 数据文件路径
    file_path = Path("RGB数值.xlsx")
    print(f"数据文件路径: {file_path}")

    # 加载数据
    data = load_led_data(file_path)

    # 获取数据维度
    rows, cols, _ = data.shape

    # 构建原始颜色数据
    original_red = np.zeros((rows, cols, 3), dtype=np.uint8)
    original_green = np.zeros((rows, cols, 3), dtype=np.uint8)
    original_blue = np.zeros((rows, cols, 3), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            original_red[i, j] = data[i, j, :3]
            original_green[i, j] = data[i, j, 3:6]
            original_blue[i, j] = data[i, j, 6:9]

    print("开始执行高级颜色校正...")

    # 目标RGB值
    target_red = np.array([220, 0, 0])
    target_green = np.array([0, 220, 0])
    target_blue = np.array([0, 0, 220])

    pinv_corrected_red, pinv_corrected_green, pinv_corrected_blue = color_correction(
        data, False, 0.01
    )

    # 保存数值结果到文本文件
    filename = "pinv_correction_results.txt"
    save_numerical_results(
        original_red,
        original_green,
        original_blue,
        pinv_corrected_red,
        pinv_corrected_green,
        pinv_corrected_blue,
        target_red,
        target_green,
        target_blue,
        filename,
    )

    # 保存校正后的数据
    file_path = Path("RGB伪逆校正后数值.xlsx")
    save_corrected_data(
        pinv_corrected_red, pinv_corrected_green, pinv_corrected_blue, file_path
    )

    optim_corrected_red, optim_corrected_green, optim_corrected_blue = color_correction(
        data, True, 0.01
    )

    filename = "optim_correction_results.txt"
    save_numerical_results(
        original_red,
        original_green,
        original_blue,
        optim_corrected_red,
        optim_corrected_green,
        optim_corrected_blue,
        target_red,
        target_green,
        target_blue,
        filename,
    )

    # 保存校正后的数据
    file_path = Path("RGB优化校正后数值.xlsx")
    save_corrected_data(
        optim_corrected_red, optim_corrected_green, optim_corrected_blue, file_path
    )

    # 可视化结果
    visualize_results(
        original_red,
        original_green,
        original_blue,
        pinv_corrected_red,
        pinv_corrected_green,
        pinv_corrected_blue,
        optim_corrected_red,
        optim_corrected_green,
        optim_corrected_blue,
    )


if __name__ == "__main__":
    main()
