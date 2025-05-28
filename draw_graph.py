import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns # 用于更好看的绘图样式
import os
import glob
from datetime import datetime # 用于文件名中的时间戳转换
from collections import Counter
from scipy import stats as scipy_stats # 用于计算偏度和峰度

# --- 配置 ---
DATA_DIR = "video_data"  # 存放TRNG数据文件的文件夹名
ANALYSIS_OUTPUT_DIR = "graphs"  # 存放分析结果图表的文件夹名
DEFAULT_HIST_BINS = 50  # 直方图默认的柱子(bins)数量
BITMAP_IMAGE_SIZE = 128 # 生成位图图像的边长 (例如 128x128 像素)

# --- 辅助函数 ---
def hex_string_to_bit_list(hex_s):
    """将十六进制字符串转换为0和1的整数列表 (比特列表)"""
    if not isinstance(hex_s, str) or not all(c in '0123456789abcdefABCDEF' for c in hex_s):
        # print(f"警告: 无效的十六进制字符串 '{hex_s}'")
        return []
    
    num_of_bits = len(hex_s) * 4
    try:
        int_val = int(hex_s, 16)
    except ValueError:
        # print(f"警告: 无法将 '{hex_s}' 转换为整数。")
        return []
    
    # 将整数转换为特定长度的二进制字符串，然后转为比特列表
    # bin(int_val)[2:] 去掉 "0b" 前缀
    # .zfill(num_of_bits) 左边补0直到达到总比特数
    return [int(bit) for bit in bin(int_val)[2:].zfill(num_of_bits)]

def select_data_file(data_dir):
    """列出数据文件并让用户选择，或自动选择最新的一个。"""
    search_path = os.path.join(data_dir, "*_video_rng.txt")
    files = glob.glob(search_path) # 使用glob查找匹配的文件
    
    if not files:
        print(f"在 '{data_dir}' 文件夹中没有找到匹配 '*_video_rng.txt' 的数据文件。")
        return None

    # 按文件修改时间降序排序 (最新的在前)
    files.sort(key=os.path.getmtime, reverse=True)

    print("\n找到以下TRNG数据文件:")
    for i, f_path in enumerate(files):
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(f_path)).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  {i+1}. {os.path.basename(f_path)} (修改于: {file_mod_time})")

    if len(files) == 1:
        print(f"自动选择最新的文件: {os.path.basename(files[0])}")
        return files[0]
    else:
        while True:
            try:
                choice_str = input(f"请选择要分析的文件编号 (1-{len(files)}), 或直接按 Enter 选择最新的 ({os.path.basename(files[0])}): ")
                if not choice_str:  # 用户直接按回车
                    return files[0]
                choice_idx = int(choice_str) - 1
                if 0 <= choice_idx < len(files):
                    return files[choice_idx]
                else:
                    print("无效的选择，请输入列表中的编号。")
            except ValueError:
                print("无效输入，请输入数字。")

# --- 主要分析逻辑 ---
def analyze_trng_data(file_path):
    """加载、分析TRNG数据并生成图表"""
    print(f"\n--- 正在分析文件: {os.path.basename(file_path)} ---")

    try:
        # 使用 Pandas 加载数据, 列名: GenerationTimestamp,HexValue,RealValue0To1
        df = pd.read_csv(file_path, comment='#') # comment='#' 忽略可能存在的注释行
        if df.empty:
            print("数据文件为空。")
            return
        
        # 数据清洗和转换
        if 'RealValue0To1' not in df.columns:
            print("错误: 数据文件中缺少 'RealValue0To1' 列。")
            return
        if 'HexValue' not in df.columns:
            print("警告: 数据文件中缺少 'HexValue' 列。比特流相关分析将跳过。")
            # 即使缺少HexValue，如果RealValue0To1存在，仍可进行部分分析
        
        # 将RealValue0To1转换为数值型，无法转换的值(如"N/A")会变成NaN
        df['RealValue0To1'] = pd.to_numeric(df['RealValue0To1'], errors='coerce')
        df.dropna(subset=['RealValue0To1'], inplace=True) # 移除包含NaN的行 (针对RealValue0To1列)

        if df.empty:
            print("在 'RealValue0To1' 列中没有有效的数值数据。")
            return
            
        real_values = df['RealValue0To1'].values
        # HexValue列可能包含非字符串或NaN，先确保是字符串列表
        if 'HexValue' in df.columns:
            hex_values = df['HexValue'].astype(str).tolist() 
        else:
            hex_values = []


    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。")
        return
    except pd.errors.EmptyDataError:
        print(f"错误: 文件 '{file_path}' 为空或格式不正确。")
        return
    except Exception as e:
        print(f"读取或解析文件时发生错误: {e}")
        return

    num_samples = len(real_values)
    if num_samples == 0: # 再次检查，因为dropna后可能为空
        print("在 'RealValue0To1' 列中没有有效的数值数据可供分析。")
        return
        
    print(f"\n共加载 {num_samples} 个有效的 RealValue0To1 样本进行分析。")

    # 为图表创建输出目录 (基于输入文件名)
    base_filename_no_ext = os.path.splitext(os.path.basename(file_path))[0]
    output_plot_dir_for_file = os.path.join(ANALYSIS_OUTPUT_DIR, base_filename_no_ext)
    os.makedirs(output_plot_dir_for_file, exist_ok=True)
    print(f"分析图表将保存到: {output_plot_dir_for_file}")

    # 1. 分析 RealValue0To1 (实数 [0,1) )
    print("\n--- 实数 RealValue0To1 分析 ---")
    mean_val = np.mean(real_values)
    median_val = np.median(real_values)
    std_dev = np.std(real_values) # ddof=0 by default for np.std
    min_val = np.min(real_values)
    max_val = np.max(real_values)
    # scipy.stats.skew 和 .kurtosis 默认使用有偏估计 (bias=True)
    skew_val = scipy_stats.skew(real_values) 
    kurt_val = scipy_stats.kurtosis(real_values, fisher=True) # Fisher=True: 正态分布峰度为0

    print(f"  均值 (Mean):         {mean_val:.6f} (理想 U[0,1) ≈ 0.5)")
    print(f"  中位数 (Median):       {median_val:.6f} (理想 U[0,1) ≈ 0.5)")
    print(f"  标准差 (Std Dev):    {std_dev:.6f} (理想 U[0,1) ≈ {1/np.sqrt(12):.4f})")
    print(f"  最小值 (Min):        {min_val:.6f}")
    print(f"  最大值 (Max):        {max_val:.6f}")
    print(f"  偏度 (Skewness):     {skew_val:.6f} (理想 U[0,1) = 0)")
    print(f"  峰度 (Kurtosis):     {kurt_val:.6f} (理想 U[0,1) = -1.2, Fisher定义下)")


    # 1a. RealValue0To1 的直方图
    plt.figure(figsize=(12, 7))
    # 使用密度而不是频数，便于与理想均匀分布的密度1.0比较
    sns.histplot(real_values, bins=DEFAULT_HIST_BINS, kde=False, stat="density", color='skyblue', edgecolor='black', label='实际数据密度')
    # 理想均匀分布 U[0,1) 的概率密度函数值为1
    plt.axhline(1.0, color='r', linestyle='dashed', linewidth=1.5, label='理想均匀分布密度 (1.0)')
    plt.title(f'实数值 RealValue0To1 分布直方图 (共 {num_samples} 个样本)', fontsize=16)
    plt.xlabel('数值', fontsize=14)
    plt.ylabel('密度', fontsize=14)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    hist_path = os.path.join(output_plot_dir_for_file, f"{base_filename_no_ext}_real_value_histogram.png")
    plt.savefig(hist_path)
    plt.close()
    print(f"  直方图已保存到: {os.path.basename(hist_path)}")

    # 1b. RealValue0To1 的连续值散点图 (R_i vs R_{i+1})
    if num_samples > 1:
        plt.figure(figsize=(8, 8))
        # 使用较小的点和一定的透明度来处理大量数据点
        plt.scatter(real_values[:-1], real_values[1:], alpha=0.3, s=3, edgecolor='none', c='blue')
        plt.title(f'连续随机数散点图 ($R_i$ vs $R_{{i+1}}$) (共 {num_samples-1} 对)', fontsize=16)
        plt.xlabel('$R_i$', fontsize=14)
        plt.ylabel('$R_{i+1}$', fontsize=14)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.gca().set_aspect('equal', adjustable='box') # 使X轴和Y轴等比例，图像为正方形
        plt.grid(True, linestyle='--', alpha=0.5)
        scatter_path = os.path.join(output_plot_dir_for_file, f"{base_filename_no_ext}_real_value_scatter.png")
        plt.savefig(scatter_path)
        plt.close()
        print(f"  连续值散点图已保存到: {os.path.basename(scatter_path)}")

    # 2. 分析 HexValue (十六进制哈希值) -> 转换为比特流
    print("\n--- 十六进制值 HexValue (比特流) 分析 ---")
    all_bits = []
    if hex_values: # 检查列表是否为空
        for hex_s in hex_values:
            all_bits.extend(hex_string_to_bit_list(str(hex_s))) # 确保传入的是字符串

        if not all_bits:
            print("  未能从HexValue中提取任何比特进行分析。")
        else:
            total_bits = len(all_bits)
            counts = Counter(all_bits) # 统计0和1的个数
            zeros = counts.get(0, 0)
            ones = counts.get(1, 0)
            print(f"  从HexValue中提取的总比特数: {total_bits}")
            if total_bits > 0: # 避免除以零
                print(f"  0 的数量: {zeros} (占比: {zeros/total_bits:.4f})")
                print(f"  1 的数量: {ones} (占比: {ones/total_bits:.4f}) (理想占比均为0.5)")

                # 2a. 比特平衡条形图
                plt.figure(figsize=(7, 5))
                sns.barplot(x=['0 比特', '1 比特'], y=[zeros, ones], palette=['lightcoral', 'lightgreen'])
                plt.title(f'比特流中0和1的分布 (总比特数: {total_bits})', fontsize=16)
                plt.ylabel('数量', fontsize=14)
                for index, value in enumerate([zeros, ones]): # 在柱子上显示具体数值
                    plt.text(index, value + (0.01 * total_bits), str(value), ha='center', va='bottom')
                bit_balance_path = os.path.join(output_plot_dir_for_file, f"{base_filename_no_ext}_bit_balance.png")
                plt.savefig(bit_balance_path)
                plt.close()
                print(f"  比特平衡图已保存到: {os.path.basename(bit_balance_path)}")

                # 2b. 位图图像 (Bitmap Image)
                if total_bits >= BITMAP_IMAGE_SIZE * BITMAP_IMAGE_SIZE:
                    bitmap_bits_to_plot = all_bits[:BITMAP_IMAGE_SIZE * BITMAP_IMAGE_SIZE]
                    # 将比特列表转换为Numpy数组并重塑为二维
                    bitmap_array = np.array(bitmap_bits_to_plot).reshape((BITMAP_IMAGE_SIZE, BITMAP_IMAGE_SIZE))
                    
                    plt.figure(figsize=(8, 8))
                    plt.imshow(bitmap_array, cmap='gray', interpolation='nearest') # 'gray'意味着0是黑，1是白
                    plt.title(f'随机比特位图 ({BITMAP_IMAGE_SIZE}x{BITMAP_IMAGE_SIZE})', fontsize=16)
                    plt.xticks([]) # 关闭x轴刻度
                    plt.yticks([]) # 关闭y轴刻度
                    bitmap_path = os.path.join(output_plot_dir_for_file, f"{base_filename_no_ext}_bitmap_image.png")
                    plt.savefig(bitmap_path)
                    plt.close()
                    print(f"  位图图像已保存到: {os.path.basename(bitmap_path)}")
                else:
                    print(f"  比特数不足以生成 {BITMAP_IMAGE_SIZE}x{BITMAP_IMAGE_SIZE} 的位图 (需要 {BITMAP_IMAGE_SIZE*BITMAP_IMAGE_SIZE}，现有 {total_bits})。")
    else:
        print("  数据文件中没有 HexValue 数据，或数据为空，跳过比特流分析。")

    print("\n--- 分析完成 ---")

if __name__ == "__main__":
    # --- 字体设置 ---
    user_font_path = '/System/Library/Fonts/STHeiti Light.ttc'  # 您指定的字体路径
    font_successfully_set = False

    print("--- 正在尝试配置绘图字体 ---")
    try:
        if os.path.exists(user_font_path):
            # 如果用户指定的字体文件存在，则尝试使用它
            font_prop = fm.FontProperties(fname=user_font_path)
            font_name = font_prop.get_name() # 获取Matplotlib识别的字体名
            
            # 将此字体设置为默认字体系列
            # 有几种方式可以设置，更推荐的方式是直接设置 'font.family'
            # 或者将其添加到特定字体类型（如sans-serif）的优先级列表
            plt.rcParams['font.family'] = font_name 
            # 或者, 如果你知道这个字体是无衬线的:
            # plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
            
            plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
            print(f"信息: Matplotlib 字体尝试设置为 '{font_name}' (来自路径: {user_font_path})。")
            font_successfully_set = True
        else:
            print(f"警告: 指定的字体文件未找到: {user_font_path}。")

        if not font_successfully_set:
            # 如果用户指定字体失败，尝试之前的通用回退列表
            print("信息: 尝试通用回退字体列表...")
            potential_fonts = ['PingFang SC', 'STHeiti', 'Heiti SC', 'SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei', 'sans-serif']
            for font_name_fallback in potential_fonts:
                try:
                    plt.rcParams['font.sans-serif'] = [font_name_fallback] # 将其设为sans-serif的首选
                    plt.rcParams['axes.unicode_minus'] = False
                    # 可以加一个简单的测试来确认字体是否真的能用（可选）
                    # fig_test, ax_test = plt.subplots(figsize=(0.1,0.1)); ax_test.set_title("测"); plt.close(fig_test)
                    print(f"信息: Matplotlib 中文字体尝试设置为 '{font_name_fallback}'。")
                    font_successfully_set = True
                    break # 找到一个可用字体就停止
                except Exception:
                    continue # 尝试下一个字体
            
            if not font_successfully_set:
                print("警告: 未能成功设置任何已知的中文字体。图表中的中文可能无法正常显示。")
        
        # 无论如何，确保这个参数被设置
        if not 'axes.unicode_minus' in plt.rcParams or not plt.rcParams['axes.unicode_minus']:
             plt.rcParams['axes.unicode_minus'] = False


    except Exception as e:
        print(f"设置字体过程中发生错误: {e}。图表中的中文可能无法正常显示。")
        if not 'axes.unicode_minus' in plt.rcParams or not plt.rcParams['axes.unicode_minus']:
            plt.rcParams['axes.unicode_minus'] = False # 确保至少这个被设置
    
    print("--- 字体配置尝试结束 ---")
    # --- 字体设置结束 ---

    # 确保主分析输出目录存在
    os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)

    selected_file_path = select_data_file(DATA_DIR)
    if selected_file_path:
        analyze_trng_data(selected_file_path)
    else:
        print("没有选择数据文件，程序退出。")