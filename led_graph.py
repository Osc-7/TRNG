import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # 用于更好看的绘图样式
import os
import glob
from datetime import datetime # 用于文件名中的时间戳转换
from collections import Counter
import matplotlib.font_manager as fm # 用于字体管理

# --- 配置 ---
# DATA_DIR 是您的 LED TRNG 数据文件所在的文件夹名
# 在之前的 led_detector.py 脚本中, 我们将其设置为 "data"
DATA_DIR = "data"
# ANALYSIS_OUTPUT_DIR 是存放分析结果图表的文件夹名
ANALYSIS_OUTPUT_DIR = "led_data_analysis"
DEFAULT_BAR_COLOR = 'cornflowerblue'

# --- 辅助函数 ---
def select_data_file(data_dir):
    """列出数据文件并让用户选择，或自动选择最新的一个。"""
    # LED TRNG脚本生成的文件名格式为 YYYY-MM-DD_HH-MM-SS.txt
    # 或者最初可能是 random_bits_log.txt
    # 我们将匹配 data_dir 下的所有 .txt 文件
    search_path = os.path.join(data_dir, "*.txt")
    files = glob.glob(search_path)
    
    if not files:
        print(f"在 '{data_dir}' 文件夹中没有找到 .txt 数据文件。")
        return None

    # 按文件修改时间降序排序 (最新的在前)
    files.sort(key=os.path.getmtime, reverse=True)

    print("\n找到以下LED TRNG数据文件:")
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

# --- 绘图与分析函数 ---
def plot_binary_string_frequencies(df_led, output_dir, base_fn):
    """分析并绘制不同二进制字符串出现的频率"""
    if 'BinaryString' not in df_led.columns or df_led['BinaryString'].empty:
        print("  数据中无 'BinaryString' 列或该列为空，跳过二进制字符串频率分析。")
        return

    binary_counts = df_led['BinaryString'].astype(str).value_counts().sort_index()
    num_unique_strings = len(binary_counts)
    total_strings = df_led['BinaryString'].count()

    if num_unique_strings == 0:
        print("  没有有效的二进制字符串可供分析频率。")
        return

    print(f"\n--- 二进制字符串频率分析 (共 {total_strings} 个样本) ---")
    # 假设是4位二进制数，有16种可能
    num_bits = len(binary_counts.index[0]) if num_unique_strings > 0 else 0
    expected_count = total_strings / (2**num_bits) if num_bits > 0 else total_strings / num_unique_strings

    print(f"  检测到 {num_unique_strings} 种不同的二进制字符串 (基于 {num_bits} 位)。")
    print(f"  理想情况下，每种字符串期望出现次数约: {expected_count:.2f}")
    # print("  实际出现次数:")
    # for string, count in binary_counts.items():
    #     print(f"    '{string}': {count}")

    plt.figure(figsize=(12, 7))
    bars = sns.barplot(x=binary_counts.index, y=binary_counts.values, color=DEFAULT_BAR_COLOR, edgecolor='black')
    plt.axhline(expected_count, color='r', linestyle='dashed', linewidth=1.5, label=f'理想平均频数 ({expected_count:.1f})')
    plt.title(f'{num_bits}位 二进制字符串出现频率 (总样本数: {total_strings})', fontsize=16)
    plt.xlabel('二进制字符串', fontsize=14)
    plt.ylabel('出现次数', fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout() # 调整布局以防止标签重叠
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 在柱状图上显示数值
    for bar in bars.patches:
        bars.annotate(format(bar.get_height(), '.0f'), # .0f 表示整数
                       (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                       ha='center', va='bottom',
                       size=9, xytext=(0, 5),
                       textcoords='offset points')

    fig_path = os.path.join(output_dir, f"{base_fn}_binary_string_frequency.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"  二进制字符串频率图已保存到: {os.path.basename(fig_path)}")


def plot_overall_bit_balance(df_led, output_dir, base_fn):
    """分析并绘制整体比特流中0和1的平衡性"""
    if 'BinaryString' not in df_led.columns or df_led['BinaryString'].empty:
        print("  数据中无 'BinaryString' 列或该列为空，跳过比特平衡分析。")
        return
        
    all_bits_str = "".join(df_led['BinaryString'].astype(str))
    if not all_bits_str:
        print("  未能从 'BinaryString' 列提取任何比特。")
        return

    all_bits_list = [int(bit) for bit in all_bits_str if bit in '01'] # 转换为整数列表
    
    if not all_bits_list:
        print("  提取的比特流为空或无效。")
        return

    total_bits = len(all_bits_list)
    counts = Counter(all_bits_list)
    zeros = counts.get(0, 0)
    ones = counts.get(1, 0)

    print(f"\n--- 整体比特流平衡性分析 (总比特数: {total_bits}) ---")
    if total_bits > 0:
        print(f"  0 的数量: {zeros} (占比: {zeros/total_bits:.4f})")
        print(f"  1 的数量: {ones} (占比: {ones/total_bits:.4f}) (理想占比均为0.5)")

        plt.figure(figsize=(7, 5))
        bars = sns.barplot(x=['0 比特', '1 比特'], y=[zeros, ones], palette=['lightcoral', 'lightgreen'])
        plt.title(f'整体比特流中0和1的分布 (总比特数: {total_bits})', fontsize=16)
        plt.ylabel('数量', fontsize=14)
        for index, value in enumerate([zeros, ones]):
            plt.text(index, value + (0.01 * total_bits), str(value), ha='center', va='bottom')
        
        fig_path = os.path.join(output_dir, f"{base_fn}_overall_bit_balance.png")
        plt.savefig(fig_path)
        plt.close()
        print(f"  整体比特平衡图已保存到: {os.path.basename(fig_path)}")
    else:
        print("  没有可供分析的比特。")

def plot_individual_led_bias(df_led, output_dir, base_fn):
    """分析并绘制每个LED位置上0和1的偏置情况"""
    if 'BinaryString' not in df_led.columns or df_led['BinaryString'].empty:
        print("  数据中无 'BinaryString' 列或该列为空，跳过独立LED偏置分析。")
        return

    # 确保所有二进制字符串长度一致，取第一个作为参考长度
    try:
        # 过滤掉非字符串或空字符串的数据
        valid_binary_strings = df_led['BinaryString'][df_led['BinaryString'].apply(lambda x: isinstance(x, str) and len(x) > 0)]
        if valid_binary_strings.empty:
            print("  没有有效的二进制字符串用于独立LED偏置分析。")
            return
        num_leds = len(valid_binary_strings.iloc[0])
        if not all(len(s) == num_leds for s in valid_binary_strings):
            print("  错误: 二进制字符串长度不一致，无法进行独立LED偏置分析。")
            return
    except IndexError: # 如果 valid_binary_strings 为空
        print("  没有有效的二进制字符串用于确定LED数量。")
        return


    print(f"\n--- 独立LED偏置分析 (共 {num_leds} 个LED) ---")
    
    led_biases_data = {'LED_Position': [], 'Bit_Value': [], 'Count': [], 'Proportion': []}

    for i in range(num_leds):
        bits_at_position_i = [s[i] for s in valid_binary_strings if s[i] in '01']
        if not bits_at_position_i:
            print(f"  LED {i+1} (位置 {i}) 没有有效比特数据。")
            continue

        total_at_pos = len(bits_at_position_i)
        counts_at_pos = Counter(bits_at_position_i)
        zeros_at_pos = counts_at_pos.get('0', 0)
        ones_at_pos = counts_at_pos.get('1', 0)

        print(f"  LED {i+1}: 0s={zeros_at_pos} ({zeros_at_pos/total_at_pos:.4f}), 1s={ones_at_pos} ({ones_at_pos/total_at_pos:.4f})")
        
        led_biases_data['LED_Position'].extend([f'LED {i+1}'] * 2)
        led_biases_data['Bit_Value'].extend(['0', '1'])
        led_biases_data['Count'].extend([zeros_at_pos, ones_at_pos])
        led_biases_data['Proportion'].extend([zeros_at_pos/total_at_pos if total_at_pos else 0, 
                                             ones_at_pos/total_at_pos if total_at_pos else 0])

    if not led_biases_data['LED_Position']: # 如果没有任何LED数据被处理
        print("  未能收集到任何独立LED的偏置数据。")
        return

    bias_df = pd.DataFrame(led_biases_data)

    plt.figure(figsize=(max(8, num_leds * 2), 6)) # 动态调整宽度
    sns.barplot(x='LED_Position', y='Proportion', hue='Bit_Value', data=bias_df, palette={'0': 'lightcoral', '1': 'lightgreen'})
    plt.title(f'各LED位置上0/1比特占比 (共 {valid_binary_strings.count()} 个有效样本)', fontsize=16)
    plt.xlabel('LED 位置', fontsize=14)
    plt.ylabel('占比', fontsize=14)
    plt.ylim(0, 1.05) # Y轴从0到1，留一点上面空间给标签
    plt.axhline(0.5, color='grey', linestyle='dashed', linewidth=1, label='理想占比 (0.5)')
    plt.legend(title='比特值')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    fig_path = os.path.join(output_dir, f"{base_fn}_individual_led_bias.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"  独立LED偏置图已保存到: {os.path.basename(fig_path)}")


# --- 主分析函数 ---
def analyze_led_detector_data(file_path):
    """加载、分析LED TRNG数据并生成图表"""
    print(f"\n--- 正在分析LED TRNG数据文件: {os.path.basename(file_path)} ---")

    try:
        # 列名: Timestamp,BinaryString
        df = pd.read_csv(file_path, comment='#', dtype={'BinaryString': str}) # 确保BinaryString作为字符串读取
        if df.empty:
            print("数据文件为空。")
            return
            
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。")
        return
    except pd.errors.EmptyDataError:
        print(f"错误: 文件 '{file_path}' 为空或格式不正确。")
        return
    except Exception as e:
        print(f"读取或解析文件时发生错误: {e}")
        return

    # 为图表创建输出目录 (基于输入文件名)
    base_filename_no_ext = os.path.splitext(os.path.basename(file_path))[0]
    output_plot_dir_for_file = os.path.join(ANALYSIS_OUTPUT_DIR, base_filename_no_ext)
    os.makedirs(output_plot_dir_for_file, exist_ok=True)
    print(f"分析图表将保存到: {output_plot_dir_for_file}")

    # 调用各个分析和绘图函数
    plot_binary_string_frequencies(df, output_plot_dir_for_file, base_filename_no_ext)
    plot_overall_bit_balance(df, output_plot_dir_for_file, base_filename_no_ext)
    plot_individual_led_bias(df, output_plot_dir_for_file, base_filename_no_ext)
    
    print("\n--- LED TRNG 数据分析完成 ---")


# --- 主执行逻辑 ---
if __name__ == "__main__":
    # 尝试设置Matplotlib的字体以支持中文显示
    print("--- 正在尝试配置绘图字体 ---")
    try:
        font_path_stheitilight = '/System/Library/Fonts/STHeiti Light.ttc'
        font_successfully_set = False
        if os.path.exists(font_path_stheitilight):
            font_prop = fm.FontProperties(fname=font_path_stheitilight)
            font_name = font_prop.get_name()
            plt.rcParams['font.family'] = font_name
            plt.rcParams['axes.unicode_minus'] = False
            print(f"信息: Matplotlib 字体已设置为 '{font_name}' (来自: {font_path_stheitilight})。")
            font_successfully_set = True
        else:
            print(f"警告: 指定的字体路径 '{font_path_stheitilight}' 未找到。尝试通用回退字体。")

        if not font_successfully_set:
            potential_fonts = ['PingFang SC', 'SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei', 'sans-serif']
            for font_name_fallback in potential_fonts:
                try:
                    plt.rcParams['font.sans-serif'] = [font_name_fallback]
                    plt.rcParams['axes.unicode_minus'] = False
                    print(f"信息: Matplotlib 中文字体尝试设置为 '{font_name_fallback}'。")
                    font_successfully_set = True
                    break
                except Exception:
                    continue
            if not font_successfully_set:
                print("警告: 未能成功设置任何已知的中文字体。图表中的中文可能无法正常显示。")
        
        if not plt.rcParams['axes.unicode_minus']: # 确保至少这个被设置，即使字体设置失败
             plt.rcParams['axes.unicode_minus'] = False
             
    except Exception as e:
        print(f"设置字体过程中发生错误: {e}。图表中的中文可能无法正常显示。")
        plt.rcParams['axes.unicode_minus'] = False
    print("--- 字体配置尝试结束 ---")

    # 确保主分析输出目录存在
    os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)

    selected_file = select_data_file(DATA_DIR)
    if selected_file:
        analyze_led_detector_data(selected_file)
    else:
        print("没有选择数据文件，程序退出。")