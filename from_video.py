import cv2
import numpy as np
import hashlib
import time
from datetime import datetime # 用于时间戳和文件名
import os # 用于创建文件夹
from collections import Counter

# --- 配置参数 ---
CAMERA_INDEX = 0
ROI_X, ROI_Y, ROI_W, ROI_H = 100, 100, 200, 200
SAMPLE_SIZE_BITS = 256
MAX_FRAMES_TO_COLLECT_ENTROPY = 100
VON_NEUMANN_DEBIAS = True
SHA256_HASH = True

OUTPUT_SUBDIR = "video_data"
DELAY_BETWEEN_ITERATIONS_SECONDS = 0.5

# --- 辅助函数 (与之前版本相同) ---
def get_lsbs_from_frame(frame, roi_x, roi_y, roi_w, roi_h):
    """从ROI中像素的R, G, B通道提取最低有效位 (LSB)。"""
    roi = frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
    bits = []
    for row in roi:
        for pixel in row:
            bits.append(pixel[0] & 1)
            bits.append(pixel[1] & 1)
            bits.append(pixel[2] & 1)
    return bits

def von_neumann_debias(bit_stream):
    """冯·诺依曼去偏置"""
    debiased_bits = []
    i = 0
    while i < len(bit_stream) - 1:
        pair = (bit_stream[i], bit_stream[i+1])
        if pair == (0, 1):
            debiased_bits.append(0)
        elif pair == (1, 0):
            debiased_bits.append(1)
        i += 2
    return debiased_bits

def bit_list_to_byte_string(bit_list):
    """将比特列表转换为字节字符串。"""
    if not bit_list: return b''
    byte_array = bytearray()
    for i in range(0, len(bit_list), 8):
        byte = 0
        chunk = bit_list[i:i+8]
        if len(chunk) < 8 and i + len(chunk) < len(bit_list):
             continue 
        for bit_index, bit_value in enumerate(chunk):
            byte |= (bit_value << (7 - bit_index))
        byte_array.append(byte)
    return bytes(byte_array)

def map_hex_to_real01(hex_string):
    """将十六进制字符串(代表256位哈希)的前64位映射到[0,1)的实数。"""
    if not hex_string or len(hex_string) < 16 :
        return None
    hex_prefix = hex_string[:16]
    int_value = int(hex_prefix, 16)
    real_value = int_value / (2**64)
    return real_value

# --- 主要TRNG逻辑 (generate_random_bits_from_camera 与之前版本相同) ---
def generate_random_bits_from_camera(cap, target_bits_for_hash):
    """使用摄像头传感器噪声生成一个随机比特块，返回十六进制哈希字符串或特殊控制符。"""
    collected_bits = []
    frames_processed = 0
    raw_bits_to_collect = target_bits_for_hash * 4 if VON_NEUMANN_DEBIAS else target_bits_for_hash

    # 定义此函数内部使用的窗口名称，与调整阶段的窗口区分开
    generation_window_name = "摄像头TRNG - 数据采集中 (按'q'中断本轮)"

    while len(collected_bits) < raw_bits_to_collect and \
          frames_processed < MAX_FRAMES_TO_COLLECT_ENTROPY:
        ret, frame = cap.read()
        if not ret:
            print("错误: 无法读取帧 (数据采集中)。")
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)
        lsbs = get_lsbs_from_frame(frame, ROI_X, ROI_Y, ROI_W, ROI_H)
        collected_bits.extend(lsbs)
        frames_processed += 1

        display_frame = frame.copy()
        cv2.rectangle(display_frame, (ROI_X, ROI_Y), (ROI_X + ROI_W, ROI_Y + ROI_H), (0, 255, 0), 2)
        cv2.putText(display_frame, f"收集比特: {len(collected_bits)}/{raw_bits_to_collect}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(generation_window_name, display_frame) # 使用特定的窗口名
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("用户在摄像头窗口请求中断当前轮次生成。")
            return "USER_QUIT_ITERATION"

    if len(collected_bits) < raw_bits_to_collect / 2 and VON_NEUMANN_DEBIAS :
        print(f"警告: 收集到的原始比特过少 ({len(collected_bits)})。")

    processed_bits = collected_bits
    if VON_NEUMANN_DEBIAS:
        processed_bits = von_neumann_debias(collected_bits)

    if not processed_bits:
        print("警告: 去偏置后没有剩余比特。")
        return None

    if SHA256_HASH:
        byte_data = bit_list_to_byte_string(processed_bits)
        if not byte_data:
            print("警告: 没有数据进行哈希处理。")
            return None
        hashed_output = hashlib.sha256(byte_data).digest()
        return hashed_output.hex()
    else:
        return "".join(map(str, processed_bits[:target_bits_for_hash]))

# --- 主执行逻辑 ---
if __name__ == "__main__":
    try:
        duration_seconds_str = input("请输入希望脚本运行的总时间（秒）: ")
        duration_seconds = int(duration_seconds_str)
        if duration_seconds <= 0:
            print("运行时间必须为正数。")
            exit()
    except ValueError:
        print("无效的时间输入。请输入一个数字。")
        exit()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"错误: 无法打开摄像头 {CAMERA_INDEX}。")
        exit()

    # --- 文件名和路径设置 (基于脚本启动的初始时刻) ---
    script_initial_datetime_obj = datetime.now()
    formatted_script_initial_time = script_initial_datetime_obj.strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f"{formatted_script_initial_time}_video_rng.txt"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = os.path.join(script_dir, OUTPUT_SUBDIR)
    try:
        os.makedirs(data_dir_path, exist_ok=True)
    except OSError as e:
        print(f"创建目录 {data_dir_path} 失败: {e}。将尝试在当前脚本目录记录。")
        data_dir_path = script_dir
    full_output_path = os.path.join(data_dir_path, output_filename)
    # --- 文件名和路径设置结束 ---

    # --- 新增：视频源调整阶段 ---
    print("-" * 30)
    print("摄像头已打开。请调整您的视频源以获得最佳噪声效果。")
    print(f"  (例如，对准均匀的墙面，或轻微遮挡镜头。ROI区域为绿色框)")
    print(f"日志文件计划保存到: {full_output_path}")
    print("在摄像头预览窗口: 按 's' 键确认调整完毕。")
    print("                  按 'q' 键可直接退出整个脚本。")
    print("-" * 30)

    adjustment_window_name = "摄像头TRNG - 视频源调整阶段"
    adjustment_confirmed = False
    while True:
        ret, frame = cap.read()
        if not ret:
            print("错误: 无法读取摄像头画面用于调整。")
            cap.release()
            cv2.destroyAllWindows()
            exit()
        
        frame_adj_display = cv2.flip(frame, 1) # 调整时也翻转，保持一致性
        cv2.rectangle(frame_adj_display, (ROI_X, ROI_Y), (ROI_X + ROI_W, ROI_Y + ROI_H), (0, 255, 0), 2)
        cv2.putText(frame_adj_display, "调整视频源: 按 's' 确认调整, 按 'q' 退出脚本", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow(adjustment_window_name, frame_adj_display)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            print("在调整阶段用户选择退出脚本。")
            cap.release()
            cv2.destroyAllWindows()
            exit()
        elif key == ord('s'):
            print("视频源调整已确认。")
            adjustment_confirmed = True
            break 
    
    if not adjustment_confirmed: # 理论上不会执行到这里，因为上面是死循环或退出
        print("视频源调整未确认，脚本退出。")
        cap.release()
        cv2.destroyAllWindows()
        exit()
    
    # 调整完毕，等待用户在终端按Enter键开始正式计时
    cv2.destroyWindow(adjustment_window_name) # 关闭调整窗口
    print("-" * 30)
    input("请在此终端窗口按 Enter 键开始正式计时和生成数据...")
    print("-" * 30)
    # --- 视频源调整阶段结束 ---


    # --- 正式开始计时和数据生成 ---
    script_start_time_ts = time.time() # Unix timestamp for time calculations
    script_end_time_ts = script_start_time_ts + duration_seconds
    
    log_file = None
    try:
        log_file = open(full_output_path, "a", encoding="utf-8")
        log_file.write("GenerationTimestamp,HexValue,RealValue0To1\n")
        log_file.flush()

        print(f"开始计时和生成！脚本将运行约 {duration_seconds} 秒。")
        print(f"每次迭代后延迟 {DELAY_BETWEEN_ITERATIONS_SECONDS} 秒。")
        print(f"SHA256哈希: {'启用' if SHA256_HASH else '禁用'}")
        print("-" * 30)
        
        iteration_count = 0
        keep_script_running = True

        while time.time() < script_end_time_ts and keep_script_running:
            iteration_count += 1
            current_iter_start_time = time.time()
            # 确保在下一次迭代开始前，仍有足够的时间（至少是迭代延迟的2倍，粗略估计）
            # 这样可以避免在脚本即将结束时，还启动一个完整的、耗时的 generate_random_bits_from_camera 调用
            if script_end_time_ts - current_iter_start_time < DELAY_BETWEEN_ITERATIONS_SECONDS * 2 and iteration_count > 1: # 给点余量
                print(f"剩余时间不足以完成一次完整的迭代和延迟，提前结束。")
                break

            print(f"\n迭代 {iteration_count} (脚本剩余时间约 {script_end_time_ts - current_iter_start_time:.1f} 秒)...")
            
            generated_data_hex = generate_random_bits_from_camera(cap, SAMPLE_SIZE_BITS) 

            if generated_data_hex == "USER_QUIT_ITERATION":
                user_choice = input("用户中断了当前轮次。按 'c' 继续下一轮, 或按 'e' 退出整个脚本: ").strip().lower()
                if user_choice == 'e':
                    print("用户选择退出脚本。")
                    keep_script_running = False
                continue

            # 检查此轮迭代是否超出总运行时间（在数据生成后检查）
            current_op_time = time.time()
            if current_op_time > script_end_time_ts and generated_data_hex is not None:
                print(f"  此轮数据生成完成时间 ({datetime.fromtimestamp(current_op_time).strftime('%H:%M:%S.%f')[:-3]}) 超出脚本总结束时间 ({datetime.fromtimestamp(script_end_time_ts).strftime('%H:%M:%S.%f')[:-3]})，数据已丢弃。")
                generated_data_hex = None

            if generated_data_hex is not None:
                generation_timestamp_obj = datetime.now() # 记录数据生成/处理完成的精确时间
                generation_timestamp_str = generation_timestamp_obj.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                real_value_0_1 = map_hex_to_real01(generated_data_hex)
                
                real_value_str = f"{real_value_0_1:.18f}" if real_value_0_1 is not None else "N/A"

                log_entry = f"{generation_timestamp_str},{generated_data_hex},{real_value_str}\n"
                log_file.write(log_entry)
                log_file.flush()
                print(f"  记录: {generation_timestamp_str}, 哈希: {generated_data_hex[:16]}..., 实数[0,1): {real_value_str}") # 缩短哈希显示

            else:
                if keep_script_running :
                    print(f"  迭代 {iteration_count}: 未能生成有效数据或数据已丢弃。")
            
            if time.time() < script_end_time_ts and keep_script_running:
                time_remaining_in_script = script_end_time_ts - time.time()
                actual_delay = min(DELAY_BETWEEN_ITERATIONS_SECONDS, time_remaining_in_script)
                if actual_delay > 0:
                    time.sleep(actual_delay)
        
        print("-" * 30)
        print("脚本运行时间已到或用户选择退出。")

    except KeyboardInterrupt:
        print("\n用户通过 Ctrl+C 中断程序。")
    finally:
        if log_file:
            log_file.close()
            print(f"日志文件 {full_output_path} 已关闭。")
        cap.release()
        cv2.destroyAllWindows()
        print("摄像头已释放，所有窗口已关闭。")