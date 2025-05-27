import cv2
import numpy as np
from datetime import datetime
import os

# --- 配置参数 ---
NUM_LEDS = 4
ROI_SIZE = 20  # <--- Example: Changed to 20 for a smaller box
BRIGHTNESS_THRESHOLD = 150
OUTPUT_SUBDIR = "data"
CAMERA_INDEX = 0
font = cv2.FONT_HERSHEY_SIMPLEX # Moved font definition here as it's used in select_roi_callback too

# --- 全局变量 ---
rois = []
current_led_selection = 0
frame_copy_for_selection = None # Initialize to None

# --- 鼠标回调函数，用于选择ROI ---
def select_roi_callback(event, x, y, flags, param):
    global rois, current_led_selection, frame_copy_for_selection # frame_copy_for_selection needs to be global if modified here

    if event == cv2.EVENT_LBUTTONDOWN:
        if current_led_selection < NUM_LEDS:
            # 定义ROI的左上角点
            # 为了简单，我们使用一个固定大小的正方形ROI
            # x, y 是点击点，我们将它作为ROI的中心点
            roi_x = int(x - ROI_SIZE / 2)
            roi_y = int(y - ROI_SIZE / 2)
            
            # 确保ROI在画面内
            # Check if frame_copy_for_selection is not None before accessing shape
            if frame_copy_for_selection is not None:
                frame_height, frame_width = frame_copy_for_selection.shape[:2]
                roi_x = max(0, roi_x)
                roi_y = max(0, roi_y)
                # Ensure ROI does not go out of bounds on the right/bottom
                roi_x_end = min(frame_width, roi_x + ROI_SIZE)
                roi_y_end = min(frame_height, roi_y + ROI_SIZE)
                
                actual_roi_w = roi_x_end - roi_x
                actual_roi_h = roi_y_end - roi_y

                if actual_roi_w > 0 and actual_roi_h > 0:
                    rois.append((roi_x, roi_y, actual_roi_w, actual_roi_h))
                    print(f"LED {current_led_selection + 1} ROI selected: ({roi_x}, {roi_y}, {actual_roi_w}, {actual_roi_h})")
                    
                    # 在副本上绘制已选ROI，用于显示
                    cv2.rectangle(frame_copy_for_selection, (roi_x, roi_y), (roi_x + actual_roi_w, roi_y + actual_roi_h), (0, 255, 0), 2)
                    cv2.putText(frame_copy_for_selection, f"LED{current_led_selection+1}", (roi_x, roi_y - 5), font, 0.5, (0,255,0), 1)
                    
                    current_led_selection += 1
                else:
                    print("Selected ROI is too small or outside bounds (possibly due to click near edge). Try again.")
            else:
                print("Error: Frame for selection not available yet.")


        if current_led_selection == NUM_LEDS:
            print("All ROIs selected. Press 's' to start detection or 'r' to reselect.")

# --- 主程序 ---
def main():
    global rois, current_led_selection, frame_copy_for_selection # Added frame_copy_for_selection here too

    # --- 文件名和路径设置 ---
    # ... (this part remains the same as in the previous correct version) ...
    start_time = datetime.now()
    formatted_start_time = start_time.strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f"{formatted_start_time}.txt"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = os.path.join(script_dir, OUTPUT_SUBDIR)
    try:
        os.makedirs(data_dir_path, exist_ok=True)
        print(f"输出文件夹: {data_dir_path}")
    except OSError as e:
        print(f"创建目录 {data_dir_path} 失败: {e}")
        print("将尝试在当前脚本目录记录日志。")
        data_dir_path = script_dir
    full_output_path = os.path.join(data_dir_path, output_filename)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"错误：无法打开摄像头 {CAMERA_INDEX}")
        return

    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", select_roi_callback) # Now select_roi_callback is defined

    # ... (rest of the main function remains the same) ...
    print(f"请在弹出的窗口中依次点击 {NUM_LEDS} 个灯的中心位置来定义ROI。")
    print("每个ROI将是一个围绕点击点的 {}x{} 像素的区域。".format(ROI_SIZE, ROI_SIZE))
    print(f"日志文件将保存为: {full_output_path}")
    print("完成后，按 's' 键开始检测和记录。")
    print("按 'r' 键可以重新选择所有ROI。")
    print("按 'q' 键退出程序。")

    log_file = None
    detection_started = False
    last_binary_string = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            print("错误：无法读取视频帧")
            break
        
        if not detection_started or current_led_selection < NUM_LEDS:
            frame_copy_for_selection = frame.copy() # Assign frame_copy_for_selection here
            # ... (rest of the drawing logic for ROI selection) ...
            for i, (x_roi,y_roi,w_roi,h_roi) in enumerate(rois): # Renamed to avoid conflict with callback params
                cv2.rectangle(frame_copy_for_selection, (x_roi,y_roi), (x_roi+w_roi, y_roi+h_roi), (0,255,0), 2)
                cv2.putText(frame_copy_for_selection, f"LED{i+1}", (x_roi, y_roi-5), font, 0.5, (0,255,0), 1)
            
            if current_led_selection < NUM_LEDS:
                 cv2.putText(frame_copy_for_selection, f"Click to select LED {current_led_selection + 1}", (10, 30), font, 0.7, (255, 255, 0), 2)
            else:
                 cv2.putText(frame_copy_for_selection, "All ROIs set. Press 's' to start. 'r' to reset.", (10, 30), font, 0.7, (0, 255, 255), 2)
            display_frame = frame_copy_for_selection
        else:
            display_frame = frame.copy()


        if detection_started and len(rois) == NUM_LEDS:
            # ... (detection logic) ...
            binary_states = []
            for i, (x_roi, y_roi, w_roi, h_roi) in enumerate(rois): # Renamed to avoid conflict
                roi_frame = frame[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
                gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
                avg_brightness = np.mean(gray_roi)
                state = 1 if avg_brightness > BRIGHTNESS_THRESHOLD else 0
                binary_states.append(state)
                
                color = (0, 255, 0) if state == 1 else (0, 0, 255)
                cv2.rectangle(display_frame, (x_roi, y_roi), (x_roi + w_roi, y_roi + h_roi), color, 2)
                cv2.putText(display_frame, str(state), (x_roi + w_roi // 2 - 5, y_roi + h_roi // 2 + 5), font, 0.7, color, 2)
                cv2.putText(display_frame, f"L{i+1}", (x_roi, y_roi-5), font, 0.5, color, 1)

            current_binary_string = "".join(map(str, binary_states))
            cv2.putText(display_frame, f"Binary: {current_binary_string}", (10, frame.shape[0] - 40), font, 0.8, (255,0,0), 2)
            if rois: 
                 cv2.putText(display_frame, f"Avg Brightness (L1 for ref): {np.mean(cv2.cvtColor(frame[rois[0][1]:rois[0][1]+rois[0][3], rois[0][0]:rois[0][0]+rois[0][2]], cv2.COLOR_BGR2GRAY)):.1f}", (10, frame.shape[0] - 15), font, 0.6, (200,200,0), 1)


            if current_binary_string != last_binary_string:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                log_entry = f"{timestamp},{current_binary_string}\n"
                if log_file: 
                    log_file.write(log_entry)
                    log_file.flush()
                last_binary_string = current_binary_string
        
        cv2.imshow("Frame", display_frame)
        
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            if len(rois) == NUM_LEDS:
                if not detection_started:
                    print(f"开始检测，日志记录到: {full_output_path}") 
                    try:
                        log_file = open(full_output_path, "a") 
                        log_file.write("Timestamp,BinaryString\n")
                        detection_started = True
                    except IOError as e:
                        print(f"无法打开日志文件 {full_output_path}: {e}")
                else:
                    print("检测已开始。")
            else:
                print(f"请先选择所有 {NUM_LEDS} 个 ROI。")
        elif key == ord('r'):
            print("重置ROI。请重新选择。")
            rois = []
            current_led_selection = 0
            detection_started = False
            if log_file:
                log_file.close()
                log_file = None
            last_binary_string = ""

    if log_file:
        log_file.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()