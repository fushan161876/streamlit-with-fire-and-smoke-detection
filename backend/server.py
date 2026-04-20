import cv2
import numpy as np
import base64
import json
import time
import os
import uuid
import threading
import queue
from datetime import datetime
from db import init_db, insert_alert
from config_manager import load_config

# 🔥 仅保留必要的导入
from coco_utils_imgpath import COCO_test_helper
from utils_imgpath import post_process, IMG_SIZE

# 全局初始化
co_helper = COCO_test_helper(enable_letter_box=True)

# 配置路径
MODEL_PATH = "/zmn/streaminfer/models/day_total_260304.rknn" 
SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "..", "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
COMM_FILE = os.path.join(os.path.dirname(__file__), "..", "comm.json")

# 内存队列
frame_queue = queue.Queue(maxsize=10)
result_queue = queue.Queue(maxsize=10)

# 全局状态
running_flag = False
current_config = load_config()

def image_to_base64(img):
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buffer).decode('utf-8')

def capture_thread_func():
    global running_flag, current_config

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|"
    "analyzeduration;100000|"
    "log_level;fatal"
    )
    
    rtsp_url = current_config.get("rtsp_url", "")
    print(f"尝试连接流: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    use_mock = not cap.isOpened()
    if use_mock:
        print("无法打开RTSP流，降级为模拟虚拟视频流")
    
    frame_idx = 0
    while running_flag:
        if use_mock:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f"Mock RTSP Stream {datetime.now().strftime('%H:%M:%S')}", (50, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if np.random.random() < 0.1:
                cv2.rectangle(frame, (300, 200), (350, 250), (0, 0, 255), -1)
            time.sleep(1.0)
        else:
            ret, frame = cap.read()
            if not ret:
                time.sleep(2)
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                continue
            frame_idx += 1
            if frame_idx % 25 != 0:
                continue
        
        if not frame_queue.full():
            frame_queue.put(frame)
        else:
            try:
                frame_queue.get_nowait()
                frame_queue.put(frame)
            except queue.Empty:
                pass

    if not use_mock:
        cap.release()
    print("抓帧线程退出")

def inference_thread_func():
    global running_flag, current_config
    
    use_rknn = False
    print("【日志】推理线程启动，开始初始化 RKNN...")
    try:
        from rknnlite.api import RKNNLite
        rknn = RKNNLite(verbose=False)
        if rknn.load_rknn(MODEL_PATH) == 0 and rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) == 0:
            use_rknn = True
            print("RKNN 初始化成功")
    except Exception as e:
        print(f"RKNN 加载异常: {e}")

    conf_thresh = current_config.get("conf_threshold", 0.6)
    cam_name = current_config.get("camera_name", "默认摄像头")

    while running_flag:
        try:
            frame = frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        detections = []
        if use_rknn:
            try:
                # 1. 清理上一帧的缓存，防止内存泄漏
                if co_helper.letter_box_info_list:
                    co_helper.letter_box_info_list.pop()
                
                # 2. 预处理
                img_lb = co_helper.letter_box(im=frame.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0,0,0))
                img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
                inp = np.expand_dims(img_rgb, axis=0)

                # 3. 推理
                outputs = rknn.inference(inputs=[inp])

                # 4. 后处理
                boxes, classes, scores = post_process(outputs)

                # 🔥 5. 【关键】第一步先判断是否为 None
                if boxes is not None and len(boxes) > 0:
                    
                    # 6. 置信度过滤
                    valid_indices = scores >= conf_thresh
                    boxes_filtered = boxes[valid_indices]
                    classes_filtered = classes[valid_indices]
                    scores_filtered = scores[valid_indices]
                    
                    if len(boxes_filtered) > 0:
                        # 7. 坐标还原
                        real_boxes = co_helper.get_real_box(boxes_filtered)
                        
                        # 8. 只保留 fire (class 0)
                        fire_mask = classes_filtered == 0
                        final_boxes = real_boxes[fire_mask]
                        final_scores = scores_filtered[fire_mask]
                        final_classes = classes_filtered[fire_mask]

                        # 9. 组装结果
                        for box, score, cls in zip(final_boxes, final_scores, final_classes):
                            x1, y1, x2, y2 = box
                            detections.append({
                                "box": [int(x1), int(y1), int(x2), int(y2)],
                                "conf": float(score),
                                "class": int(cls),
                                "label": "fire"
                            })
                            
            except Exception as e:
                print(f"【日志】❌ RKNN推理异常: {e}")
                import traceback
                traceback.print_exc()
                detections = []

        # 以下是画框和保存逻辑
        alerts_this_frame = []
        timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        for det in detections:
            single_alert_frame = frame.copy()  
            x1, y1, x2, y2 = map(int, det["box"])
            conf = det["conf"]
            label = det["label"]
            
            cv2.rectangle(single_alert_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(single_alert_frame, f"{label} {conf:.2f}", (x1, max(y1-10, 0)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            alert_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"
            snap_filename = f"{alert_id}.jpg"
            snap_path = os.path.join(SNAPSHOT_DIR, snap_filename)
            cv2.imwrite(snap_path, single_alert_frame) 
            
            insert_alert(alert_id, timestamp_str, cam_name, label, conf, snap_path)
            
            alerts_this_frame.append({
                "type": "alert",
                "alert_id": alert_id,
                "timestamp": timestamp_str,
                "camera_name": cam_name,
                "target_type": label,
                "confidence": conf,
                "snapshot": image_to_base64(single_alert_frame)
            })
        
        res_data = {
            "frame_msg": {
                "type": "frame",
                "timestamp": timestamp_str,
                "camera_id": cam_name,
                "image": image_to_base64(frame)
            },
            "alerts": alerts_this_frame
        }
        
        with open(COMM_FILE, "w", encoding="utf-8") as f:
            json.dump(res_data, f)

    if use_rknn:
        rknn.release()
    print("推理线程退出")

thread_cap = None
thread_inf = None

def start_system():
    global running_flag, thread_cap, thread_inf
    if running_flag:
        return
    running_flag = True
    
    while not frame_queue.empty(): frame_queue.get_nowait()
    while not result_queue.empty(): result_queue.get_nowait()
    
    thread_cap = threading.Thread(target=capture_thread_func, daemon=True)
    thread_inf = threading.Thread(target=inference_thread_func, daemon=True)
    thread_cap.start()
    thread_inf.start()
    print("系统启动拉流与推理")

def stop_system():
    global running_flag
    running_flag = False
    print("发出系统停止信号")

def main():
    init_db()
    CMD_FILE = os.path.join(os.path.dirname(__file__), "..", "cmd.txt")
    print("后端服务已启动，等待指令...")
    try:
        while True:
            if os.path.exists(CMD_FILE):
                with open(CMD_FILE, "r") as f:
                    cmd = f.read().strip()
                os.remove(CMD_FILE)
                if cmd == "start":
                    start_system()
                elif cmd == "stop":
                    stop_system()
            time.sleep(1)
    except KeyboardInterrupt:
        stop_system()
        print("后端服务退出")

if __name__ == "__main__":
    main()