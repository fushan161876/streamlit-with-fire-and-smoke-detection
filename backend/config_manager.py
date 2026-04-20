import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")

def load_config():
    if not os.path.exists(CONFIG_PATH):
        # 默认配置
        default_config = {
            "rtsp_url": "rtsp://127.0.0.1:8554/stream", # 默认虚拟流
            "camera_name": "默认摄像头",
            "conf_threshold": 0.6
        }
        save_config(default_config)
        return default_config
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_config(config_data):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4)
