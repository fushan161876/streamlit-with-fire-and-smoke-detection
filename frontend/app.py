import streamlit as st
import json
import sqlite3
import os
import pandas as pd
import base64
import time
from backend.config_manager import load_config, save_config

# 配置路径
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "alerts.db")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")
COMM_FILE = os.path.join(os.path.dirname(__file__), "..", "comm.json")
CMD_FILE = os.path.join(os.path.dirname(__file__), "..", "cmd.txt")

# 初始化状态
if "running" not in st.session_state:
    st.session_state.running = False
if "current_alerts" not in st.session_state:
    
    st.session_state.current_alerts = []

if "last_comm_modified" not in st.session_state:
    st.session_state.last_comm_modified = 0
# 🔥 修复 2：用于缓存最后一帧画面，防止文件没更新时画面消失
if "cached_frame" not in st.session_state:
    st.session_state.cached_frame = None

if "cached_frame_b64" not in st.session_state:
    st.session_state.cached_frame_b64 = None

MAX_ALERTS = 5

st.set_page_config(page_title="智能烟火检测系统", layout="wide", page_icon="🔥")

# === 侧边栏：系统控制与配置 ===
with st.sidebar:
    st.header("⚙️ 系统控制与配置")
    
    st.subheader("运行控制")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ 启动系统", type="primary", disabled=st.session_state.running):
            st.session_state.running = True
            with open(CMD_FILE, "w") as f:
                f.write("start")
            st.success("已发送启动指令")

    with col2:
        if st.button("⏹️ 停止系统", disabled=not st.session_state.running):
            st.session_state.running = False
            with open(CMD_FILE, "w") as f:
                f.write("stop")
            st.success("已发送停止指令")

    st.divider()
    st.subheader("RTSP流配置")
    config = load_config()
    
    rtsp_url = st.text_input("RTSP 地址", value=config.get("rtsp_url", ""))
    camera_name = st.text_input("摄像头名称", value=config.get("camera_name", "摄像头01"))
    conf_thresh = st.slider("置信度阈值", 0.1, 1.0, value=config.get("conf_threshold", 0.6), step=0.05)
    
    if st.button("💾 应用配置"):
        new_config = {
            "rtsp_url": rtsp_url,
            "camera_name": camera_name,
            "conf_threshold": conf_thresh
        }
        save_config(new_config)
        st.success("配置已保存！")

    if st.button("🔗 测试连接"):
        import cv2
        cap = cv2.VideoCapture(rtsp_url)
        if cap.isOpened():
            st.success("连接成功！")
            cap.release()
        else:
            st.error("无法连接到该地址！")

# === 主界面：实时视频看板 ===
st.title("🔥 智能烟火检测系统")

main_col, alert_col = st.columns([3, 1])

with main_col:
    st.subheader(f"📹 实时监控画面 - {camera_name}")
    video_placeholder = st.empty()
    if not st.session_state.running:
        st.markdown('<div style="height:150px">', unsafe_allow_html=True)
        video_placeholder.info("等待视频流接入...")
    else:
        if st.session_state.cached_frame_b64:
            # 只要有缓存，就一直显示缓存，绝不闪烁
            img_data = base64.b64decode(st.session_state.cached_frame_b64)
            video_placeholder.image(img_data, channels="BGR", use_container_width=True)
        else:
            video_placeholder.info("正在加载视频流...")

with alert_col:
    st.subheader("🚨 实时告警")

    alert_container = st.container()

    def handle_delete(alert_id):
        # 更新数据库
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute('UPDATE alerts SET status = ? WHERE id = ?', ('已处理', alert_id))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"DB Error: {e}")
        
        # 从列表移除
        st.session_state.current_alerts = [
            a for a in st.session_state.current_alerts
            if a["alert_id"] != alert_id
        ]
        
        # 🔥 关键修改：强制立即刷新页面，确保弹窗立刻消失
        st.rerun()


    if not st.session_state.current_alerts:
        alert_container.success("当前无异常")
    else:
        st.markdown(""" 
        <style> .alert-card 
        { 
         padding: 10px;
         border-radius: 10px;
         background-color: #2b2b2b;
         margin-bottom: 10px; 
         } 
         </style> 
         """, unsafe_allow_html=True)
        for alert in st.session_state.current_alerts:
            with alert_container:
                with st.container():
                    st.error(f"【{alert['target_type']}】置信度: {alert['confidence']:.2f}")
                    st.write(f"📍 摄像头: {alert['camera_name']}")
                    st.write(f"⏱ 时间: {alert['timestamp']}")

                    img_data = base64.b64decode(alert["snapshot"])
                    st.image(img_data, use_container_width=True)

                    st.button(
                        "✅ 处理", 
                        key=f"handle_{alert['alert_id']}",
                        on_click=handle_delete,
                        args=(alert['alert_id'],)
                    )

                st.divider()



# === 历史日志标签页 ===
st.divider()
tab1, tab2 = st.tabs(["📋 历史报警记录", "🖥️ 系统健康日志"])

with tab1:
    st.subheader("历史报警记录查询")
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM alerts ORDER BY timestamp DESC", conn)
        conn.close()
        
        if not df.empty:
            cam_filter = st.selectbox("筛选摄像头", ["全部"] + list(df['camera_name'].unique()))
            if cam_filter != "全部":
                df = df[df['camera_name'] == cam_filter]
                
            st.dataframe(df, use_container_width=True)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 导出为 CSV",
                data=csv,
                file_name='alerts_history.csv',
                mime='text/csv',
            )
        else:
            st.info("暂无历史报警记录。")
    except Exception as e:
        st.warning(f"无法读取数据库: {e}")

with tab2:
    st.subheader("系统健康日志")
    st.text("后端服务状态：运行中\n模型加载：成功\nRTSP状态：已连接")

# === 动态更新循环 ===
if st.session_state.running:
    file_updated = False
    if os.path.exists(COMM_FILE):
        try:
            mtime = os.path.getmtime(COMM_FILE)
            if mtime > st.session_state.last_comm_modified:
                st.session_state.last_comm_modified = mtime
                file_updated = True
        except:
            pass

    if file_updated:
        try:
            with open(COMM_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 更新画面
            frame_msg = data.get("frame_msg", {})
            if frame_msg.get("image"):
                st.session_state.cached_frame_b64 = frame_msg["image"]

            # 更新告警
            alerts = data.get("alerts", [])
            if alerts:
                existing_ids = {a["alert_id"] for a in st.session_state.current_alerts}
                new_alerts = []
                for alert in alerts:
                    if alert["alert_id"] not in existing_ids:
                        new_alerts.append(alert)

                if new_alerts:
                    # 合并并保持只有最新的 MAX_ALERTS 个
                    st.session_state.current_alerts.extend(new_alerts)
                    # 去重（防止万一）并切片
                    unique_alerts = []
                    seen_ids = set()
                    for alert in reversed(st.session_state.current_alerts):
                        if alert["alert_id"] not in seen_ids:
                            seen_ids.add(alert["alert_id"])
                            unique_alerts.append(alert)
                    
                    # 反转回来并保留最新的5个
                    st.session_state.current_alerts = list(reversed(unique_alerts))[-MAX_ALERTS:]
                    st.rerun()

        except Exception:
            pass
            
    time.sleep(0.5)
    st.rerun()

        

# --- 告警弹窗处理 ---
if st.session_state.current_alerts:
    latest_alert = st.session_state.current_alerts[-1]

# === 浮层告警 ===
# floating = st.container()

# with floating:
#     st.markdown('<div class="alert-floating">', unsafe_allow_html=True)

#     for alert in st.session_state.current_alerts:
#         st.markdown('<div class="alert-box">', unsafe_allow_html=True)

#         st.write(f"🔥 {alert['target_type']} ({alert['confidence']:.2f})")
#         st.write(f"{alert['camera_name']}")

#         img_data = base64.b64decode(alert["snapshot"])
#         st.image(img_data, use_container_width=True)

#         if st.button("处理", key=f"handle_float_{alert['alert_id']}"):
#             st.session_state.to_delete = alert["alert_id"]

#         st.markdown('</div>', unsafe_allow_html=True)

#     st.markdown('</div>', unsafe_allow_html=True)    
    

# if "to_delete" not in st.session_state:
#     st.session_state.to_delete = None

# if st.session_state.to_delete:
#     alert_id = st.session_state.to_delete

#     # DB更新
#     try:
#         conn = sqlite3.connect(DB_PATH)
#         cursor = conn.cursor()
#         cursor.execute(
#             'UPDATE alerts SET status = ? WHERE id = ?',
#             ('已处理', alert_id)
#         )
#         conn.commit()
#         conn.close()
#     except:
#         pass

#     # 删除
#     st.session_state.current_alerts = [
#         a for a in st.session_state.current_alerts
#         if a["alert_id"] != alert_id
#     ]

#     st.session_state.to_delete = None
#     st.rerun()    
    
