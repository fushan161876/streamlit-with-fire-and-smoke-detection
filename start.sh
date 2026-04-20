#!/bin/bash
cd "$(dirname "$0")"

# 停止可能残留的进程
pkill -f "streamlit run frontend/app.py" || true
pkill -f "python3 backend/server.py" || true

echo "启动后端 WebSocket + 推理服务..."
python3 backend/server.py &
BACKEND_PID=$!

echo "启动前端 Streamlit 服务..."
PYTHONPATH=$(pwd) streamlit run frontend/app.py --server.port 8051 --server.address 0.0.0.0 &
FRONTEND_PID=$!

echo "系统已启动!"
echo "后端 PID: $BACKEND_PID"
echo "前端 PID: $FRONTEND_PID"
echo "请访问: http://127.0.0.1:8051"

wait $FRONTEND_PID
