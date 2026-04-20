#!/bin/bash
cd "$(dirname "$0")"

echo "停止后端 WebSocket + 推理服务..."
pkill -f "python3 backend/server.py" || true

echo "停止前端 Streamlit 服务..."
pkill -f "streamlit run frontend/app.py" || true

echo "系统已停止!"
