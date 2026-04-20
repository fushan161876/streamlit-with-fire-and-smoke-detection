import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "alerts.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            camera_name TEXT,
            type TEXT,
            confidence REAL,
            snapshot_path TEXT,
            status TEXT
        )
    ''')
    conn.commit()
    conn.close()

def insert_alert(alert_id, timestamp, camera_name, alert_type, confidence, snapshot_path):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO alerts (id, timestamp, camera_name, type, confidence, snapshot_path, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (alert_id, timestamp, camera_name, alert_type, confidence, snapshot_path, '未处理'))
    conn.commit()
    conn.close()

def update_alert_status(alert_id, status):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE alerts SET status = ? WHERE id = ?
    ''', (status, alert_id))
    conn.commit()
    conn.close()

def get_all_alerts():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM alerts ORDER BY timestamp DESC')
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]
