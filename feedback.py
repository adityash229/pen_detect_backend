"""
Handles SQLite feedback storage and retrieval.
"""

import sqlite3
from datetime import datetime

DB_NAME = "feedback.db"

def init_db():
    """Initialize feedback table."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS detections(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            confidence REAL,
            status TEXT,
            user_label TEXT
        )
    """)
    conn.commit()
    conn.close()

def insert_feedback(confidence, status, user_label):
    """Insert user feedback into database."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("""
        INSERT INTO detections(timestamp, confidence, status, user_label)
        VALUES (?, ?, ?, ?)
    """, (datetime.now().isoformat(), confidence, status, user_label))

    conn.commit()
    conn.close()

def get_feedback():
    """Retrieve all feedback records."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("SELECT * FROM detections ORDER BY timestamp DESC")
    data = c.fetchall()

    conn.close()
    return data