import sqlite3
import json
import os
import numpy as np

class CalligraphyDB:
    def __init__(self, db_path="data/calligraphy.db"):
        self.db_path = db_path
        self.conn = None
        self._initialize_db()
    
    def _initialize_db(self):
        """初始化数据库"""
        # 确保目录存在
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        # 创建标准字符特征表 - 修复表名错误
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS standard_chars (
            char_code TEXT PRIMARY KEY,
            character TEXT,
            font_style TEXT,
            stroke_features TEXT,
            structure_features TEXT
        )
        """)
        
        # 创建用户作品表 - 修复表名错误
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT,
            char_code TEXT,
            score REAL,
            features TEXT,
            FOREIGN KEY (char_code) REFERENCES standard_chars(char_code)
        )
        """)
        
        self.conn.commit()
    
    def insert_standard_char(self, char_code, character, font_style, features):
        """插入标准字符特征"""
        cursor = self.conn.cursor()
        cursor.execute("""
        INSERT OR REPLACE INTO standard_chars 
        (char_code, character, font_style, stroke_features, structure_features)
        VALUES (?, ?, ?, ?, ?)
        """, (
            char_code,
            character,
            font_style,
            json.dumps(features["stroke"]),
            json.dumps(features["structure"])
        ))
        self.conn.commit()
    
    def get_standard_char_features(self, char_code, font_style="regular"):
        """获取标准字符特征"""
        cursor = self.conn.cursor()
        cursor.execute("""
        SELECT stroke_features, structure_features 
        FROM standard_chars 
        WHERE char_code=? AND font_style=?
        """, (char_code, font_style))
        
        result = cursor.fetchone()
        if result:
            return {
                "stroke": json.loads(result[0]),
                "structure": json.loads(result[1])
            }
        return None
    
    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()