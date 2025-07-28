import numpy as np
from .database import CalligraphyDB

class CalligraphyEvaluator:
    def __init__(self, db_path="data/calligraphy.db", font_style="regular"):
        self.db = CalligraphyDB(db_path)
        self.font_style = font_style
    
    def evaluate(self, features, char_code):
        """评价书法作品"""
        # 获取标准特征
        standard_features = self.db.get_standard_char_features(char_code, self.font_style)
        if not standard_features:
            return None
        
        # 计算笔画得分
        stroke_score = self.calculate_stroke_score(
            features.get("stroke", {}), 
            standard_features.get("stroke", {})
        )
        
        # 计算结构得分
        structure_score = self.calculate_structure_score(
            features.get("structure", []), 
            standard_features.get("structure", [])
        )
        
        # 综合评分
        total_score = 0.6 * stroke_score + 0.4 * structure_score
        
        return {
            "total_score": total_score,
            "stroke_score": stroke_score,
            "structure_score": structure_score,
            "details": self.generate_details(features, standard_features)
        }
    
    def calculate_stroke_score(self, user, standard):
        """计算笔画得分"""
        # 添加默认值处理
        user_width_mean = user.get("stroke_width_mean", 0)
        std_width_mean = standard.get("stroke_width_mean", 0)
        user_width_std = user.get("stroke_width_std", 0)
        std_width_std = standard.get("stroke_width_std", 0)
        user_curvature_mean = user.get("curvature_mean", 0)
        std_curvature_mean = standard.get("curvature_mean", 0)
        # 宽度相似度
        width_sim = 1 - abs(user["stroke_width_mean"] - standard["stroke_width_mean"]) / max(standard["stroke_width_mean"], 1)
        
        # 宽度均匀性
        width_uniformity = 1 - (user["stroke_width_std"] / max(standard["stroke_width_std"], 1))
        
        # 曲率相似度
        curvature_sim = 1 - abs(user["curvature_mean"] - standard["curvature_mean"]) / max(standard["curvature_mean"], 0.1)
        
        # 组合得分
        return max(0, min(1, 0.4 * width_sim + 0.3 * width_uniformity + 0.3 * curvature_sim))
    
    def calculate_structure_score(self, user, standard):
        """计算结构得分"""
        total_score = 0
        for i in range(9):
            # 密度相似度
            density_sim = 1 - abs(user[i]["density"] - standard[i]["density"])
            
            # 重心偏移相似度
            offset_sim = 1 - abs(user[i]["center_offset"] - standard[i]["center_offset"])
            
            total_score += 0.6 * density_sim + 0.4 * offset_sim
        
        return max(0, min(1, total_score / 9))
    
    def generate_details(self, user, standard):
        """生成详细评价"""
        details = {
            "stroke": {
                "width_mean": (user["stroke"]["stroke_width_mean"], standard["stroke"]["stroke_width_mean"]),
                "width_std": (user["stroke"]["stroke_width_std"], standard["stroke"]["stroke_width_std"]),
                "curvature_mean": (user["stroke"]["curvature_mean"], standard["stroke"]["curvature_mean"])
            },
            "structure": []
        }
        
        for i in range(9):
            details["structure"].append({
                "density": (user["structure"][i]["density"], standard["structure"][i]["density"]),
                "center_offset": (user["structure"][i]["center_offset"], standard["structure"][i]["center_offset"])
            })
        
        return details
    
    def close(self):
        """关闭数据库"""
        self.db.close()