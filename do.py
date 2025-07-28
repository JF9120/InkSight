from utils.preprocessor import ImagePreprocessor
from core.feature_extractor import FeatureExtractor
import os
import hashlib
import joblib
import numpy as np

class ProcessingPipeline:
    def __init__(self, cache_dir="cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.preprocessor = ImagePreprocessor()
        self.extractor = FeatureExtractor()
    
    def process_image(self, image_path):
        """处理单个图像：预处理 + 特征提取"""
        # 生成缓存键
        with open(image_path, "rb") as f:
            file_content = f.read()
        hash_key = hashlib.md5(file_content).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"{hash_key}.pkl")
        
        # 检查缓存
        if os.path.exists(cache_file):
            try:
                return joblib.load(cache_file)
            except:
                # 缓存文件损坏，重新处理
                pass
        
        # 无缓存则处理
        img = self.preprocessor.preprocess(image_path)
        features = self.extractor.extract_all_features(img)
        
        result = {
            "preprocessed": img,
            "features": features
        }
        
        # 保存缓存
        joblib.dump(result, cache_file)
        return result