import cv2
import numpy as np
from scipy.spatial import distance

class FeatureExtractor:
    def extract_stroke_features(self, img):
        """提取笔画特征"""
        # 确保图像是二值化的
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        # 计算距离变换
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
        
        # 骨架化
        skeleton = self.thin_font(binary)
        
        # 曲率分析
        curvature = self.calculate_curvature(skeleton) if skeleton is not None else []
        
        return {
            "stroke_width_mean": float(np.mean(dist_transform[dist_transform > 0])),
            "stroke_width_std": float(np.std(dist_transform[dist_transform > 0])),
            "curvature_mean": float(np.mean(curvature)) if curvature else 0.0,
            "curvature_std": float(np.std(curvature)) if curvature else 0.0
        }
    
    def thin_font(self, img):
        """优化后的骨架化方法"""
        # 确保图像是二值化的
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        # 使用更稳定的骨架化方法
        skeleton = cv2.ximgproc.thinning(
            binary, 
            thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
        )
        
        return skeleton
    
    def zhang_suen_thinning(self, img):
        """实现Zhang-Suen细化算法"""
        # 确保图像是二值化的
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        # 初始化细化后的图像
        thinned = binary.copy()
        
        # 设置算法参数
        height, width = thinned.shape[:2]
        changed = True
        
        while changed:
            changed = False
            
            # 第一步
            markers = set()
            for y in range(1, height - 1):
                for x in range(1, width - 1):
                    p2, p3, p4 = thinned[y-1, x], thinned[y-1, x+1], thinned[y, x+1]
                    p5, p6, p7 = thinned[y+1, x+1], thinned[y+1, x], thinned[y+1, x-1]
                    p8, p9, p1 = thinned[y, x-1], thinned[y-1, x-1], thinned[y, x]
                    
                    if p1 == 0:  # 只处理前景像素
                        continue
                    
                    # 计算条件
                    bp = sum([p2, p3, p4, p5, p6, p7, p8, p9])
                    if bp < 2 or bp > 6:
                        continue
                    
                    ap = 0
                    transitions = [(p2, p3), (p3, p4), (p4, p5), (p5, p6), 
                                  (p6, p7), (p7, p8), (p8, p9), (p9, p2)]
                    for (a, b) in transitions:
                        if a == 0 and b == 255:
                            ap += 1
                    
                    if ap != 1:
                        continue
                    
                    if p2 * p4 * p6 == 0 and p4 * p6 * p8 == 0:
                        markers.add((y, x))
            
            # 删除标记点
            for (y, x) in markers:
                thinned[y, x] = 0
                changed = True
            
            # 第二步
            markers = set()
            for y in range(1, height - 1):
                for x in range(1, width - 1):
                    p2, p3, p4 = thinned[y-1, x], thinned[y-1, x+1], thinned[y, x+1]
                    p5, p6, p7 = thinned[y+1, x+1], thinned[y+1, x], thinned[y+1, x-1]
                    p8, p9, p1 = thinned[y, x-1], thinned[y-1, x-1], thinned[y, x]
                    
                    if p1 == 0:  # 只处理前景像素
                        continue
                    
                    # 计算条件
                    bp = sum([p2, p3, p4, p5, p6, p7, p8, p9])
                    if bp < 2 or bp > 6:
                        continue
                    
                    ap = 0
                    transitions = [(p2, p3), (p3, p4), (p4, p5), (p5, p6), 
                                  (p6, p7), (p7, p8), (p8, p9), (p9, p2)]
                    for (a, b) in transitions:
                        if a == 0 and b == 255:
                            ap += 1
                    
                    if ap != 1:
                        continue
                    
                    if p2 * p4 * p8 == 0 and p2 * p6 * p8 == 0:
                        markers.add((y, x))
            
            # 删除标记点
            for (y, x) in markers:
                thinned[y, x] = 0
                changed = True
        
        return thinned
    
    def calculate_curvature(self, skeleton):
        """计算曲率"""
        contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return []
        
        curvature = []
        for contour in contours:
            for i in range(1, len(contour)-1):
                p1 = contour[i-1][0]
                p2 = contour[i][0]
                p3 = contour[i+1][0]
                
                # 计算曲率
                v1 = p1 - p2
                v2 = p3 - p2
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-5)
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                curvature.append(float(angle))  # 转换为Python float
        
        return curvature
    
    def analyze_structure(self, img):
        """九宫格结构分析"""
        height, width = img.shape
        grid_features = []
        
        for i in range(3):
            for j in range(3):
                # 提取当前网格区域
                x1, x2 = j*width//3, (j+1)*width//3
                y1, y2 = i*height//3, (i+1)*height//3
                grid = img[y1:y2, x1:x2]
                
                # 计算网格内像素密度
                density = np.sum(grid > 0) / (grid.size + 1e-5)
                
                # 计算重心偏移
                offset = 0.0
                if np.any(grid > 0):
                    y_indices, x_indices = np.where(grid > 0)
                    center_x = np.mean(x_indices) / grid.shape[1]
                    center_y = np.mean(y_indices) / grid.shape[0]
                    offset = distance.euclidean((center_x, center_y), (0.5, 0.5))
                
                grid_features.append({
                    "density": float(density),
                    "center_offset": float(offset)
                })
        
        return grid_features
    
    def extract_all_features(self, img):
        """提取所有特征"""
        stroke_features = self.extract_stroke_features(img)
        structure_features = self.analyze_structure(img)
        
        return {
            "stroke": stroke_features,
            "structure": structure_features
        }