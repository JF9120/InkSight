import cv2
import numpy as np
from scipy import ndimage

class ArtEvaluator:
    def __init__(self):
        pass
    
    def evaluate_artistic_features(self, image, original_gray=None):
        """
        评估艺术特征：顿笔、笔锋、墨色梯度等
        :param image: 预处理后的二值图像
        :param original_gray: 原始灰度图像（用于墨色梯度分析）
        """
        # 预处理确保图像是二值化的
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        
        # 反转颜色：文字为白色(255)，背景为黑色(0)
        binary = 255 - binary
        
        # 1. 顿笔检测 - 笔画起始/结束处的宽度变化
        pen_pressure_score = self.detect_pen_pressure(binary)
        
        # 2. 笔锋检测 - 笔画末端的尖锐程度
        stroke_tip_score = self.detect_stroke_tips(binary)
        
        # 3. 笔画流畅度 - 曲率变化
        stroke_fluency_score = self.detect_stroke_fluency(binary)
        
        # 4. 墨色梯度检测
        ink_gradient_score = 0.0
        if original_gray is not None:
            # 确保原始灰度图像与掩码尺寸一致
            if binary.shape != original_gray.shape:
                # 调整掩码尺寸以匹配原始图像
                resized_binary = cv2.resize(binary, (original_gray.shape[1], original_gray.shape[0]))
            else:
                resized_binary = binary
            
            # 创建笔画区域的掩码
            ink_mask = resized_binary.copy()
            ink_gradient_score = self.detect_ink_gradient(original_gray, ink_mask)
        
        # 综合艺术得分
        art_score = (0.35 * pen_pressure_score + 
                     0.25 * stroke_tip_score + 
                     0.20 * stroke_fluency_score +
                     0.20 * ink_gradient_score)
        
        feedback = []
        if pen_pressure_score < 0.5:
            feedback.append("顿笔不够明显")
        if stroke_tip_score < 0.5:
            feedback.append("笔锋不够锐利")
        if stroke_fluency_score < 0.5:
            feedback.append("笔画流畅度不足")
        if ink_gradient_score < 0.4:
            feedback.append("墨色变化不足，缺乏层次感")
        
        return {
            "art_score": art_score,
            "pen_pressure": pen_pressure_score,
            "stroke_tips": stroke_tip_score,
            "stroke_fluency": stroke_fluency_score,
            "ink_gradient": ink_gradient_score,
            "feedback": "，".join(feedback) if feedback else "笔画艺术表现良好"
        }
    
    def detect_pen_pressure(self, image):
        """检测顿笔特征"""
        # 计算笔画宽度变化
        dist_transform = cv2.distanceTransform(image, cv2.DIST_L2, 3)
        dist_values = dist_transform[dist_transform > 0]
        
        if len(dist_values) == 0:
            return 0.0
        
        # 计算宽度变化的统计特征
        width_mean = np.mean(dist_values)
        width_std = np.std(dist_values)
        
        # 较大的标准差表示有顿笔变化
        return min(1.0, width_std / (width_mean * 0.5))
    
    def detect_stroke_tips(self, image):
        """检测笔锋特征"""
        # 使用骨架化找到笔画末端
        skeleton = self.thin_font(image)
        endpoints = self.find_endpoints(skeleton)
        
        if len(endpoints) == 0:
            return 0.0
        
        # 在端点周围分析尖锐程度
        tip_scores = []
        for y, x in endpoints:
            # 提取端点周围的小区域
            roi = image[max(0, y-5):min(y+6, image.shape[0]), 
                         max(0, x-5):min(x+6, image.shape[1])]
            
            if roi.size == 0:
                continue
                
            # 计算区域的梯度
            sobelx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(sobelx**2 + sobely**2)
            
            # 计算尖锐度得分
            if grad_mag.size > 0:
                max_grad = np.max(grad_mag)
                tip_scores.append(min(1.0, max_grad / 100.0))
        
        return np.mean(tip_scores) if tip_scores else 0.0
    
    def detect_stroke_fluency(self, image):
        """检测笔画流畅度"""
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return 0.0
        
        # 计算所有轮廓的曲率变化
        curvature_changes = []
        for contour in contours:
            if len(contour) < 10:
                continue
                
            # 计算曲率
            curvature = self.calculate_curvature(contour)
            if len(curvature) > 0:
                # 曲率变化的标准差越小，笔画越流畅
                curvature_std = np.std(curvature)
                curvature_changes.append(min(1.0, 1.0 / (curvature_std + 0.1)))
        
        return np.mean(curvature_changes) if curvature_changes else 0.0
    
    def thin_font(self, img):
        """骨架化方法"""
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        skeleton = cv2.ximgproc.thinning(
            binary, 
            thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
        )
        return skeleton
    
    def find_endpoints(self, skeleton):
        """在骨架图中找到端点"""
        kernel = np.array([[1, 1, 1],
                           [1, 10, 1],
                           [1, 1, 1]], dtype=np.uint8)
        
        # 使用卷积找到端点（只有一个邻居的点）
        conv = cv2.filter2D(skeleton, -1, kernel)
        endpoints = np.argwhere((skeleton > 0) & (conv == 11))
        return endpoints
    
    def calculate_curvature(self, contour):
        """计算轮廓的曲率"""
        curvature = []
        for i in range(1, len(contour)-1):
            p1 = contour[i-1][0]
            p2 = contour[i][0]
            p3 = contour[i+1][0]
            
            # 计算曲率
            v1 = p1 - p2
            v2 = p3 - p2
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-5)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            curvature.append(float(angle))
        
        return curvature
    
    def detect_ink_gradient(self, gray_image, binary_mask):
        """
        检测墨色梯度特征
        :param gray_image: 原始灰度图像
        :param binary_mask: 笔画区域的二值掩码
        :return: 墨色梯度得分 (0-1)
        """
        # 确保掩码是二值图像
        _, binary_mask = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)
        
        # 仅保留笔画区域
        stroke_area = cv2.bitwise_and(gray_image, gray_image, mask=binary_mask)
        
        # 计算墨色梯度
        sobelx = cv2.Sobel(stroke_area, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(stroke_area, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        
        # 归一化梯度值 (0-1)
        if np.max(grad_mag) > 0:
            grad_mag = grad_mag / np.max(grad_mag)
        
        # 计算有效梯度区域比例
        effective_gradient = np.sum(grad_mag > 0.3) / (np.sum(binary_mask > 0) + 1e-5)
        
        # 计算梯度变化的连贯性
        gradient_coherence = self.calculate_gradient_coherence(grad_mag)
        
        # 综合得分
        return min(1.0, 0.6 * effective_gradient + 0.4 * gradient_coherence)
    
    def calculate_gradient_coherence(self, grad_mag):
        """计算梯度变化的连贯性"""
        # 计算梯度方向一致性
        rows, cols = grad_mag.shape
        coherence = 0
        count = 0
        
        # 如果图像太小，直接返回0
        if rows < 3 or cols < 3:
            return 0.0
        
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                if grad_mag[i, j] > 0.3:
                    # 检查8邻域内梯度方向是否一致
                    local_grad = grad_mag[i-1:i+2, j-1:j+2]
                    if np.std(local_grad) < 0.2:
                        coherence += 1
                    count += 1
        
        return coherence / count if count > 0 else 0