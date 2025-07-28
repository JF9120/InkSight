import cv2
import numpy as np
from PIL import Image, ExifTags
class ImagePreprocessor:
    def __init__(self, target_size=(128, 128)):
        self.target_size = target_size
    
    def load_image(self, image_path):
        """改进的图像加载函数，处理JPG格式问题"""
        try:
            # 使用PIL读取图像以处理EXIF方向信息
            pil_img = Image.open(image_path)
            
            # 检查并应用EXIF方向信息
            try:
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == 'Orientation':
                        break
                exif = dict(pil_img._getexif().items())
                if exif[orientation] == 3:
                    pil_img = pil_img.rotate(180, expand=True)
                elif exif[orientation] == 6:
                    pil_img = pil_img.rotate(270, expand=True)
                elif exif[orientation] == 8:
                    pil_img = pil_img.rotate(90, expand=True)
            except (AttributeError, KeyError, IndexError):
                # 没有EXIF信息
                pass
            
            # 转换为OpenCV格式 (BGR)
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            # 转换为灰度图
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            return img
        except Exception as e:
            raise ValueError(f"无法加载图像 {image_path}: {str(e)}")
    
    def preprocess(self, image):
        """完整的预处理流程"""
        if isinstance(image, str):
            img = self.load_image(image)
        else:
            img = image.copy()
        
        img = self.to_grayscale(img)
        img = self.remove_noise(img)
        img = self.binarize(img)
        img = self.normalize_size(img)
        return img
    
    def to_grayscale(self, img):
        """转换为灰度图"""
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    
    def remove_noise(self, img, kernel_size=3):
        """去噪"""
        return cv2.medianBlur(img, kernel_size)
    
    def binarize(self, img):
        """自适应二值化 - 修复方法"""
        # 确保图像是8位单通道
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        
        # 使用更稳定的二值化方法
        return cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
    
    def normalize_size(self, img):
        """尺寸归一化"""
        return cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
    