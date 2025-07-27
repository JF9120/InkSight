import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import cv2
import os
import numpy as np
from do import ProcessingPipeline
from core.evaluator import CalligraphyEvaluator
from core.database import CalligraphyDB
import requests
import base64
import json
import sqlite3
from core.art import ArtEvaluator

class CalligraphyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("硬笔书法智能评价系统")
        self.root.geometry("1000x700")
        
        # 初始化组件
        self.processor = ProcessingPipeline()
        self.evaluator = None
        self.db = None
        self.current_image_path = None
        self.current_features = None
        self.char_code = None
        self.font_style = tk.StringVar(value="regular")  # 默认字体样式
        
        # 检查数据库目录
        os.makedirs("data", exist_ok=True)
        # 添加项目根目录变量
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        
        # 百度OCR配置
        self.ocr_config = {
            "api_key": "",
            "secret_key": "",
            "token_url": "https://aip.baidubce.com/oauth/2.0/token",
            "ocr_url": "https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic"
        }
        
        self.create_widgets()
    
    def generate_char_map(self):
        """使用Unicode编码生成字符映射表"""
        char_map = {}
        
        # 覆盖Unicode基本多文种平面（U+0000 到 U+FFFF）
        for code_point in range(0x4E00, 0x9FFF + 1):  # 汉字区
            char = chr(code_point)
            char_code = hex(code_point)[2:].upper().zfill(4)
            char_map[char_code] = char
        
        # 添加ASCII字符
        for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789":
            char_code = hex(ord(char))[2:].upper().zfill(4)
            char_map[char_code] = char
        
        # 添加标点符号
        punctuations = "，。、；：？！「」『』【】（）《》〈〉“”‘’…—～·"
        for char in punctuations:
            char_code = hex(ord(char))[2:].upper().zfill(4)
            char_map[char_code] = char
        
        return char_map
    
    def check_database(self, char_code):
        """检查字符在数据库中的存在情况"""
        font_style = self.font_style.get()
        # 使用绝对路径构建数据库路径
        db_path = os.path.join(self.project_root, "data", f"calligraphy_{font_style}.db")
        
        # 打印路径用于调试
        print(f"检查数据库路径: {db_path}")
        
        if not os.path.exists(db_path):
            return False, f"数据库文件不存在: {db_path}"
        
        try:
            # 连接到数据库
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 查询字符是否存在 - 只使用char_code查询
            cursor.execute("""
            SELECT COUNT(*) 
            FROM standard_chars 
            WHERE char_code = ?
            """, (char_code,))
            
            count = cursor.fetchone()[0]
            conn.close()
            
            if count > 0:
                return True, f"字符 '{char_code}' 存在于数据库中"
            else:
                return False, f"字符 '{char_code}' 不在数据库中"
        except Exception as e:
            return False, f"数据库查询失败: {str(e)}"
    
    def create_widgets(self):
        """创建GUI组件"""
        # 顶部按钮
        top_frame = tk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 字体选择下拉框
        font_label = tk.Label(top_frame, text="选择字体:")
        font_label.pack(side=tk.LEFT, padx=5)
        
        font_combo = ttk.Combobox(top_frame, textvariable=self.font_style, 
                                 values=["light", "medium", "regular"], state="readonly")
        font_combo.pack(side=tk.LEFT, padx=5)
        
        # 功能按钮
        tk.Button(top_frame, text="加载作品", command=self.load_image).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="分析特征", command=self.analyze_features).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="识别文字", command=self.recognize_text).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="评价作品", command=self.evaluate).pack(side=tk.LEFT, padx=5)
        
        # 图像显示
        image_frame = tk.LabelFrame(self.root, text="作品预览")
        image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.image_label = tk.Label(image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 特征和结果
        result_frame = tk.Frame(self.root)
        result_frame.pack(fill=tk.BOTH, padx=10, pady=5)
        
        # 特征显示
        features_frame = tk.LabelFrame(result_frame, text="特征提取结果")
        features_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.features_text = tk.Text(features_frame, height=10)
        self.features_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 评分结果
        results_frame = tk.LabelFrame(result_frame, text="评价结果")
        results_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.results_text = tk.Text(results_frame, height=10)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_image(self):
        """加载图像"""
        file_path = filedialog.askopenfilename(
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not file_path:
            return
        
        try:
            self.current_image_path = file_path
            self.status_var.set(f"已加载: {os.path.basename(file_path)}")
            
            # 显示图像
            img = Image.open(file_path)
            img.thumbnail((500, 500))
            photo = ImageTk.PhotoImage(img)
            
            self.image_label.config(image=photo)
            self.image_label.image = photo
        except Exception as e:
            messagebox.showerror("错误", f"加载图像失败: {str(e)}")
            self.status_var.set(f"错误: {str(e)}")
    
    def analyze_features(self):
        """分析特征"""
        if not self.current_image_path:
            messagebox.showwarning("警告", "请先加载图像")
            return
        
        try:
            self.status_var.set("正在分析特征...")
            self.root.update()
            # 打印文件信息用于调试
            file_ext = os.path.splitext(self.current_image_path)[1].lower()
            print(f"分析图像: {self.current_image_path}, 格式: {file_ext}")
            # 处理图像
            result = self.processor.process_image(self.current_image_path)
            self.current_features = result["features"]
            
            # 保存预处理后的图像用于艺术评价
            self.preprocessed_image = result.get("preprocessed", None)
            
            # 保存原始灰度图像用于墨色分析
            self.original_gray = cv2.imread(self.current_image_path, cv2.IMREAD_GRAYSCALE)
            
            # 显示特征
            self.display_features()
            self.status_var.set("特征分析完成")
        except Exception as e:
            messagebox.showerror("错误", f"特征分析失败: {str(e)}")
            self.status_var.set(f"错误: {str(e)}")
    
    def display_features(self):
        """显示特征信息"""
        if not self.current_features:
            return
        
        text = "=== 笔画特征 ===\n"
        stroke = self.current_features["stroke"]
        text += f"平均笔画宽度: {stroke['stroke_width_mean']:.4f}\n"
        text += f"笔画宽度标准差: {stroke['stroke_width_std']:.4f}\n"
        text += f"平均曲率: {stroke['curvature_mean']:.4f}\n"
        text += f"曲率标准差: {stroke['curvature_std']:.4f}\n\n"
        
        text += "=== 结构特征 ===\n"
        for i, grid in enumerate(self.current_features["structure"]):
            text += f"网格 {i+1}: 密度={grid['density']:.4f}, 偏移={grid['center_offset']:.4f}\n"
        
        self.features_text.delete(1.0, tk.END)
        self.features_text.insert(tk.END, text)
    
    def recognize_text(self):
        """使用百度OCR识别文字，并通过动态映射表获取编码"""
        if not self.current_image_path:
            messagebox.showwarning("警告", "请先加载图像")
            return
        
        try:
            self.status_var.set("正在识别文字...")
            self.root.update()
            
            # 1. 获取百度OCR access token
            token = self.get_baidu_token()
            if not token:
                raise Exception("获取百度OCR token失败")
            
            # 2. 读取图像并编码为base64
            with open(self.current_image_path, "rb") as f:
                base64_data = base64.b64encode(f.read()).decode()
            
            # 3. 准备OCR请求参数
            headers = {'content-type': 'application/x-www-form-urlencoded'}
            params = {
                "access_token": token,
                "image": base64_data,
                "language_type": "CHN_ENG",  # 中英文混合
                "detect_direction": "true",   # 检测文字方向
                "recognize_granularity": "small"  # 精细识别模式
            }
            
            # 4. 发送OCR请求
            response = requests.post(self.ocr_config["ocr_url"], data=params, headers=headers)
            response.raise_for_status()  # 检查HTTP错误
            
            # 5. 解析OCR结果
            result = response.json()
            if "error_code" in result:
                error_msg = result.get("error_msg", "未知错误")
                raise Exception(f"OCR识别错误: {error_msg} (错误码: {result['error_code']})")
            
            if "words_result" not in result or not result["words_result"]:
                raise Exception("未识别到文字")
            
            # 6. 获取识别的第一个字符
            recognized_text = result["words_result"][0]["words"]
            if not recognized_text:
                raise Exception("识别结果为空")
            
            first_char = recognized_text[0]  # 取第一个字符
            self.status_var.set(f"识别结果: {first_char}")
            
            # 7. 动态生成字符映射表
            char_map = self.generate_char_map()
            
            # 8. 查找字符对应的编码 - 确保4位十六进制格式
            char_code = None
            
            # 方法1: 通过字符查找编码
            for code, char in char_map.items():
                if char == first_char:
                    char_code = code
                    break
            
            # 方法2: 如果找不到，使用Unicode编码
            if not char_code:
                try:
                    char_code = hex(ord(first_char))[2:].upper().zfill(4)
                    # 验证编码格式 (4位十六进制)
                    if len(char_code) != 4 or not all(c in "0123456789ABCDEF" for c in char_code):
                        raise ValueError("无效的Unicode编码")
                    
                    # 添加到映射表
                    char_map[char_code] = first_char
                except Exception as e:
                    # 方法3: 使用字符本身作为编码
                    char_code = first_char
                    messagebox.showinfo("提示", f"使用字符本身作为编码: {char_code}")
            
            # 9. 保存识别结果
            self.char_code = char_code
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"识别到的字符: {first_char}\n编码: {char_code}")
            self.status_var.set(f"识别完成: {first_char} (编码: {char_code})")
        
        except Exception as e:
            messagebox.showerror("识别错误", f"文字识别失败: {str(e)}")
            self.status_var.set(f"错误: {str(e)}")
    
    def get_baidu_token(self):
        """获取百度OCR access token"""
        params = {
            "grant_type": "client_credentials",
            "client_id": self.ocr_config["api_key"],
            "client_secret": self.ocr_config["secret_key"]
        }
        
        try:
            response = requests.get(self.ocr_config["token_url"], params=params)
            response.raise_for_status()
            token = response.json().get("access_token")
            return token
        except Exception as e:
            messagebox.showerror("OCR错误", f"获取token失败: {str(e)}")
            return None
    
    def evaluate(self):
        """评价作品"""
        if not self.current_features or not self.char_code:
            messagebox.showwarning("警告", "请先分析特征并识别文字")
            return
        
        try:
            self.status_var.set("正在评价作品...")
            self.root.update()
            
            # 动态加载当前字体的数据库
            font_style = self.font_style.get()
            # 使用绝对路径构建数据库路径
            db_path = os.path.join(self.project_root, "data", f"calligraphy_{font_style}.db")
            
            # 打印路径用于调试
            print(f"评价使用数据库路径: {db_path}")
            
            # 检查数据库是否存在
            if not os.path.exists(db_path):
                # 添加详细的错误提示和解决方案
                error_msg = (
                    f"未找到{font_style}字体的数据库！\n\n"
                    f"路径: {db_path}\n\n"
                    "请按以下步骤操作：\n"
                    "1. 运行 builddata.py 生成标准字体图片\n"
                    "2. 运行 modelapp.py --style {font_style} 构建数据库\n\n"
                    "完成后重启本程序。"
                )
                messagebox.showerror("数据库错误", error_msg)
                return
            
            # 检查字符是否在数据库中
            exists, message = self.check_database(self.char_code)
            if not exists:
                # 添加调试信息
                char_map = self.generate_char_map()
                char = char_map.get(self.char_code, "未知字符")
                
                # 提供详细解决方案
                solution = (
                    f"{message}\n\n"
                    f"字符: {char} (编码: {self.char_code})\n"
                    f"字体样式: {font_style}\n"
                    f"数据库路径: {db_path}\n\n"
                    "可能原因及解决方案：\n"
                    "1. 该字符不在标准字符集中\n"
                    "2. 数据库构建不完整\n"
                    "3. 字体文件不包含该字符\n\n"
                    "请尝试：\n"
                    "a) 检查数据库构建日志\n"
                    "b) 重新构建数据库\n"
                    "c) 使用不同字体"
                )
                messagebox.showerror("评价错误", solution)
                return
            
            # 创建评价器
            self.evaluator = CalligraphyEvaluator(db_path, font_style)
            
            # 评价作品
            evaluation = self.evaluator.evaluate(self.current_features, self.char_code)
            if not evaluation:
                raise Exception("评价失败，未找到标准特征")
            
            # 添加艺术评价（只在笔画和结构得分合格时）
            art_evaluation = {
                "art_score": 0.0, 
                "feedback": "笔画或结构得分过低，不进行艺术评价"
            }
            
            # 只有当笔画和结构得分都超过0.5时才进行艺术评价
            if evaluation["stroke_score"] > 0.5 and evaluation["structure_score"] > 0.5:
                # 获取预处理后的图像（在特征分析时已保存）
                if hasattr(self, 'preprocessed_image') and self.preprocessed_image is not None:
                    art_evaluator = ArtEvaluator()
                    art_evaluation = art_evaluator.evaluate_artistic_features(
                        self.preprocessed_image, 
                        self.original_gray  # 传递原始灰度图像用于墨色分析
                    )
                
                # 将艺术得分纳入总分（权重30%）
                evaluation["total_score"] = (
                    0.7 * evaluation["total_score"] + 
                    0.3 * art_evaluation["art_score"]
                )
            
            # 添加艺术评价结果
            evaluation["art"] = art_evaluation
            
            # 显示结果
            self.display_evaluation(evaluation)
            self.status_var.set("评价完成")
            
            # 保存结果到数据库
            self.save_to_database(evaluation)
        except Exception as e:
            messagebox.showerror("评价错误", f"评价失败: {str(e)}")
            self.status_var.set(f"错误: {str(e)}")
    
    def display_evaluation(self, evaluation):
        """显示评价结果"""
        text = f"=== 书法评价结果 ===\n"
        text += f"字体标准: {self.font_style.get()}\n"
        text += f"综合得分: {evaluation['total_score']:.2f}/1.00\n"
        text += f"笔画得分: {evaluation['stroke_score']:.2f}/1.00\n"
        text += f"结构得分: {evaluation['structure_score']:.2f}/1.00\n"
        
        # 显示艺术得分
        if "art" in evaluation:
            art = evaluation["art"]
            text += f"艺术得分: {art['art_score']:.2f}/1.00\n"
            text += f"艺术评价: {art['feedback']}\n"
            
            # 显示墨色梯度得分
            if "ink_gradient" in art:
                text += f"墨色梯度得分: {art['ink_gradient']:.2f}/1.00\n"
        
        text += "\n详细分析:\n"
        text += f"笔画宽度: 您的 {evaluation['details']['stroke']['width_mean'][0]:.4f} vs 标准 {evaluation['details']['stroke']['width_mean'][1]:.4f}\n"
        text += f"笔画均匀性: 您的 {evaluation['details']['stroke']['width_std'][0]:.4f} vs 标准 {evaluation['details']['stroke']['width_std'][1]:.4f}\n"
        text += f"笔画曲率: 您的 {evaluation['details']['stroke']['curvature_mean'][0]:.4f} vs 标准 {evaluation['details']['stroke']['curvature_mean'][1]:.4f}\n"
        
        # 添加结构分析
        text += "\n结构分析:\n"
        for i, grid in enumerate(evaluation['details']['structure']):
            text += f"网格 {i+1}: 密度(您:{grid['density'][0]:.3f} vs 标准:{grid['density'][1]:.3f}) "
            text += f"偏移(您:{grid['center_offset'][0]:.3f} vs 标准:{grid['center_offset'][1]:.3f})\n"
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, text)
    
    def save_to_database(self, evaluation):
        """保存结果到数据库"""
        if not self.current_image_path or not self.char_code:
            return
        
        try:
            # 创建临时数据库连接
            font_style = self.font_style.get()
            db_path = os.path.join(self.project_root, "data", f"calligraphy_{font_style}.db")
            temp_db = CalligraphyDB(db_path)
            
            # 插入用户提交记录
            temp_db.conn.execute("""
            INSERT INTO user_submissions (file_path, char_code, score, features)
            VALUES (?, ?, ?, ?)
            """, (
                self.current_image_path,
                self.char_code,
                evaluation["total_score"],
                json.dumps(self.current_features)
            ))
            temp_db.conn.commit()
            temp_db.close()
            self.status_var.set("结果已保存到数据库")
        except sqlite3.Error as e:
            messagebox.showerror("数据库错误", f"保存结果失败: {str(e)}")
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'db') and self.db:
            self.db.close()
        if hasattr(self, 'evaluator') and self.evaluator:
            self.evaluator.close()

if __name__ == "__main__":
    root = tk.Tk()
    app = CalligraphyApp(root)
    root.mainloop()