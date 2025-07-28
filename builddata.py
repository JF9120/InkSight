import os
import json
from PIL import Image, ImageFont, ImageDraw
import argparse
import traceback

class FontImageGenerator:
    def __init__(self, font_dir="fonts", output_dir="base"):
        self.font_dir = font_dir
        self.output_dir = output_dir
        self.font_files = {
            "light": "LXGWWenKaiMono-Light.ttf",
            "medium": "LXGWWenKaiMono-Medium.ttf",
            "regular": "LXGWWenKaiMono-Regular.ttf"
        }
        self.common_chars = self.load_common_chars()
        self.char_map = {}
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        for style in self.font_files.keys():
            os.makedirs(os.path.join(output_dir, style), exist_ok=True)
    
    def load_common_chars(self):
        """加载常用汉字集（GB2312字符集，共6763个汉字）"""
        chars = []
        # GB2312 字符集范围: 0xB0A1-0xF7FE
        for code in range(0xB0A1, 0xF7FE + 1):
            try:
                # 将GB2312编码转换为Unicode字符
                byte1 = (code >> 8) & 0xFF
                byte2 = code & 0xFF
                char = bytes([byte1, byte2]).decode('gb2312')
                chars.append(char)
            except UnicodeDecodeError:
                continue
        return chars
    
    def generate_font_images(self):
        """生成所有字体样式的字符图片"""
        # 创建一个包含所有信息的字典
        total_map = {
            "global_map": {},  # 全局字符映射（编码->字符）
            "style_maps": {}   # 按字体分类的映射
        }
        
        for style, font_file in self.font_files.items():
            font_path = os.path.join(self.font_dir, font_file)
            if not os.path.exists(font_path):
                print(f"警告: 找不到字体文件 {font_path}, 跳过")
                continue
            
            print(f"正在生成 '{style}' 字体图片...")
            # 生成当前字体的字符映射
            style_map = self.generate_style_images(font_path, style)
            if style_map:  # 检查是否成功生成了映射
                total_map["style_maps"][style] = style_map
        
        # 添加全局字符映射
        total_map["global_map"] = self.char_map
        
        # 保存字符映射关系
        self.save_char_map(total_map)
        print(f"所有字体图片已生成到: {self.output_dir}")
    
    def generate_style_images(self, font_path, style, size=128):
        """生成特定字体的字符图片"""
        char_list = {}
        try:
            # 检查字体文件是否存在
            if not os.path.exists(font_path):
                print(f"错误: 找不到字体文件 {font_path}")
                print("请将字体文件放在 'fonts' 目录下")
                return char_list
            
            # 加载字体
            font = ImageFont.truetype(font_path, int(size * 0.7))
            print(f"成功加载字体: {font_path}")
        except Exception as e:
            print(f"字体加载失败: {str(e)}")
            traceback.print_exc()
            return char_list
        
        for char in self.common_chars:
            try:
                # 获取字符编码 - 确保4位大写十六进制格式
                char_code = hex(ord(char))[2:].upper().zfill(4)
                
                # 创建图片
                img = Image.new("L", (size, size), 255)
                draw = ImageDraw.Draw(img)
                
                # 绘制字符（调整位置使其居中）
                bbox = draw.textbbox((0, 0), char, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                x = (size - text_width) / 2 - bbox[0]
                y = (size - text_height) / 2 - bbox[1]
                
                draw.text((x, y), char, fill=0, font=font)
                
                # 保存图片
                filename = f"{char_code}.png"
                img.save(os.path.join(self.output_dir, style, filename))
                
                # 记录字符映射
                char_list[char] = filename
                if char_code not in self.char_map:
                    self.char_map[char_code] = char
            except Exception as e:
                print(f"生成字符 '{char}' 图片失败: {str(e)}")
                traceback.print_exc()
        
        return char_list
    
    def save_char_map(self, total_map):
        """保存编码与汉字的映射关系"""
        map_path = os.path.join(self.output_dir, "char_map.json")
        
        # 保存为JSON文件
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(total_map, f, ensure_ascii=False, indent=2)
        
        print(f"字符映射关系已保存到: {map_path}")


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='生成标准字体图片')
    parser.add_argument('--font-dir', default="fonts", help='字体文件目录')
    parser.add_argument('--output-dir', default="base", help='输出目录')
    args = parser.parse_args()
    
    print("=" * 50)
    print("硬笔书法评价系统 - 标准字体图片生成器")
    print("=" * 50)
    
    # 创建生成器并执行
    generator = FontImageGenerator(
        font_dir=args.font_dir,
        output_dir=args.output_dir
    )
    generator.generate_font_images()
    
    print("=" * 50)
    print("生成完成!")
    print("=" * 50)