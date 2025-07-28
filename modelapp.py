import os
import json
from tqdm import tqdm
from core.database import CalligraphyDB
from do import ProcessingPipeline
import argparse
import time
import traceback

def build_database(font_style, base_dir="base", db_dir="data"):
    """构建特定字体的数据库"""
    # 创建数据库路径
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, f"calligraphy_{font_style}.db")
    
    # 初始化数据库和处理器
    db = CalligraphyDB(db_path)
    processor = ProcessingPipeline()
    
    # 加载全局字符映射
    char_map_path = os.path.join(base_dir, "char_map.json")
    if not os.path.exists(char_map_path):
        print(f"错误: 找不到字符映射文件 {char_map_path}")
        print("请先运行 builddata.py 生成标准字体图片")
        return
    
    with open(char_map_path, "r", encoding="utf-8") as f:
        total_map = json.load(f)
        global_map = total_map["global_map"]  # 获取全局映射
    
    # 获取特定字体的映射
    style_maps = total_map.get("style_maps", {})
    # 检查请求的字体样式是否存在
    if font_style not in style_maps:
        print(f"错误: 找不到字体样式 {font_style} 的映射")
        return
    
    # 创建新的字符列表 - 使用字符编码作为键
    char_list = {}
    for char, filename in style_maps[font_style].items():
        char_code = filename.split('.')[0]  # 从文件名提取编码
        char_list[char_code] = char  # 存储为 编码 -> 字符
    
    char_dir = os.path.join(base_dir, font_style)
    
    print(f"开始构建数据库: {font_style} 字体")
    print(f"共有 {len(char_list)} 个字符需要处理")
    
    # 断点续建：读取已经处理过的字符
    checkpoint_file = f"checkpoint_{font_style}.txt"
    processed_chars = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            processed_chars = set(f.read().splitlines())
        print(f"发现断点文件，已处理 {len(processed_chars)} 个字符")
    
    # 处理每个字符
    processed_count = 0
    start_time = time.time()
    error_log = []
    
    for char_code, char in tqdm(char_list.items(), desc=f"处理 {font_style} 字体"):
        # 如果已经处理过，则跳过
        if char_code in processed_chars:
            continue
            
        # 构建图片路径 - 使用编码作为文件名
        char_path = os.path.join(char_dir, f"{char_code}.png")
        
        # 检查字符图片是否存在
        if not os.path.exists(char_path):
            error_msg = f"字符图片不存在: {char_path}"
            print(error_msg)
            error_log.append(error_msg)
            continue
        
        try:
            # 处理图像并提取特征
            result = processor.process_image(char_path)
            
            # 插入数据库 - 使用char_code作为键
            db.insert_standard_char(char_code, char, font_style, result["features"])
            
            # 记录已处理
            processed_count += 1
            processed_chars.add(char_code)  # 使用char_code作为断点记录
            with open(checkpoint_file, "a", encoding="utf-8") as f:
                f.write(f"{char_code}\n")
            
            # 每处理100个字符显示一次进度
            if processed_count % 100 == 0:
                elapsed = time.time() - start_time
                rate = processed_count / elapsed if elapsed > 0 else 0
                remaining = (len(char_list) - len(processed_chars)) / rate if rate > 0 else float('inf')
                print(f"进度: {processed_count}/{len(char_list)} | "
                      f"速率: {rate:.2f} 字符/秒 | "
                      f"预计剩余时间: {remaining/60:.1f} 分钟")
                
        except Exception as e:
            error_msg = f"处理字符 {char} (编码: {char_code}) 失败: {str(e)}"
            error_log.append(error_msg)
            print(error_msg)
            traceback.print_exc()
    
    # 保存错误日志
    error_log_path = f"error_log_{font_style}.txt"
    with open(error_log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(error_log))
    
    # 打印构建统计信息
    total_chars = len(char_list)
    success_rate = (processed_count / total_chars) * 100 if total_chars > 0 else 0
    print(f"\n数据库构建完成! 成功处理: {processed_count}/{total_chars} 字符 ({success_rate:.2f}%)")
    
    if error_log:
        print(f"发现 {len(error_log)} 个错误，详见: {error_log_path}")
    
    # 完成后删除断点文件
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    
    total_time = time.time() - start_time
    print(f"数据库文件位置: {db_path}")
    print(f"总耗时: {total_time/60:.1f} 分钟 | 平均速率: {len(char_list)/total_time:.2f} 字符/秒")
    db.close()

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='构建标准字符数据库')
    parser.add_argument('--style', default="regular", choices=["light", "medium", "regular"],
                        help='要构建的字体样式 (默认: regular)')
    parser.add_argument('--all', action='store_true',
                        help='构建所有字体样式')
    args = parser.parse_args()
    
    # 创建数据目录
    os.makedirs("data", exist_ok=True)
    
    if args.all:
        styles = ["light", "medium", "regular"]
        print(f"将构建所有字体样式: {', '.join(styles)}")
        for style in styles:
            build_database(style)
    else:
        print(f"将构建字体样式: {args.style}")
        build_database(args.style)