#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全景街景图片车站检测脚本
使用滑动窗口方式处理全景图片，检测其中的车站目标
"""

import os
import cv2
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import time
from datetime import datetime

# ==================== 配置区域 ====================
# 在这里修改你的输入和输出路径
INPUT_PATH = "/Users/songdingan/Downloads/detect_bus/images"  # ← 请在这里填入你的全景图片文件夹路径
OUTPUT_ROOT = "/Users/songdingan/Downloads/detect_bus"           # ← 请在这里填入你的输出根目录路径

# 模型配置
MODEL_PATH = 'train/weights/best.pt'        # 模型文件路径
CONFIDENCE_THRESHOLD = 0.6                 # 置信度阈值

# 滑动窗口配置
WINDOW_SIZE = 640                          # 滑动窗口大小 640x640
STEP_SIZE = 512                           # 滑动步长 (512意味着128像素重叠)
NMS_THRESHOLD = 0.4                       # 非极大值抑制阈值，用于去除重复检测

# 类别名称映射配置
CLASS_NAME_MAPPING = {
    # 将模型检测的类别名称映射为指定的缩写
    # 格式: "模型类别名": "CSV列名"
    # 你可以根据实际的模型类别名称进行修改
    "顶棚": "DP",        # 顶棚 -> DP
    "栏杆": "LG",        # 栏杆 -> LG  
    "座椅": "ZY",        # 座椅 -> ZY
    "座椅栏杆": "FS",    # 座椅栏杆 -> FS
    "站牌": "ZP",        # 站牌 -> ZP
    "侧边防护": "CBFH",  # 侧边防护 -> CBFH
    
    # 如果有其他英文类别名，也可以添加映射
    "shelter": "DP",     # 如果模型用英文 shelter 表示顶棚
    "railing": "LG",     # 如果模型用英文 railing 表示栏杆
    "seat": "ZY",        # 如果模型用英文 seat 表示座椅
    "sign": "ZP",        # 如果模型用英文 sign 表示站牌
    "barrier": "CBFH",   # 如果模型用英文 barrier 表示侧边防护
}
# ==================================================

class SlidingWindowDetector:
    def __init__(self, model_path=MODEL_PATH, conf_threshold=CONFIDENCE_THRESHOLD):
        """
        初始化滑动窗口检测器
        
        Args:
            model_path: 模型文件路径
            conf_threshold: 置信度阈值
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None
        self.class_names = []
        self.mapped_class_names = []
        self.load_model()
        
    def load_model(self):
        """加载YOLO模型"""
        try:
            print(f"正在加载模型: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # 获取所有类别名称
            if hasattr(self.model, 'names') and self.model.names:
                self.class_names = list(self.model.names.values())
            else:
                print("警告: 无法获取模型类别名称")
                self.class_names = []
            
            # 生成映射后的类别名称
            self.mapped_class_names = self.get_mapped_class_names()
            
            print(f"模型加载成功! 原始检测类别: {self.class_names}")
            print(f"映射后CSV列名: {self.mapped_class_names}")
            print(f"滑动窗口配置: {WINDOW_SIZE}x{WINDOW_SIZE}, 步长: {STEP_SIZE}")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    def sliding_window_detect(self, image_path):
        """
        使用滑动窗口检测全景图片
        
        Args:
            image_path: 图片路径
            
        Returns:
            tuple: (统计信息, 带检测框的图片, 所有检测结果)
        """
        print(f"正在使用滑动窗口检测: {Path(image_path).name}")
        
        # 读取原图
        original_img = cv2.imread(image_path)
        if original_img is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        h, w = original_img.shape[:2]
        print(f"  原图尺寸: {w}x{h}")
        
        # 计算滑动窗口数量
        windows_x = (w - WINDOW_SIZE) // STEP_SIZE + 1
        windows_y = (h - WINDOW_SIZE) // STEP_SIZE + 1
        total_windows = windows_x * windows_y
        
        print(f"  滑动窗口数量: {windows_x}x{windows_y} = {total_windows}")
        
        all_detections = []
        window_count = 0
        
        # 滑动窗口检测
        for y in range(0, h - WINDOW_SIZE + 1, STEP_SIZE):
            for x in range(0, w - WINDOW_SIZE + 1, STEP_SIZE):
                window_count += 1
                
                # 提取窗口
                window = original_img[y:y+WINDOW_SIZE, x:x+WINDOW_SIZE]
                
                # 对窗口进行检测
                results = self.model(window, conf=self.conf_threshold, imgsz=WINDOW_SIZE)
                result = results[0]
                
                # 处理检测结果
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        # 获取窗口内的坐标
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # 转换到原图坐标
                        global_x1 = x + x1
                        global_y1 = y + y1
                        global_x2 = x + x2
                        global_y2 = y + y2
                        
                        # 获取类别和置信度
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = self.model.names[cls_id] if cls_id in self.model.names else str(cls_id)
                        
                        all_detections.append({
                            'x1': global_x1,
                            'y1': global_y1,
                            'x2': global_x2,
                            'y2': global_y2,
                            'conf': conf,
                            'class_id': cls_id,
                            'class_name': class_name
                        })
                
                # 显示进度
                if window_count % 10 == 0 or window_count == total_windows:
                    print(f"    进度: {window_count}/{total_windows} 窗口")
        
        print(f"  原始检测结果: {len(all_detections)} 个")
        
        # 应用非极大值抑制去除重复检测
        filtered_detections = self.apply_nms(all_detections)
        print(f"  NMS后检测结果: {len(filtered_detections)} 个")
        
        # 在原图上绘制检测框
        result_img = self.draw_detections(original_img.copy(), filtered_detections)
        
        # 生成统计信息
        stats = self.get_detection_stats_from_detections(filtered_detections)
        
        return stats, result_img, filtered_detections
    
    def apply_nms(self, detections):
        """
        应用非极大值抑制去除重复检测
        
        Args:
            detections: 检测结果列表
            
        Returns:
            list: 过滤后的检测结果
        """
        if not detections:
            return []
        
        # 按类别分组应用NMS
        filtered_detections = []
        
        # 获取所有唯一的类别
        unique_classes = list(set([det['class_id'] for det in detections]))
        
        for class_id in unique_classes:
            # 筛选当前类别的检测结果
            class_detections = [det for det in detections if det['class_id'] == class_id]
            
            if not class_detections:
                continue
            
            # 准备NMS输入
            boxes = np.array([[det['x1'], det['y1'], det['x2'], det['y2']] for det in class_detections])
            scores = np.array([det['conf'] for det in class_detections])
            
            # 应用OpenCV的NMS
            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(), 
                scores.tolist(), 
                self.conf_threshold, 
                NMS_THRESHOLD
            )
            
            # 收集保留的检测结果
            if len(indices) > 0:
                for i in indices.flatten():
                    filtered_detections.append(class_detections[i])
        
        return filtered_detections
    
    def draw_detections(self, image, detections):
        """
        在图片上绘制检测框
        
        Args:
            image: 原图
            detections: 检测结果列表
            
        Returns:
            numpy.ndarray: 带检测框的图片
        """
        # 定义颜色 (BGR格式)
        colors = {
            0: (0, 255, 0),    # 绿色
            1: (255, 0, 0),    # 蓝色
            2: (0, 0, 255),    # 红色
            3: (255, 255, 0),  # 青色
            4: (255, 0, 255),  # 紫色
            5: (0, 255, 255),  # 黄色
        }
        
        for det in detections:
            x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
            conf = det['conf']
            class_name = det['class_name']
            class_id = det['class_id']
            
            # 选择颜色
            color = colors.get(class_id % len(colors), (0, 255, 0))
            
            # 绘制检测框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            
            # 准备标签文本
            label = f"{class_name}: {conf:.2f}"
            
            # 计算文本尺寸
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # 绘制文本背景
            cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            
            # 绘制文本
            cv2.putText(image, label, (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness)
        
        return image
    
    def get_detection_stats_from_detections(self, detections):
        """
        从检测结果生成统计信息
        
        Args:
            detections: 检测结果列表
            
        Returns:
            dict: 统计信息
        """
        stats = {
            'total_detections': len(detections),
            'class_counts': {},
            'confidences': []
        }
        
        for det in detections:
            class_name = det['class_name']
            conf = det['conf']
            
            stats['class_counts'][class_name] = stats['class_counts'].get(class_name, 0) + 1
            stats['confidences'].append(conf)
        
        return stats
    
    def get_mapped_class_names(self):
        """
        获取映射后的类别名称列表
        
        Returns:
            list: 映射后的类别名称
        """
        mapped_names = []
        for class_name in self.class_names:
            # 查找映射，如果没有找到则使用原名称
            mapped_name = CLASS_NAME_MAPPING.get(class_name, class_name)
            mapped_names.append(mapped_name)
        
        return mapped_names
    
    def get_mapped_class_name(self, original_name):
        """
        获取单个类别的映射名称
        
        Args:
            original_name: 原始类别名
            
        Returns:
            str: 映射后的类别名
        """
        return CLASS_NAME_MAPPING.get(original_name, original_name)
    
    def parse_filename_info(self, filename, csv_row):
        """
        解析文件名信息，按照 _ 分割成四列
        
        Args:
            filename: 图片文件名 (例如: 2764_121.296209_31.063385_2210.jpg)
            csv_row: CSV行数据字典
        """
        try:
            # 去掉文件扩展名
            base_name = filename.rsplit('.', 1)[0]
            
            # 按 _ 分割
            parts = base_name.split('_')
            
            if len(parts) >= 4:
                csv_row['image_id'] = parts[0]        # 图片ID/编号
                csv_row['longitude'] = parts[1]       # 经度
                csv_row['latitude'] = parts[2]        # 纬度  
                csv_row['time_code'] = parts[3]       # 时间代码/其他信息
            else:
                # 如果分割后少于4部分，填充空值
                csv_row['image_id'] = parts[0] if len(parts) > 0 else ''
                csv_row['longitude'] = parts[1] if len(parts) > 1 else ''
                csv_row['latitude'] = parts[2] if len(parts) > 2 else ''
                csv_row['time_code'] = parts[3] if len(parts) > 3 else ''
                print(f"    警告: 文件名格式不标准: {filename}")
                
        except Exception as e:
            # 解析失败时填充空值
            csv_row['image_id'] = ''
            csv_row['longitude'] = ''
            csv_row['latitude'] = ''
            csv_row['time_code'] = ''
            print(f"    警告: 文件名解析失败: {filename}, 错误: {e}")
    
    def append_to_csv(self, csv_row, csv_file, is_initialized):
        """
        将单行数据追加到CSV文件
        
        Args:
            csv_row: 要写入的行数据字典
            csv_file: CSV文件路径
            is_initialized: 是否已经初始化（写入表头）
        """
        try:
            # 确保列的顺序：filename, 文件名信息列, 然后是检测类别列
            filename_cols = ['filename', 'image_id', 'longitude', 'latitude', 'time_code']
            detection_cols = sorted([col for col in csv_row.keys() if col not in filename_cols])
            columns = filename_cols + detection_cols
            
            if not is_initialized:
                # 第一次写入，创建文件并写入表头
                df = pd.DataFrame([csv_row], columns=columns)
                df.to_csv(csv_file, index=False, encoding='utf-8-sig', mode='w')
            else:
                # 追加模式，只写入数据行
                df = pd.DataFrame([csv_row], columns=columns)
                df.to_csv(csv_file, index=False, encoding='utf-8-sig', mode='a', header=False)
                
        except Exception as e:
            print(f"    警告: 写入CSV文件失败: {e}")
    
    def process_batch_images(self, input_dir, output_root, image_extensions=('.jpg', '.jpeg', '.png', '.bmp')):
        """
        批量处理全景图片并按需求输出结果
        
        Args:
            input_dir: 输入图片目录
            output_root: 输出根目录
            image_extensions: 支持的图片格式
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"输入目录不存在: {input_dir}")
            return
        
        output_path = Path(output_root)
        
        # 创建输出目录结构
        detected_dir = output_path / "detected"           # 有检测结果的图片(带检测框)
        no_detection_dir = output_path / "no_detection"   # 无检测结果的图片
        
        detected_dir.mkdir(parents=True, exist_ok=True)
        no_detection_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取所有图片文件
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"在 {input_dir} 中未找到图片文件")
            return
        
        print(f"找到 {len(image_files)} 张图片，开始滑动窗口批量检测...")
        print(f"输出目录: {output_path}")
        print(f"原始检测类别: {self.class_names}")
        print(f"CSV列名映射: {dict(zip(self.class_names, self.mapped_class_names))}")
        
        # 准备CSV文件
        csv_file = output_path / 'detection_results.csv'
        csv_initialized = False
        
        # 统计信息
        total_stats = {
            'total_images': len(image_files),
            'images_with_detections': 0,
            'images_without_detections': 0,
            'total_detections': 0,
            'class_counts': {},
            'processing_time': 0
        }
        
        start_time = time.time()
        
        for i, image_file in enumerate(image_files, 1):
            print(f"\n进度: {i}/{len(image_files)} - {image_file.name}")
            
            try:
                # 使用滑动窗口检测图片
                stats, result_img, detections = self.sliding_window_detect(str(image_file))
                
                # 准备CSV行数据
                csv_row = {'filename': image_file.name}
                
                # 解析文件名信息（按 _ 分割）
                self.parse_filename_info(image_file.name, csv_row)
                
                # 初始化所有映射后的类别为0
                for mapped_name in self.mapped_class_names:
                    csv_row[mapped_name] = 0
                
                # 标记检测到的类别为1（使用映射后的名称）
                for original_class_name in stats['class_counts'].keys():
                    mapped_name = self.get_mapped_class_name(original_class_name)
                    if mapped_name in csv_row:
                        csv_row[mapped_name] = 1
                
                # 立即写入CSV文件
                self.append_to_csv(csv_row, csv_file, csv_initialized)
                if not csv_initialized:
                    csv_initialized = True
                    print(f"  → CSV文件已创建: {csv_file}")
                print(f"  → 检测结果已写入CSV")
                
                # 根据是否有检测结果保存图片到不同文件夹
                if stats['total_detections'] > 0:
                    # 有检测结果 - 保存带检测框的图片
                    dest_path = detected_dir / image_file.name
                    cv2.imwrite(str(dest_path), result_img)
                    
                    total_stats['images_with_detections'] += 1
                    total_stats['total_detections'] += stats['total_detections']
                    
                    # 累计各类别数量（使用映射后的名称）
                    for original_class_name, count in stats['class_counts'].items():
                        mapped_name = self.get_mapped_class_name(original_class_name)
                        total_stats['class_counts'][mapped_name] = total_stats['class_counts'].get(mapped_name, 0) + count
                    
                    print(f"  → 检测到 {stats['total_detections']} 个目标，已保存带检测框的图片到 detected/")
                    if stats['class_counts']:
                        for original_class_name, count in stats['class_counts'].items():
                            mapped_name = self.get_mapped_class_name(original_class_name)
                            print(f"    {mapped_name}({original_class_name}): {count} 个")
                else:
                    # 无检测结果 - 复制原图
                    dest_path = no_detection_dir / image_file.name
                    shutil.copy2(image_file, dest_path)
                    
                    total_stats['images_without_detections'] += 1
                    print(f"  → 未检测到目标，已复制原图到 no_detection/")
                
            except Exception as e:
                print(f"处理图片 {image_file.name} 时出错: {e}")
                # 出错的图片也记录到CSV中，所有类别都为0
                csv_row = {'filename': image_file.name}
                
                # 解析文件名信息（按 _ 分割）
                self.parse_filename_info(image_file.name, csv_row)
                
                for mapped_name in self.mapped_class_names:
                    csv_row[mapped_name] = 0
                
                # 立即写入CSV文件
                self.append_to_csv(csv_row, csv_file, csv_initialized)
                if not csv_initialized:
                    csv_initialized = True
                    print(f"  → CSV文件已创建: {csv_file}")
                print(f"  → 错误结果已写入CSV")
                continue
        
        total_stats['processing_time'] = time.time() - start_time
        
        # 显示总体统计
        self.print_batch_stats(total_stats, output_path, csv_file)
    
    def print_batch_stats(self, total_stats, output_path, csv_file):
        """打印批量处理统计信息"""
        print("\n" + "="*60)
        print("滑动窗口全景图片批量检测完成 - 总体统计")
        print("="*60)
        print(f"总图片数: {total_stats['total_images']}")
        print(f"有检测结果的图片数: {total_stats['images_with_detections']}")
        print(f"无检测结果的图片数: {total_stats['images_without_detections']}")
        print(f"检测成功率: {total_stats['images_with_detections']/total_stats['total_images']*100:.1f}%")
        print(f"总检测目标数: {total_stats['total_detections']}")
        print(f"处理时间: {total_stats['processing_time']:.2f} 秒")
        print(f"平均每张图片处理时间: {total_stats['processing_time']/total_stats['total_images']:.2f} 秒")
        
        if total_stats['class_counts']:
            print("\n各类别总统计 (CSV列名):")
            for class_name, count in total_stats['class_counts'].items():
                print(f"  {class_name}: {count} 个")
        
        print(f"\n输出文件:")
        print(f"  有检测结果的图片 (带检测框): {output_path}/detected/")
        print(f"  无检测结果的图片 (原图): {output_path}/no_detection/")
        print(f"  检测结果CSV文件: {csv_file}")
        
        # 显示CSV文件统计
        try:
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                print(f"  CSV文件包含 {len(df)} 行数据，{len(df.columns)} 列")
                print("  CSV文件格式:")
                print("    - filename: 图片文件名")
                print("    - image_id: 图片ID/编号")
                print("    - longitude: 经度坐标")
                print("    - latitude: 纬度坐标")
                print("    - time_code: 时间代码/其他信息")
                print("    检测类别列 (映射后名称):")
                print("      DP=顶棚, LG=栏杆, ZY=座椅, FS=座椅栏杆, ZP=站牌, CBFH=侧边防护")
                
                # 显示检测类别统计
                filename_info_cols = ['filename', 'image_id', 'longitude', 'latitude', 'time_code']
                for col in df.columns:
                    if col not in filename_info_cols:
                        detected_count = df[col].sum()
                        # 查找对应的原始类别名称
                        original_name = None
                        for orig, mapped in CLASS_NAME_MAPPING.items():
                            if mapped == col:
                                original_name = orig
                                break
                        
                        if original_name:
                            print(f"    - {col}({original_name}): {detected_count} 张图片检测到此类别")
                        else:
                            print(f"    - {col}: {detected_count} 张图片检测到此类别")
        except Exception as e:
            print(f"  CSV文件统计读取失败: {e}")

def main():
    """主函数"""
    
    # 检查配置
    if INPUT_PATH == "your_panorama_images_folder":
        print("❌ 错误: 请先在脚本顶部配置区域设置 INPUT_PATH")
        print("请修改脚本中的这一行:")
        print('INPUT_PATH = "your_panorama_images_folder"  # ← 请在这里填入你的全景图片文件夹路径')
        return
    
    if not os.path.exists(INPUT_PATH):
        print(f"❌ 错误: 输入路径不存在: {INPUT_PATH}")
        print("请检查 INPUT_PATH 设置是否正确")
        return
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 模型文件不存在: {MODEL_PATH}")
        print("请检查 MODEL_PATH 设置是否正确")
        return
    
    print("🚀 滑动窗口全景街景图片车站检测")
    print("="*50)
    print(f"输入路径: {INPUT_PATH}")
    print(f"输出路径: {OUTPUT_ROOT}")
    print(f"模型路径: {MODEL_PATH}")
    print(f"置信度阈值: {CONFIDENCE_THRESHOLD}")
    print(f"滑动窗口配置: {WINDOW_SIZE}x{WINDOW_SIZE}, 步长: {STEP_SIZE}")
    print(f"NMS阈值: {NMS_THRESHOLD}")
    print("="*50)
    
    # 创建检测器
    detector = SlidingWindowDetector()
    
    # 开始批量处理
    detector.process_batch_images(INPUT_PATH, OUTPUT_ROOT)
    
    print("\n✅ 处理完成!")

if __name__ == "__main__":
    main()
