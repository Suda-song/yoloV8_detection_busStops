import cv2
from ultralytics import YOLO
import os
import json
import csv
from datetime import datetime
import numpy as np
import shutil

class BusStopDetector:
    def __init__(self, model_path='bus_stop_detection/v8s_exp14/weights/best.pt'):
        """初始化公交站检测器"""
        self.model = YOLO(model_path)
        print(f"模型加载成功: {model_path}")
        
    def detect_image(self, image_path, conf_threshold=0.5, output_dir=None, 
                    enable_sliding_window=False, window_size=640, overlap=0.3):
        """
        检测图片中的公交站
        Args:
            image_path: 图片路径
            conf_threshold: 置信度阈值 (默认0.5)
            output_dir: 输出目录
            enable_sliding_window: 是否启用滑动窗口检测（适用于大图片）
            window_size: 滑动窗口大小
            overlap: 窗口重叠比例
        Returns:
            检测结果字典
        """
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图片: {image_path}")
            return None
        
        # 获取图片尺寸
        height, width = image.shape[:2]
        print(f"图片尺寸: {width} × {height}")
        
        # 判断是否需要使用滑动窗口
        if enable_sliding_window and (width > 1200 or height > 1200):
            print("检测到大尺寸图片，启用滑动窗口检测...")
            detections = self.detect_with_sliding_window(
                image, conf_threshold, window_size, overlap
            )
        else:
            # 直接检测
            detections = self.detect_single_image(image, conf_threshold)
        
        # 在图片上绘制检测结果并保存
        annotated_image = self.draw_results(image, detections)
        
        # 确保输出目录存在
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"detected_{os.path.basename(image_path)}")
        else:
            output_path = f"detected_{os.path.basename(image_path)}"
            
        cv2.imwrite(output_path, annotated_image)
        print(f"检测结果已保存到: {output_path}")
        
        return {
            'image_path': image_path,
            'image_size': [width, height],
            'detections': detections,
            'detection_count': len(detections),
            'output_image_path': output_path,
            'annotated_image': annotated_image
        }
    
    def detect_single_image(self, image, conf_threshold):
        """单张图片检测"""
        results = self.model(image, conf=conf_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence)
                    })
        
        return detections
    
    def detect_with_sliding_window(self, image, conf_threshold, window_size=640, overlap=0.3):
        """
        使用滑动窗口检测大图片
        Args:
            image: 输入图片
            conf_threshold: 置信度阈值
            window_size: 窗口大小
            overlap: 重叠比例
        Returns:
            合并后的检测结果
        """
        height, width = image.shape[:2]
        stride = int(window_size * (1 - overlap))
        
        all_detections = []
        
        # 计算窗口数量
        num_windows_h = max(1, (height - window_size) // stride + 1)
        num_windows_w = max(1, (width - window_size) // stride + 1)
        
        print(f"使用滑动窗口检测: {num_windows_w} × {num_windows_h} 个窗口")
        
        for i in range(num_windows_h):
            for j in range(num_windows_w):
                # 计算窗口位置
                y1 = min(i * stride, height - window_size)
                x1 = min(j * stride, width - window_size)
                y2 = y1 + window_size
                x2 = x1 + window_size
                
                # 提取窗口区域
                window = image[y1:y2, x1:x2]
                
                # 检测窗口
                results = self.model(window, conf=conf_threshold)
                
                # 处理检测结果并调整坐标
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            wx1, wy1, wx2, wy2 = box.xyxy[0].cpu().numpy()
                            confidence = box.conf[0].cpu().numpy()
                            
                            # 调整坐标到原图
                            abs_x1 = int(x1 + wx1)
                            abs_y1 = int(y1 + wy1)
                            abs_x2 = int(x1 + wx2)
                            abs_y2 = int(y1 + wy2)
                            
                            all_detections.append({
                                'bbox': [abs_x1, abs_y1, abs_x2, abs_y2],
                                'confidence': float(confidence),
                                'window': [i, j]
                            })
        
        # 合并重叠的检测框（非极大值抑制）
        merged_detections = self.merge_overlapping_detections(all_detections)
        
        print(f"滑动窗口检测完成，原始检测: {len(all_detections)}, 合并后: {len(merged_detections)}")
        
        return merged_detections
    
    def merge_overlapping_detections(self, detections, iou_threshold=0.5):
        """
        合并重叠的检测框
        Args:
            detections: 检测结果列表
            iou_threshold: IoU阈值
        Returns:
            合并后的检测结果
        """
        if not detections:
            return []
        
        # 按置信度排序
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        merged = []
        used = [False] * len(detections)
        
        for i in range(len(detections)):
            if used[i]:
                continue
                
            current = detections[i]
            merged.append(current)
            used[i] = True
            
            # 检查与其他检测框的重叠
            for j in range(i + 1, len(detections)):
                if used[j]:
                    continue
                    
                if self.calculate_iou(current['bbox'], detections[j]['bbox']) > iou_threshold:
                    used[j] = True
        
        return merged
    
    def calculate_iou(self, bbox1, bbox2):
        """计算两个边界框的IoU"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 计算交集
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # 计算并集
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def convert_to_yolo_format(self, bbox, image_width, image_height):
        """
        将边界框坐标转换为YOLO格式
        Args:
            bbox: [x1, y1, x2, y2] 格式的边界框
            image_width: 图片宽度
            image_height: 图片高度
        Returns:
            [x_center, y_center, width, height] 格式的YOLO坐标
        """
        x1, y1, x2, y2 = bbox
        
        # 计算中心点和宽高
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
        
        # 归一化到0-1范围
        x_center /= image_width
        y_center /= image_height
        width /= image_width
        height /= image_height
        
        return [x_center, y_center, width, height]
    
    def detect_folder_batch(self, input_folder, output_folder, conf_threshold=0.5, 
                           enable_sliding_window=False, generate_yolo_data=True):
        """
        批量检测文件夹中的图片
        Args:
            input_folder: 输入文件夹路径
            output_folder: 输出文件夹路径
            conf_threshold: 置信度阈值
            enable_sliding_window: 是否启用滑动窗口
            generate_yolo_data: 是否生成YOLO格式训练数据
        """
        # 支持的图片格式
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        # 获取所有图片文件
        image_files = []
        for file in os.listdir(input_folder):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(input_folder, file))
        
        print(f"找到 {len(image_files)} 张图片")
        
        if not image_files:
            print("未找到图片文件")
            return
        
        # 创建输出文件夹结构
        self.create_output_folders(output_folder, generate_yolo_data)
        
        # 批量检测
        all_results = []
        detected_images = []
        not_detected_images = []
        
        for i, image_path in enumerate(image_files):
            print(f"\n处理第 {i+1}/{len(image_files)} 张图片: {os.path.basename(image_path)}")
            
            # 检测图片
            result = self.detect_image(
                image_path, 
                conf_threshold=conf_threshold,
                output_dir=os.path.join(output_folder, 'annotated_images'),
                enable_sliding_window=enable_sliding_window
            )
            
            if result:
                all_results.append(result)
                
                # 分类图片
                if result['detection_count'] > 0:
                    detected_images.append(result)
                    print(f"检测到 {result['detection_count']} 个公交站")
                else:
                    not_detected_images.append(result)
                    print("未检测到公交站")
                
                # 保存单张图片结果
                self.save_single_result(result, output_folder, i)
        
        # 生成YOLO格式训练数据
        if generate_yolo_data:
            self.generate_yolo_training_data(all_results, output_folder)
        
        # 分类保存图片
        self.classify_and_save_images(all_results, output_folder)
        
        # 生成CSV文件
        self.generate_csv_files(all_results, detected_images, not_detected_images, output_folder)
        
        # 保存汇总结果
        self.save_summary_results(all_results, output_folder)
        
        print(f"\n=== 检测完成 ===")
        print(f"总图片数: {len(all_results)}")
        print(f"检测到公交站的图片: {len(detected_images)}")
        print(f"未检测到公交站的图片: {len(not_detected_images)}")
        print(f"结果保存在: {output_folder}")
    
    def create_output_folders(self, output_folder, generate_yolo_data=True):
        """创建输出文件夹结构"""
        folders = [
            'detected_images',      # 检测到公交站的图片
            'not_detected_images',  # 未检测到公交站的图片
            'annotated_images',     # 带标注的图片
            'json_results',         # JSON结果文件
            'csv_files',           # CSV文件
        ]
        
        if generate_yolo_data:
            folders.extend([
                'yolo_dataset',     # YOLO格式数据集
                'yolo_dataset/images',  # YOLO图片
                'yolo_dataset/labels',  # YOLO标签
            ])
        
        for folder in folders:
            os.makedirs(os.path.join(output_folder, folder), exist_ok=True)
    
    def generate_yolo_training_data(self, all_results, output_folder):
        """生成YOLO格式的训练数据"""
        yolo_images_dir = os.path.join(output_folder, 'yolo_dataset', 'images')
        yolo_labels_dir = os.path.join(output_folder, 'yolo_dataset', 'labels')
        
        print("\n生成YOLO格式训练数据...")
        
        for i, result in enumerate(all_results):
            image_path = result['image_path']
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # 复制图片到YOLO数据集
            yolo_image_path = os.path.join(yolo_images_dir, f"{image_name}.jpg")
            shutil.copy2(image_path, yolo_image_path)
            
            # 生成标签文件
            label_path = os.path.join(yolo_labels_dir, f"{image_name}.txt")
            
            with open(label_path, 'w') as f:
                for detection in result['detections']:
                    # 转换为YOLO格式 (class_id x_center y_center width height)
                    bbox = detection['bbox']
                    yolo_bbox = self.convert_to_yolo_format(
                        bbox, result['image_size'][0], result['image_size'][1]
                    )
                    
                    # 公交站类别ID为0
                    line = f"0 {' '.join([str(x) for x in yolo_bbox])}\n"
                    f.write(line)
        
        # 生成YOLO配置文件
        self.generate_yolo_config(output_folder, len(all_results))
        
        print(f"YOLO数据集已生成: {yolo_images_dir}")
    
    def generate_yolo_config(self, output_folder, data_count):
        """生成YOLO配置文件"""
        config_content = f"""# YOLO数据集配置文件
# 数据集路径
path: {output_folder}/yolo_dataset
train: images
val: images

# 类别信息
nc: 1  # 类别数量
names: ['bus_stop']  # 类别名称

# 数据集统计
total_images: {data_count}
"""
        
        config_path = os.path.join(output_folder, 'yolo_dataset', 'dataset.yaml')
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        print(f"YOLO配置文件已生成: {config_path}")
    
    def classify_and_save_images(self, all_results, output_folder):
        """分类保存图片"""
        detected_dir = os.path.join(output_folder, 'detected_images')
        not_detected_dir = os.path.join(output_folder, 'not_detected_images')
        
        for result in all_results:
            image_path = result['image_path']
            image_name = os.path.basename(image_path)
            
            if result['detection_count'] > 0:
                # 检测到公交站的图片
                dest_path = os.path.join(detected_dir, image_name)
            else:
                # 未检测到公交站的图片
                dest_path = os.path.join(not_detected_dir, image_name)
            
            shutil.copy2(image_path, dest_path)
    
    def generate_csv_files(self, all_images, detected_images, not_detected_images, output_folder):
        """生成CSV文件"""
        csv_dir = os.path.join(output_folder, 'csv_files')
        
        # 1. 全部检测的图片文件名
        all_images_csv = os.path.join(csv_dir, 'all_images.csv')
        with open(all_images_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['文件名', '图片路径', '检测数量', '图片尺寸'])
            for result in all_images:
                writer.writerow([
                    os.path.basename(result['image_path']),
                    result['image_path'],
                    result['detection_count'],
                    f"{result['image_size'][0]}x{result['image_size'][1]}"
                ])
        
        # 2. 检测到公交站的图片文件名
        detected_csv = os.path.join(csv_dir, 'detected_images.csv')
        with open(detected_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['文件名', '图片路径', '检测数量', '置信度'])
            for result in detected_images:
                confidences = [d['confidence'] for d in result['detections']]
                writer.writerow([
                    os.path.basename(result['image_path']),
                    result['image_path'],
                    result['detection_count'],
                    ', '.join([f"{c:.3f}" for c in confidences])
                ])
        
        # 3. 未检测到公交站的图片文件名
        not_detected_csv = os.path.join(csv_dir, 'not_detected_images.csv')
        with open(not_detected_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['文件名', '图片路径'])
            for result in not_detected_images:
                writer.writerow([
                    os.path.basename(result['image_path']),
                    result['image_path']
                ])
        
        print(f"CSV文件已生成: {csv_dir}")
    
    def save_single_result(self, result, output_folder, image_index):
        """保存单张图片的检测结果"""
        json_dir = os.path.join(output_folder, 'json_results')
        image_name = os.path.splitext(os.path.basename(result['image_path']))[0]
        
        # 保存JSON结果
        json_path = os.path.join(json_dir, f"{image_name}_result.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    
    def save_summary_results(self, all_results, output_folder):
        """保存汇总结果"""
        summary = {
            'total_images': len(all_results),
            'detected_images': len([r for r in all_results if r['detection_count'] > 0]),
            'not_detected_images': len([r for r in all_results if r['detection_count'] == 0]),
            'total_detections': sum(r['detection_count'] for r in all_results),
            'average_confidence': np.mean([
                np.mean([d['confidence'] for d in r['detections']]) 
                for r in all_results if r['detections']
            ]) if any(r['detections'] for r in all_results) else 0,
            'detection_time': datetime.now().isoformat(),
            'results': all_results
        }
        
        summary_path = os.path.join(output_folder, 'detection_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"汇总结果已保存: {summary_path}")
    
    def draw_results(self, image, detections):
        """在图片上绘制检测结果"""
        annotated_image = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            # 绘制边界框
            cv2.rectangle(annotated_image, 
                         (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                         (0, 255, 0), 2)
            
            # 绘制标签
            label = f"Bus Stop: {confidence:.3f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # 绘制标签背景
            cv2.rectangle(annotated_image, 
                         (bbox[0], bbox[1] - label_size[1] - 10),
                         (bbox[0] + label_size[0], bbox[1]), 
                         (0, 255, 0), -1)
            
            # 绘制标签文字
            cv2.putText(annotated_image, label, 
                       (bbox[0], bbox[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated_image

def main():
    # 输入和输出路径
    input_folder = r"F:\training\pre"
    output_folder = r"F:\training\pre_result"
    
    # 创建检测器
    detector = BusStopDetector()
    
    # 批量检测
    detector.detect_folder_batch(
        input_folder=input_folder,
        output_folder=output_folder,
        conf_threshold=0.5,
        enable_sliding_window=True,
        generate_yolo_data=True
    )

if __name__ == "__main__":
    main()