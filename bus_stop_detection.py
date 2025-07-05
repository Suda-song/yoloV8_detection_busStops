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
        
        if output_dir:
            # 确保输出目录存在
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
            'detection_count': len(detections)
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
    
    def detect_folder_batch(self, input_folder, output_folder, conf_threshold=0.5, 
                           enable_sliding_window=False):
        """
        批量检测文件夹中的图片，按检测结果分类保存
        Args:
            input_folder: 输入图片文件夹路径
            output_folder: 输出结果文件夹路径
            conf_threshold: 置信度阈值
            enable_sliding_window: 是否启用滑动窗口检测
        """
        # 支持的图片格式
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        # 创建输出文件夹结构
        self.create_output_folders(output_folder)
        
        # 获取文件夹中的所有图片
        image_files = []
        for file in os.listdir(input_folder):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(input_folder, file))
        
        if not image_files:
            print(f"在文件夹 {input_folder} 中未找到图片文件")
            return
        
        print(f"找到 {len(image_files)} 张图片，开始检测...")
        
        # 存储所有检测结果
        all_results = []
        detected_images = []  # 检测到公交站的图片
        not_detected_images = []  # 未检测到公交站的图片
        total_detections = 0
        
        # 逐张图片检测
        for i, image_path in enumerate(image_files, 1):
            print(f"\n正在处理第 {i}/{len(image_files)} 张图片: {os.path.basename(image_path)}")
            
            # 检测当前图片
            result = self.detect_image(
                image_path, conf_threshold, None, enable_sliding_window
            )
            
            if result:
                all_results.append(result)
                total_detections += result['detection_count']
                
                # 输出当前图片的检测结果
                if result['detections']:
                    print(f"  ✓ 检测到 {len(result['detections'])} 个公交站")
                    for j, det in enumerate(result['detections'], 1):
                        print(f"    公交站 {j}: 置信度 {det['confidence']:.2f}")
                    
                    # 添加到检测到的图片列表
                    detected_images.append({
                        'filename': os.path.basename(image_path),
                        'full_path': image_path,
                        'detection_count': result['detection_count'],
                        'max_confidence': max([det['confidence'] for det in result['detections']])
                    })
                else:
                    print(f"  ✗ 未检测到公交站")
                    # 添加到未检测到的图片列表
                    not_detected_images.append({
                        'filename': os.path.basename(image_path),
                        'full_path': image_path
                    })
                
                # 立即保存当前结果到JSON文件
                self.save_single_result(result, output_folder, i)
                
                print(f"  ✓ 第 {i} 张图片检测完成并保存")
            else:
                print(f"  ✗ 第 {i} 张图片处理失败")
        
        # 分类保存图片
        self.classify_and_save_images(all_results, output_folder)
        
        # 生成CSV文件
        self.generate_csv_files(image_files, detected_images, not_detected_images, output_folder)
        
        # 保存总体结果
        self.save_summary_results(all_results, output_folder)
        
        print(f"\n=== 检测完成 ===")
        print(f"总图片数: {len(image_files)}")
        print(f"成功处理: {len(all_results)}")
        print(f"检测到公交站的图片: {len(detected_images)}")
        print(f"未检测到公交站的图片: {len(not_detected_images)}")
        print(f"总检测数: {total_detections}")
        
        # 显示置信度统计
        all_confidences = []
        for result in all_results:
            for det in result['detections']:
                all_confidences.append(det['confidence'])
        
        if all_confidences:
            print(f"平均置信度: {sum(all_confidences)/len(all_confidences):.3f}")
            print(f"最高置信度: {max(all_confidences):.3f}")
            print(f"最低置信度: {min(all_confidences):.3f}")
    
    def create_output_folders(self, output_folder):
        """创建输出文件夹结构"""
        # 主输出文件夹
        os.makedirs(output_folder, exist_ok=True)
        
        # 分类文件夹
        detected_folder = os.path.join(output_folder, "detected_images")
        not_detected_folder = os.path.join(output_folder, "not_detected_images")
        
        os.makedirs(detected_folder, exist_ok=True)
        os.makedirs(not_detected_folder, exist_ok=True)
        
        print(f"输出文件夹结构已创建:")
        print(f"  - 主文件夹: {output_folder}")
        print(f"  - 检测到公交站: {detected_folder}")
        print(f"  - 未检测到公交站: {not_detected_folder}")
    
    def classify_and_save_images(self, all_results, output_folder):
        """分类保存图片"""
        detected_folder = os.path.join(output_folder, "detected_images")
        not_detected_folder = os.path.join(output_folder, "not_detected_images")
        
        for result in all_results:
            image_path = result['image_path']
            filename = os.path.basename(image_path)
            
            # 读取原图
            image = cv2.imread(image_path)
            if image is None:
                continue
            
            # 在图片上绘制检测结果
            annotated_image = self.draw_results(image, result['detections'])
            
            # 根据检测结果分类保存
            if result['detection_count'] > 0:
                # 检测到公交站，保存到detected_images文件夹
                output_path = os.path.join(detected_folder, f"detected_{filename}")
                cv2.imwrite(output_path, annotated_image)
            else:
                # 未检测到公交站，保存到not_detected_images文件夹
                output_path = os.path.join(not_detected_folder, f"detected_{filename}")
                cv2.imwrite(output_path, annotated_image)
    
    def generate_csv_files(self, all_images, detected_images, not_detected_images, output_folder):
        """生成CSV文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 全部检测的照片文件名
        all_images_csv = os.path.join(output_folder, f"all_images_{timestamp}.csv")
        with open(all_images_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['序号', '文件名', '完整路径'])
            for i, image_path in enumerate(all_images, 1):
                writer.writerow([i, os.path.basename(image_path), image_path])
        
        # 2. 检测到公交站的照片文件名
        detected_csv = os.path.join(output_folder, f"detected_images_{timestamp}.csv")
        with open(detected_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['序号', '文件名', '完整路径', '检测数量', '最高置信度'])
            for i, img_info in enumerate(detected_images, 1):
                writer.writerow([
                    i, 
                    img_info['filename'], 
                    img_info['full_path'],
                    img_info['detection_count'],
                    f"{img_info['max_confidence']:.3f}"
                ])
        
        # 3. 没有检测到公交站的文件名
        not_detected_csv = os.path.join(output_folder, f"not_detected_images_{timestamp}.csv")
        with open(not_detected_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['序号', '文件名', '完整路径'])
            for i, img_info in enumerate(not_detected_images, 1):
                writer.writerow([i, img_info['filename'], img_info['full_path']])
        
        print(f"\nCSV文件已生成:")
        print(f"  - 全部图片: {os.path.basename(all_images_csv)}")
        print(f"  - 检测到公交站: {os.path.basename(detected_csv)}")
        print(f"  - 未检测到公交站: {os.path.basename(not_detected_csv)}")
    
    def save_single_result(self, result, output_folder, image_index):
        """
        保存单张图片的检测结果
        Args:
            result: 检测结果
            output_folder: 输出文件夹
            image_index: 图片索引
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"result_{image_index:03d}_{os.path.basename(result['image_path']).split('.')[0]}_{timestamp}.json"
        filepath = os.path.join(output_folder, filename)
        
        save_data = {
            'timestamp': datetime.now().isoformat(),
            'image_index': image_index,
            'image_path': result['image_path'],
            'image_size': result['image_size'],
            'detection_count': result['detection_count'],
            'detections': result['detections']
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"    结果已保存: {filename}")
    
    def save_summary_results(self, all_results, output_folder):
        """
        保存总体检测结果
        Args:
            all_results: 所有检测结果
            output_folder: 输出文件夹
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = os.path.join(output_folder, f"summary_results_{timestamp}.json")
        
        # 计算统计信息
        total_images = len(all_results)
        total_detections = sum(r['detection_count'] for r in all_results)
        images_with_detections = len([r for r in all_results if r['detection_count'] > 0])
        
        all_confidences = []
        for result in all_results:
            for det in result['detections']:
                all_confidences.append(det['confidence'])
        
        summary_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_images': total_images,
                'total_detections': total_detections,
                'images_with_detections': images_with_detections,
                'images_without_detections': total_images - images_with_detections,
                'detection_rate': images_with_detections / total_images if total_images > 0 else 0,
                'average_confidence': sum(all_confidences)/len(all_confidences) if all_confidences else 0,
                'max_confidence': max(all_confidences) if all_confidences else 0,
                'min_confidence': min(all_confidences) if all_confidences else 0
            },
            'all_results': all_results
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n总体结果已保存到: {summary_file}")
    
    def draw_results(self, image, detections):
        """在图片上绘制检测结果"""
        result_image = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            
            # 绘制边界框（绿色）
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制标签
            label = f"Bus Stop: {confidence:.2f}"
            cv2.putText(result_image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return result_image

def main():
    """主函数"""
    # 设置输入和输出文件夹路径
    input_folder = r"F:\training\pre"
    output_folder = r"F:\training\pre_result"
    
    # 检查输入文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"输入文件夹不存在: {input_folder}")
        return
    
    if not os.path.isdir(input_folder):
        print(f"输入路径不是文件夹: {input_folder}")
        return
    
    print(f"输入文件夹: {input_folder}")
    print(f"输出文件夹: {output_folder}")
    
    # 创建检测器
    detector = BusStopDetector()
    
    # 开始批量检测（启用滑动窗口检测以处理大图片）
    detector.detect_folder_batch(
        input_folder, 
        output_folder, 
        conf_threshold=0.5,
        enable_sliding_window=True  # 启用滑动窗口检测
    )

if __name__ == "__main__":
    main()