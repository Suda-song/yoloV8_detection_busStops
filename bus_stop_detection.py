import cv2
from ultralytics import YOLO
import os

class BusStopDetector:
    def __init__(self, model_path='bus_stop_detection/v8s_exp14/weights/best.pt'):
        """初始化公交站检测器"""
        self.model = YOLO(model_path)
        print(f"模型加载成功: {model_path}")
        
    def detect_image(self, image_path, conf_threshold=0.5):
        """
        检测图片中的公交站
        Args:
            image_path: 图片路径
            conf_threshold: 置信度阈值 (默认0.5)
        Returns:
            检测结果列表，每个结果包含边界框坐标和置信度
        """
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图片: {image_path}")
            return []
            
        # 进行检测
        results = self.model(image, conf=conf_threshold)
        
        # 处理检测结果
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
        
        # 在图片上绘制检测结果并保存
        if detections:
            annotated_image = self.draw_results(image, detections)
            output_path = f"detected_{os.path.basename(image_path)}"
            cv2.imwrite(output_path, annotated_image)
            print(f"检测结果已保存到: {output_path}")
        
        return detections
    
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
    """简单的使用示例"""
    detector = BusStopDetector()
    
    # 输入图片路径
    image_path = input("请输入图片路径: ").strip()
    
    if os.path.exists(image_path):
        # 检测公交站
        detections = detector.detect_image(image_path)
        
        # 输出结果
        if detections:
            print(f"检测到 {len(detections)} 个公交站:")
            for i, det in enumerate(detections, 1):
                print(f"  公交站 {i}: 置信度 {det['confidence']:.2f}")
        else:
            print("未检测到公交站")
    else:
        print("图片文件不存在")

if __name__ == "__main__":
    main()