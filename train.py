from ultralytics import YOLO
# 多线程添加代码
if __name__ == '__main__': 
    # 加载预训练模型
    model = YOLO('yolov8n.pt')
 
    # 训练模型
    model.train(data='data.yaml',
                    epochs=100, 
                    batch=32, 
                    imgsz=640,
                    # 数据增强设置（YOLOv8x推荐值）
                    scale=0.5,                     # 随机缩放增强
                    mosaic=1.0,                    # 启用 Mosaic（图像拼接）
                    mixup=0.0,                     # MixUp 图像混合
                    copy_paste=0.1,                # CopyPaste 增强概率
                    device="0",
                    workers=8, 
                )