from ultralytics import YOLO

# 加载预训练模型（推荐yolov8s或yolov8m）
model = YOLO('yolov8s.pt')  # 轻量级选yolov8n[1,7](@ref)

# 训练参数配置
results = model.train(
    data='transportStop.yaml',
    epochs=30,         # 迭代次数
    imgsz=640,             # 输入分辨率
    batch=8,              # 批大小（根据GPU调整）
    lr0=0.01,              # 初始学习率
    augment=True,          # 启用数据增强（旋转、缩放等）
    device='cpu',        # GPU ID（CPU用'cpu'）
    project='bus_stop_detection',
    name='v8s_exp1'
)