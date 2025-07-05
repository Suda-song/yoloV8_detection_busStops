from ultralytics import YOLO
import os
import yaml
from datetime import datetime

def continue_training():
    """基于exp14最佳模型继续训练"""
    
    # 训练配置
    config = {
        'model_path': 'bus_stop_detection/v8s_exp14/weights/best.pt',  # 使用exp14的最佳模型
        'data_yaml': 'transportStop.yaml',
        'epochs': 50,  # 新的训练轮数
        'batch_size': 8,
        'img_size': 640,
        'device': 'cpu',
        'project': 'F:/training/pre_result',
        'name': 'v8s_exp16',  # 新的实验名称
        'patience': 50,
        'save_period': 10,
        'resume': False,  # 从头开始训练，但使用预训练权重
        'pretrained': True,  # 使用预训练权重
    }
    
    print("=== 基于exp14最佳模型继续训练配置 ===")
    print(f"预训练模型: {config['model_path']}")
    print(f"数据配置: {config['data_yaml']}")
    print(f"训练轮数: {config['epochs']}")
    print(f"批次大小: {config['batch_size']}")
    print(f"图片尺寸: {config['img_size']}")
    print(f"设备: {config['device']}")
    print(f"项目路径: {config['project']}")
    print(f"实验名称: {config['name']}")
    print(f"训练模式: 从头开始训练（使用exp14最佳模型作为预训练权重）")
    
    # 检查模型文件是否存在
    if not os.path.exists(config['model_path']):
        print(f"错误: 预训练模型文件不存在: {config['model_path']}")
        return
    
    # 检查数据配置文件是否存在
    if not os.path.exists(config['data_yaml']):
        print(f"错误: 数据配置文件不存在: {config['data_yaml']}")
        return
    
    # 确保输出目录存在
    os.makedirs(config['project'], exist_ok=True)
    
    # 加载模型
    print(f"\n正在加载预训练模型: {config['model_path']}")
    model = YOLO(config['model_path'])
    
    # 开始训练
    print(f"\n开始基于exp14最佳模型继续训练...")
    print(f"训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 执行训练
        results = model.train(
            data=config['data_yaml'],
            epochs=config['epochs'],
            batch=config['batch_size'],
            imgsz=config['img_size'],
            device=config['device'],
            project=config['project'],
            name=config['name'],
            patience=config['patience'],
            save_period=config['save_period'],
            resume=config['resume'],
            pretrained=config['pretrained'],
            verbose=True,
            plots=True,  # 生成训练图表
            save=True,   # 保存模型
            exist_ok=True,  # 如果实验文件夹存在则覆盖
        )
        
        print(f"\n训练完成!")
        print(f"训练结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 显示训练结果
        print(f"\n=== 训练结果 ===")
        print(f"最佳模型保存路径: {results.save_dir}")
        print(f"最佳模型文件: {results.best}")
        print(f"最终模型文件: {results.last}")
        
        # 显示最终指标
        if hasattr(results, 'results_dict'):
            final_metrics = results.results_dict
            print(f"\n最终训练指标:")
            print(f"  mAP50: {final_metrics.get('metrics/mAP50(B)', 'N/A'):.4f}")
            print(f"  mAP50-95: {final_metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
            print(f"  Precision: {final_metrics.get('metrics/precision(B)', 'N/A'):.4f}")
            print(f"  Recall: {final_metrics.get('metrics/recall(B)', 'N/A'):.4f}")
        
        return results
        
    except Exception as e:
        print(f"训练过程中出现错误: {str(e)}")
        return None

def create_training_config():
    """创建训练配置文件"""
    config = {
        'task': 'detect',
        'mode': 'train',
        'model': 'bus_stop_detection/v8s_exp14/weights/best.pt',  # 使用exp14最佳模型
        'data': 'transportStop.yaml',
        'epochs': 50,  # 新的训练轮数
        'batch': 8,
        'imgsz': 640,
        'device': 'cpu',
        'workers': 8,
        'project': 'F:/training/pre_result',
        'name': 'v8s_exp16',
        'exist_ok': True,
        'pretrained': True,  # 使用预训练权重
        'optimizer': 'auto',
        'verbose': True,
        'seed': 0,
        'deterministic': True,
        'single_cls': False,
        'rect': False,
        'cos_lr': False,
        'close_mosaic': 10,
        'resume': False,  # 从头开始训练
        'amp': True,
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        'multi_scale': False,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'split': 'val',
        'save_json': False,
        'conf': None,
        'iou': 0.7,
        'max_det': 300,
        'half': False,
        'dnn': False,
        'plots': True,
        'source': None,
        'vid_stride': 1,
        'stream_buffer': False,
        'visualize': False,
        'augment': True,
        'agnostic_nms': False,
        'classes': None,
        'retina_masks': False,
        'embed': None,
        'show': False,
        'save_frames': False,
        'save_txt': False,
        'save_conf': False,
        'save_crop': False,
        'show_labels': True,
        'show_conf': True,
        'show_boxes': True,
        'line_width': None,
        'format': 'torchscript',
        'keras': False,
        'optimize': False,
        'int8': False,
        'dynamic': False,
        'simplify': True,
        'opset': None,
        'workspace': None,
        'nms': False,
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 1.0,
        'nbs': 64,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'bgr': 0.0,
        'mosaic': 1.0,
        'mixup': 0.0,
        'cutmix': 0.0,
        'copy_paste': 0.0,
        'copy_paste_mode': 'flip',
        'auto_augment': 'randaugment',
        'erasing': 0.4,
        'cfg': None,
        'tracker': 'botsort.yaml',
        'save_dir': 'F:/training/pre_result/v8s_exp16',
        'patience': 50,
        'save_period': 10
    }
    
    # 保存配置文件到指定路径
    config_file = 'F:/training/pre_result/continue_training_config.yaml'
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"训练配置文件已保存: {config_file}")
    return config_file

def main():
    """主函数"""
    print("=== 基于exp14最佳模型继续训练 ===")
    
    # 检查必要的文件
    required_files = [
        'bus_stop_detection/v8s_exp14/weights/best.pt',  # 使用最佳模型
        'transportStop.yaml'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("错误: 以下必要文件不存在:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\n请确保exp14训练已完成，并且文件路径正确。")
        return
    
    # 创建训练配置文件
    config_file = create_training_config()
    
    # 开始继续训练
    results = continue_training()
    
    if results:
        print(f"\n=== 训练成功完成 ===")
        print(f"新模型保存在: {results.save_dir}")
        print(f"最佳模型: {results.best}")
        print(f"最终模型: {results.last}")
        
        # 建议下一步操作
        print(f"\n=== 建议的下一步操作 ===")
        print("1. 检查训练曲线图，评估训练效果")
        print("2. 使用新模型进行测试检测")
        print("3. 如果效果满意，可以更新检测脚本中的模型路径")
        print("4. 如果还需要继续训练，可以再次运行此脚本")
    else:
        print("训练失败，请检查错误信息。")

if __name__ == "__main__":
    main()