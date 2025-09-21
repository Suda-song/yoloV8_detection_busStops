#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型验证数据集评估脚本
计算各类别在IoU阈值=0.5时的精确率、召回率、AP50和mAP50
"""

import os
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import time
from datetime import datetime

# ==================== 配置区域 ====================
# 输出路径配置（与detect_pic.py保持一致）
OUTPUT_ROOT = "/Users/songdingan/Downloads/detect_bus"  # ← 与detect_pic.py中的OUTPUT_ROOT保持一致

# 模型配置
MODEL_PATH = 'train/weights/best.pt'        # 模型文件路径
DATA_YAML = 'data.yaml'                     # 数据集配置文件路径
CONFIDENCE_THRESHOLD = 0.6                  # 置信度阈值（与detect_pic.py保持一致）
IOU_THRESHOLD = 0.5                         # IoU阈值

# 类别名称映射配置（与detect_pic.py保持一致）
CLASS_NAME_MAPPING = {
    # 将模型检测的类别名称映射为指定的缩写
    # 格式: "模型类别名": "CSV列名"
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

class ModelValidator:
    def __init__(self, model_path=MODEL_PATH, conf_threshold=CONFIDENCE_THRESHOLD, iou_threshold=IOU_THRESHOLD):
        """
        初始化模型验证器
        
        Args:
            model_path: 模型文件路径
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.class_names = []
        self.mapped_class_names = []
        
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
            
            print(f"模型加载成功!")
            print(f"原始检测类别: {self.class_names}")
            print(f"映射后CSV列名: {self.mapped_class_names}")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
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
    
    def validate_model(self, data_yaml_path):
        """
        使用验证数据集评估模型性能
        
        Args:
            data_yaml_path: 数据配置文件路径
            
        Returns:
            dict: 评估指标字典
        """
        print(f"\n开始验证数据集评估...")
        print(f"数据配置文件: {data_yaml_path}")
        print(f"IoU阈值: {self.iou_threshold}")
        print(f"置信度阈值: {self.conf_threshold}")
        
        try:
            # 运行验证
            print("正在运行验证...")
            start_time = time.time()
            
            results = self.model.val(
                data=data_yaml_path,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            validation_time = time.time() - start_time
            print(f"验证完成，耗时: {validation_time:.2f} 秒")
            
            # 提取评估指标
            metrics = self.extract_validation_metrics(results)
            metrics['validation_time'] = validation_time
            
            return metrics
            
        except Exception as e:
            print(f"验证评估失败: {e}")
            raise
    
    def extract_validation_metrics(self, results):
        """
        从验证结果中提取指标
        
        Args:
            results: 验证结果对象
            
        Returns:
            dict: 包含各类别指标的字典
        """
        metrics = {
            'overall': {},
            'per_class': {}
        }
        
        try:
            # 获取整体指标
            if hasattr(results, 'box') and results.box is not None:
                box_metrics = results.box
                
                # 整体指标
                metrics['overall'] = {
                    'map50': float(box_metrics.map50) if hasattr(box_metrics, 'map50') else 0.0,
                    'map50_95': float(box_metrics.map) if hasattr(box_metrics, 'map') else 0.0,
                    'precision': float(box_metrics.mp) if hasattr(box_metrics, 'mp') else 0.0,
                    'recall': float(box_metrics.mr) if hasattr(box_metrics, 'mr') else 0.0,
                    'f1_score': 0.0
                }
                
                # 计算整体F1分数
                p = metrics['overall']['precision']
                r = metrics['overall']['recall']
                if p > 0 and r > 0:
                    metrics['overall']['f1_score'] = 2 * (p * r) / (p + r)
                
                # 每个类别的指标
                if hasattr(box_metrics, 'ap_class_index') and hasattr(box_metrics, 'ap50'):
                    class_indices = box_metrics.ap_class_index
                    ap50_values = box_metrics.ap50
                    
                    # 获取每个类别的precision和recall
                    precision_values = getattr(box_metrics, 'p', [])
                    recall_values = getattr(box_metrics, 'r', [])
                    
                    for i, class_idx in enumerate(class_indices):
                        if class_idx < len(self.class_names):
                            original_class_name = self.class_names[class_idx]
                            mapped_class_name = self.get_mapped_class_name(original_class_name)
                            
                            precision = float(precision_values[i]) if i < len(precision_values) else 0.0
                            recall = float(recall_values[i]) if i < len(recall_values) else 0.0
                            ap50 = float(ap50_values[i]) if i < len(ap50_values) else 0.0
                            
                            # 计算F1分数
                            f1_score = 0.0
                            if precision > 0 and recall > 0:
                                f1_score = 2 * (precision * recall) / (precision + recall)
                            
                            metrics['per_class'][mapped_class_name] = {
                                'original_name': original_class_name,
                                'precision': precision,
                                'recall': recall,
                                'ap50': ap50,
                                'f1_score': f1_score
                            }
                
                print(f"成功提取了 {len(metrics['per_class'])} 个类别的评估指标")
                
            else:
                print("警告: 无法获取验证结果的box指标")
                
        except Exception as e:
            print(f"提取指标时出错: {e}")
            # 创建空的指标结构
            metrics['overall'] = {
                'map50': 0.0,
                'map50_95': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
            metrics['per_class'] = {}
        
        return metrics
    
    def save_validation_results(self, metrics, output_path):
        """
        保存验证结果到CSV文件
        
        Args:
            metrics: 评估指标字典
            output_path: 输出路径
        """
        try:
            # 准备CSV数据
            csv_data = []
            
            # 添加整体指标行
            overall_row = {
                'class_name': 'OVERALL',
                'original_name': 'OVERALL',
                'precision': metrics['overall']['precision'],
                'recall': metrics['overall']['recall'],
                'ap50': metrics['overall']['map50'],
                'f1_score': metrics['overall']['f1_score'],
                'map50_95': metrics['overall']['map50_95']
            }
            csv_data.append(overall_row)
            
            # 添加每个类别的指标
            for mapped_name, class_metrics in metrics['per_class'].items():
                class_row = {
                    'class_name': mapped_name,
                    'original_name': class_metrics['original_name'],
                    'precision': class_metrics['precision'],
                    'recall': class_metrics['recall'],
                    'ap50': class_metrics['ap50'],
                    'f1_score': class_metrics['f1_score'],
                    'map50_95': 0.0  # 单个类别的mAP50-95通常不单独提供
                }
                csv_data.append(class_row)
            
            # 保存到CSV
            df = pd.DataFrame(csv_data)
            
            # 按列排序
            columns = ['class_name', 'original_name', 'precision', 'recall', 'ap50', 'f1_score', 'map50_95']
            df = df[columns]
            
            # 保存文件
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            csv_file = output_dir / 'validation_results.csv'
            df.to_csv(csv_file, index=False, encoding='utf-8-sig')
            
            print(f"\n验证结果已保存到: {csv_file}")
            
            # 显示CSV文件内容预览
            print("\nCSV文件内容预览:")
            print(df.to_string(index=False, float_format='%.4f'))
            
        except Exception as e:
            print(f"保存验证结果失败: {e}")
            raise
    
    def print_validation_results(self, metrics):
        """
        打印验证结果
        
        Args:
            metrics: 评估指标字典
        """
        print("\n" + "="*80)
        print("验证数据集评估结果")
        print("="*80)
        print(f"IoU阈值: {self.iou_threshold}")
        print(f"置信度阈值: {self.conf_threshold}")
        if 'validation_time' in metrics:
            print(f"验证时间: {metrics['validation_time']:.2f} 秒")
        print("-"*80)
        
        # 打印整体指标
        overall = metrics['overall']
        print(f"整体指标:")
        print(f"  mAP50 (所有类别平均): {overall['map50']:.4f}")
        print(f"  mAP50-95: {overall['map50_95']:.4f}")
        print(f"  平均精确率: {overall['precision']:.4f}")
        print(f"  平均召回率: {overall['recall']:.4f}")
        print(f"  平均F1分数: {overall['f1_score']:.4f}")
        
        # 打印每个类别的指标
        if metrics['per_class']:
            print(f"\n各类别详细指标:")
            print(f"{'类别':<8} {'原始名称':<12} {'精确率':<8} {'召回率':<8} {'AP50':<8} {'F1分数':<8}")
            print("-" * 60)
            
            for mapped_name, class_metrics in metrics['per_class'].items():
                print(f"{mapped_name:<8} {class_metrics['original_name']:<12} "
                      f"{class_metrics['precision']:<8.4f} {class_metrics['recall']:<8.4f} "
                      f"{class_metrics['ap50']:<8.4f} {class_metrics['f1_score']:<8.4f}")
        
        print("="*80)
        print("指标说明:")
        print("- 精确率 (Precision): 预测为正例中实际为正例的比例")
        print("- 召回率 (Recall): 实际正例中被正确预测的比例")
        print("- AP50: 在IoU=0.5时的平均精确率")
        print("- F1分数: 精确率和召回率的调和平均值")
        print("- mAP50: 所有类别在IoU=0.5时的平均AP值")
        print("- 类别映射: DP=顶棚, LG=栏杆, ZY=座椅, FS=座椅栏杆, ZP=站牌, CBFH=侧边防护")

def main():
    """主函数"""
    
    # 检查配置
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 模型文件不存在: {MODEL_PATH}")
        print("请检查 MODEL_PATH 设置是否正确")
        return
    
    if not os.path.exists(DATA_YAML):
        print(f"❌ 错误: 数据配置文件不存在: {DATA_YAML}")
        print("请检查 DATA_YAML 设置是否正确")
        print("data.yaml文件应包含验证数据集的配置信息")
        return
    
    print("🎯 模型验证数据集评估")
    print("="*50)
    print(f"模型路径: {MODEL_PATH}")
    print(f"数据配置: {DATA_YAML}")
    print(f"输出路径: {OUTPUT_ROOT}")
    print(f"IoU阈值: {IOU_THRESHOLD}")
    print(f"置信度阈值: {CONFIDENCE_THRESHOLD}")
    print("="*50)
    
    try:
        # 创建验证器
        validator = ModelValidator()
        
        # 加载模型
        validator.load_model()
        
        # 运行验证
        metrics = validator.validate_model(DATA_YAML)
        
        # 保存结果
        validator.save_validation_results(metrics, OUTPUT_ROOT)
        
        # 显示结果
        validator.print_validation_results(metrics)
        
        print("\n✅ 验证评估完成!")
        print(f"结果已保存到: {OUTPUT_ROOT}/validation_results.csv")
        
    except Exception as e:
        print(f"\n❌ 验证评估失败: {e}")
        print("请检查:")
        print("1. 模型文件是否存在且正确")
        print("2. data.yaml配置是否正确")
        print("3. 验证数据集路径是否正确")
        print("4. 验证数据集标签格式是否正确")

if __name__ == "__main__":
    main()
