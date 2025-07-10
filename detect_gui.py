import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, 
    QHBoxLayout, QLineEdit, QMessageBox, QSpinBox, QStatusBar, QGroupBox, 
    QSizePolicy
)
from PySide6.QtGui import QPixmap, QImage, QIcon, QFont
from PySide6.QtCore import Qt
from ultralytics import YOLO
import cv2
import os

class YOLODetectGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('YOLO目标检测系统')
        self.setWindowIcon(QIcon())
        self.setMinimumSize(800, 600)
        self.setStyleSheet('QWidget { font-family: "PingFang SC", "Microsoft YaHei", Arial, sans-serif; font-size: 14px; }')
        self.model_path = 'train/weights/best.pt'  # 修改路径分隔符
        self.model = YOLO(self.model_path)
        self.image_path = None
        self.results = None
        self.init_ui()

    def init_ui(self):
        # 图片显示
        self.img_label = QLabel('请选择图片')
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setStyleSheet('background: #f0f0f0; border-radius: 6px;')
        self.img_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.img_label.setScaledContents(True)

        # 图片路径显示和选择
        self.img_path_edit = QLineEdit()
        self.img_path_edit.setReadOnly(True)
        self.btn_select_img = QPushButton('选择图片')
        self.btn_select_img.setStyleSheet('padding: 6px 18px;')
        self.btn_select_img.clicked.connect(self.select_image)

        img_path_layout = QHBoxLayout()
        img_path_layout.addWidget(self.img_path_edit)
        img_path_layout.addWidget(self.btn_select_img)

        img_group = QGroupBox('图片选择')
        img_group.setLayout(img_path_layout)

        # 模型路径显示和选择
        self.model_path_edit = QLineEdit(self.model_path)
        self.model_path_edit.setReadOnly(True)
        self.btn_select_model = QPushButton('选择模型')
        self.btn_select_model.setStyleSheet('padding: 6px 18px;')
        self.btn_select_model.clicked.connect(self.select_model)

        model_path_layout = QHBoxLayout()
        model_path_layout.addWidget(self.model_path_edit)
        model_path_layout.addWidget(self.btn_select_model)

        model_group = QGroupBox('模型选择')
        model_group.setLayout(model_path_layout)

        # 置信度阈值设置
        self.conf_label = QLabel('置信度阈值:')
        self.conf_spin = QSpinBox()
        self.conf_spin.setRange(1, 100)
        self.conf_spin.setValue(25)
        self.conf_spin.setSuffix('%')
        self.conf_spin.setToolTip('只显示高于该置信度的检测结果')
        self.conf_spin.setFixedWidth(80)
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(self.conf_label)
        conf_layout.addWidget(self.conf_spin)
        conf_layout.addStretch()

        conf_group = QGroupBox('检测参数')
        conf_group.setLayout(conf_layout)

        # 检测按钮
        self.btn_detect = QPushButton('开始检测')
        self.btn_detect.setStyleSheet('background: #0078d7; color: white; font-weight: bold; padding: 10px 30px; border-radius: 6px;')
        self.btn_detect.setFont(QFont('PingFang SC', 12, QFont.Weight.Bold))
        self.btn_detect.clicked.connect(self.detect_image)
        self.btn_detect.setEnabled(False)
        self.btn_detect.setMinimumHeight(36)

        # 检测结果统计标签
        self.result_label = QLabel('检测结果: 等待检测')
        self.result_label.setFont(QFont('PingFang SC', 16, QFont.Weight.Bold))
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet('color: #005a9e; padding: 15px; background: #ddefff; border-radius: 8px;')
        self.result_label.setMinimumHeight(60)
        self.result_label.setWordWrap(True)

        result_group = QGroupBox('检测结果')
        result_layout = QVBoxLayout()
        result_layout.addWidget(self.result_label)
        result_group.setLayout(result_layout)

        # 状态栏
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet('QStatusBar{background:#e6f0fa; color:#0078d7; font-weight:bold;}')
        self.status_bar.showMessage('准备就绪')

        # 左侧：图片
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.img_label)

        # 右侧：参数与结果
        right_layout = QVBoxLayout()
        right_layout.addWidget(img_group)
        right_layout.addWidget(model_group)
        right_layout.addWidget(conf_group)
        right_layout.addWidget(self.btn_detect)
        right_layout.addWidget(result_group)
        right_layout.addStretch()
        right_layout.setSpacing(15)

        # 主布局
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 2)
        main_layout.addLayout(right_layout, 1)
        main_layout.setSpacing(20)

        # 整体垂直布局
        v_layout = QVBoxLayout()
        v_layout.addLayout(main_layout)
        v_layout.addWidget(self.status_bar)
        v_layout.setContentsMargins(15, 15, 15, 15)
        self.setLayout(v_layout)

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, '选择图片', '', 'Image Files (*.png *.jpg *.jpeg *.bmp)')
        if file_path:
            self.image_path = file_path
            self.img_path_edit.setText(file_path)
            pixmap = QPixmap(file_path)
            self.img_label.setPixmap(pixmap)
            self.update_detect_btn_state()
            self.result_label.setText('检测结果: 等待检测')
            self.status_bar.showMessage('准备就绪')

    def select_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, '选择模型', '', 'Model Files (*.pt *.onnx *.yaml *.yml)')
        if file_path and os.path.exists(file_path):
            try:
                self.model = YOLO(file_path)
                self.model_path = file_path
                self.model_path_edit.setText(file_path)
                self.update_detect_btn_state()
                self.status_bar.showMessage('模型加载成功')
            except Exception as e:
                QMessageBox.warning(self, '模型加载失败', f'无法加载模型文件：\n{e}')
                self.status_bar.showMessage('模型加载失败')

    def update_detect_btn_state(self):
        enable = bool(self.image_path) and bool(self.model_path) and os.path.exists(self.model_path)
        self.btn_detect.setEnabled(enable)

    def detect_image(self):
        if not self.image_path or not self.model_path:
            QMessageBox.warning(self, '缺少信息', '请先选择图片和模型文件。')
            return

        self.status_bar.showMessage('正在检测...')
        QApplication.processEvents()

        try:
            conf = self.conf_spin.value() / 100.0
            results = self.model(self.image_path, conf=conf)
            self.results = results

            # 显示检测后的图片
            result_img = results[0].plot()
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            h, w, ch = result_img.shape
            bytes_per_line = ch * w
            qt_img = QImage(result_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_img)
            self.img_label.setPixmap(pixmap)

            # 计算检测结果统计
            result_text = self.get_detection_summary(results[0])
            self.result_label.setText(result_text)

            self.status_bar.showMessage('检测完成')

        except Exception as e:
            self.status_bar.showMessage('检测失败')
            QMessageBox.critical(self, '检测出错', f'检测过程中发生错误：\n{e}')
            self.result_label.setText('检测结果: 检测失败')

    def get_detection_summary(self, result):
        """根据检测结果生成统计信息"""
        boxes = result.boxes
        names = result.names
        
        if boxes is None or boxes.shape[0] == 0:
            return "检测结果: 未检测到任何目标"
        
        # 统计各类别的数量
        class_counts = {}
        total_objects = boxes.shape[0]
        
        for box in boxes:
            cls_id = int(box.cls[0]) if hasattr(box, 'cls') and box.cls is not None else -1
            if cls_id != -1 and names and cls_id in names:
                cls_name = names[cls_id]
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        
        # 生成结果文本
        result_lines = [f"检测到 {total_objects} 个目标"]
        
        # 按类别显示数量
        for cls_name, count in class_counts.items():
            result_lines.append(f"{cls_name}: {count} 个")
        
        return "\n".join(result_lines)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = YOLODetectGUI()
    window.show()
    sys.exit(app.exec())