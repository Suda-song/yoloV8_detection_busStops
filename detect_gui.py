import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QHBoxLayout, QLineEdit, QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView, QSpinBox, QStatusBar, QGroupBox, QSizePolicy, QSpacerItem
)
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont
from PyQt5.QtCore import Qt
from ultralytics import YOLO
import cv2
import os
# import time # 不再需要 time 模块

class YOLODetectGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('YOLO目标检测系统')
        self.setWindowIcon(QIcon())  # 可自定义图标路径
        self.setMinimumSize(800, 600) # 设置最小尺寸，允许缩放
        self.setStyleSheet('QWidget { font-family: "Microsoft YaHei", Arial, sans-serif; font-size: 14px; }')
        self.model_path = r'train\weights\best.pt'
        self.model = YOLO(self.model_path)
        self.image_path = None
        # self.save_dir = None # 移除保存目录
        self.results = None
        self.init_ui()

    def init_ui(self):
        # 图片显示
        # 图片显示 (允许缩放)
        self.img_label = QLabel('请选择图片')
        self.img_label.setAlignment(Qt.AlignCenter)
        # 移除 setFixedSize 以允许缩放
        self.img_label.setStyleSheet('background: #f0f0f0; border-radius: 6px;') # 柔和背景和圆角
        self.img_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored) # 允许标签根据内容缩放
        self.img_label.setScaledContents(True) # 让标签自动缩放其内容（图片）

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

        # 移除保存目录相关UI
        # self.save_dir_edit = QLineEdit()
        # self.save_dir_edit.setReadOnly(True)
        # self.btn_select_save_dir = QPushButton('选择保存目录')
        # self.btn_select_save_dir.setStyleSheet('padding: 6px 18px;')
        # self.btn_select_save_dir.clicked.connect(self.select_save_dir)
        #
        # save_dir_layout = QHBoxLayout()
        # save_dir_layout.addWidget(self.save_dir_edit)
        # save_dir_layout.addWidget(self.btn_select_save_dir)
        #
        # save_group = QGroupBox('保存设置')
        # save_group.setLayout(save_dir_layout)

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
        self.btn_detect.setFont(QFont('Microsoft YaHei', 12, QFont.Bold))
        self.btn_detect.clicked.connect(self.detect_image)
        self.btn_detect.setEnabled(False)
        self.btn_detect.setMinimumHeight(36)

        # 移除检测结果表格，替换为检测结果统计
        # self.table = QTableWidget(0, 6)
        # self.table.setHorizontalHeaderLabels(['类别', '置信度', 'x1', 'y1', 'x2', 'y2'])
        # self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        # self.table.setMinimumHeight(180)
        # self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # self.table.setStyleSheet('background: #f8faff;')
        #
        # result_group = QGroupBox('检测结果')
        # result_layout = QVBoxLayout()
        # result_layout.addWidget(self.table)
        # result_group.setLayout(result_layout)

        # 添加检测结果统计标签
        # 添加检测结果统计标签 (增大字体和调整样式)
        self.result_label = QLabel('检测结果: 等待检测')
        self.result_label.setFont(QFont('Microsoft YaHei', 16, QFont.Bold)) # 调整字体大小
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet('color: #005a9e; padding: 15px; background: #ddefff; border-radius: 8px;') # 调整颜色、内边距和圆角
        self.result_label.setMinimumHeight(60) # 适当的高度
        self.result_label.setWordWrap(True) # 允许文本换行

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
        # right_layout.addWidget(save_group) # 移除保存组
        right_layout.addWidget(conf_group)
        right_layout.addWidget(self.btn_detect)
        right_layout.addWidget(result_group) # 添加结果组
        # right_layout.addWidget(result_group) # 移除结果组
        right_layout.addStretch() # 添加伸缩项，让控件靠上
        right_layout.setSpacing(15) # 增加右侧控件垂直间距

        # 主布局 (调整左右比例和间距)
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 2) # 左侧（图片）占更大比例
        main_layout.addLayout(right_layout, 1) # 右侧（控件）占较小比例
        main_layout.setSpacing(20) # 增加左右布局间距

        # 整体垂直布局 (增加边距)
        v_layout = QVBoxLayout()
        v_layout.addLayout(main_layout)
        v_layout.addWidget(self.status_bar)
        v_layout.setContentsMargins(15, 15, 15, 15) # 设置窗口内容边距
        self.setLayout(v_layout)

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, '选择图片', '', 'Image Files (*.png *.jpg *.jpeg *.bmp)')
        if file_path:
            self.image_path = file_path
            self.img_path_edit.setText(file_path)
            pixmap = QPixmap(file_path) # 直接加载图片
            self.img_label.setPixmap(pixmap) # 设置图片，由 setScaledContents 处理缩放
            self.update_detect_btn_state()
            self.result_label.setText('检测结果: 等待检测') # 重置结果
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

    # 移除 select_save_dir 方法
    # def select_save_dir(self):
    #     dir_path = QFileDialog.getExistingDirectory(self, '选择保存目录', '')
    #     if dir_path:
    #         self.save_dir = dir_path
    #         self.save_dir_edit.setText(dir_path)
    #         self.update_detect_btn_state()

    def update_detect_btn_state(self):
        # 移除 self.save_dir 的检查
        enable = bool(self.image_path) and bool(self.model_path) and os.path.exists(self.model_path)
        self.btn_detect.setEnabled(enable)

    def detect_image(self):
        # 移除 self.save_dir 的检查
        if not self.image_path or not self.model_path:
            QMessageBox.warning(self, '缺少信息', '请先选择图片和模型文件。')
            return

        self.status_bar.showMessage('正在检测...') # 更新状态栏
        QApplication.processEvents() # 处理界面事件，防止假死

        try:
            conf = self.conf_spin.value() / 100.0
            results = self.model(self.image_path, conf=conf)
            self.results = results

            # 显示检测后的图片
            result_img = results[0].plot()
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            h, w, ch = result_img.shape
            bytes_per_line = ch * w
            qt_img = QImage(result_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_img) # 直接从 QImage 创建 QPixmap
            self.img_label.setPixmap(pixmap) # 设置图片，由 setScaledContents 处理缩放

            # 计算检测结果统计
            result_text = self.get_detection_summary(results[0])
            self.result_label.setText(result_text)

            # 移除保存图片的代码
            # timestamp = time.strftime('%Y%m%d_%H%M%S')
            # save_path = os.path.join(self.save_dir, f'result_{timestamp}.png')
            # cv2.imwrite(save_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))

            # 移除显示结果表格的代码
            # self.show_results_table(results[0])

            self.status_bar.showMessage('检测完成') # 更新状态栏

            # 移除保存成功的消息框
            # QMessageBox.information(self, '保存成功', f'检测结果已保存到：\n{save_path}')

        except Exception as e:
            self.status_bar.showMessage('检测失败')
            QMessageBox.critical(self, '检测出错', f'检测过程中发生错误：\n{e}')
            self.result_label.setText('检测结果: 检测失败') # 重置结果

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

    # 移除原来的 calculate_score 方法，替换为 get_detection_summary

    # 移除 show_results_table 方法
    # def show_results_table(self, result):
    #     boxes = result.boxes
    #     names = result.names
    #     self.table.setRowCount(0)
    #     if boxes is not None and boxes.shape[0] > 0:
    #         for i, box in enumerate(boxes):
    #             cls_id = int(box.cls[0]) if hasattr(box, 'cls') else int(box[5])
    #             cls_name = names[cls_id] if names and cls_id in names else str(cls_id)
    #             conf = float(box.conf[0]) if hasattr(box, 'conf') else float(box[4])
    #             xyxy = box.xyxy[0].cpu().numpy() if hasattr(box, 'xyxy') else box[:4]
    #             x1, y1, x2, y2 = [int(x) for x in xyxy]
    #             self.table.insertRow(i)
    #             self.table.setItem(i, 0, QTableWidgetItem(cls_name))
    #             self.table.setItem(i, 1, QTableWidgetItem(f'{conf:.2f}'))
    #             self.table.setItem(i, 2, QTableWidgetItem(str(x1)))
    #             self.table.setItem(i, 3, QTableWidgetItem(str(y1)))
    #             self.table.setItem(i, 4, QTableWidgetItem(str(x2)))
    #             self.table.setItem(i, 5, QTableWidgetItem(str(y2)))
    #     else:
    #         self.table.setRowCount(1)
    #         self.table.setItem(0, 0, QTableWidgetItem('无检测结果'))
    #         for j in range(1, 6):
    #             self.table.setItem(0, j, QTableWidgetItem(''))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = YOLODetectGUI()
    window.show()
    sys.exit(app.exec_())