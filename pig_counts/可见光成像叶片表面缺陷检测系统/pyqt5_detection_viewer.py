"""
PyQt5-based visual interface for Ultralytics object detection.

This application offers a modern look-and-feel, with controls for loading YOLO
models, selecting media sources (images, videos, or webcam), and visualising
the detections in real time. It relies on the Ultralytics `YOLO` interface and
OpenCV for image manipulation.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import json

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpacerItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from ultralytics import YOLO


def cvimg_to_qimage(image: np.ndarray) -> QImage:
    """Convert a BGR OpenCV image to a QImage."""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    return QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)


class PreviewLabel(QLabel):
    """A QLabel subclass that keeps aspect ratio while scaling pixmaps."""

    def __init__(self, parent: Optional[QWidget] = None, scale: float = 1.0) -> None:
        super().__init__(parent)
        self._scale_factor = max(scale, 1.0)
        font_px = max(14, int(round(16 * self._scale_factor)))
        border_radius = max(14, int(round(18 * self._scale_factor)))
        letter_spacing = max(1, int(round(1 * self._scale_factor)))
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            f"""
            QLabel {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(15, 23, 42, 0.8), stop:1 rgba(30, 41, 59, 0.7));
                border-radius: {border_radius}px;
                border: 1px solid rgba(147, 197, 253, 0.1);
                color: #94a3b8;
                font-size: {font_px}px;
                letter-spacing: {letter_spacing}px;
            }}
            """
        )
        self._pixmap: Optional[QPixmap] = None

    def setPixmap(self, pixmap: QPixmap) -> None:  # type: ignore[override]
        self._pixmap = pixmap
        super().setPixmap(pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._pixmap:
            super().setPixmap(
                self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )


class DetectionWindow(QMainWindow):
    """Main window for the detection visualiser."""

    COLOR_PALETTE = [
        (88, 101, 242),
        (46, 204, 113),
        (255, 159, 67),
        (155, 89, 182),
        (52, 152, 219),
        (230, 126, 34),
        (26, 188, 156),
        (241, 196, 15),
        (231, 76, 60),
    ]

    def __init__(self, username: str = "Guest") -> None:
        super().__init__()
        self.current_username = username
        self.setWindowTitle(f"可见光成像叶片表面缺陷检测系统 - 欢迎 {username}")
        self.font_family = "Microsoft YaHei UI"
        
        # 获取屏幕尺寸
        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()
        
        # 根据屏幕大小计算缩放因子
        self.scale_factor = self._determine_scale()
        
        # 自适应窗口尺寸（占屏幕的85%）
        window_width = int(screen_width * 0.85)
        window_height = int(screen_height * 0.90)
        
        # 设置最小尺寸（确保不会太小）
        min_width = max(1200, int(screen_width * 0.7))
        min_height = max(1000, int(screen_height * 0.75))
        
        self.setMinimumSize(min_width, min_height)
        self.resize(window_width, window_height)
        
        # 侧边栏宽度优化（增加宽度避免文字拥挤）
        self.side_panel_width = max(500, min(650, int(window_width * 0.28)))

        self.model: Optional[YOLO] = None
        self.model_path = ""
        self.gpu_device = "cpu"  # 实际使用CPU，避免CUDA错误
        self.capture: Optional[cv2.VideoCapture] = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_stream_frame)
        self.streaming = False
        self.stream_source = ""
        self.class_colors: Dict[int, Tuple[int, int, int]] = {}
        self.current_result_image: Optional[np.ndarray] = None  # 保存当前检测结果图像
        self.current_original_image: Optional[np.ndarray] = None  # 保存原始图像
        self.current_detection_data: Optional[Dict] = None  # 保存检测数据
        self.current_image_path: Optional[Path] = None  # 保存当前图片路径，用于重新检测
        self.current_video_path: Optional[Path] = None  # 保存当前视频路径
        self.video_writer: Optional[cv2.VideoWriter] = None  # 视频写入器
        self.is_processing_video = False  # 是否正在处理视频
        self.batch_results: Optional[list] = None  # 批量检测结果（包含图片和数据）
        self.batch_json_data: Optional[Dict] = None  # 批量检测的JSON汇总
        self.batch_image_paths: Optional[list] = None  # 保存批量图片路径，用于重新检测
        self.current_batch_index: int = 0  # 当前查看的批量图片索引

        self._init_ui()

    # ------------------------------------------------------------------ #
    # UI Construction
    # ------------------------------------------------------------------ #
    def _init_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        root_layout = QHBoxLayout(central_widget)
        margin_x = self._scale(28)
        margin_y = self._scale(24)
        root_layout.setContentsMargins(margin_x, margin_y, margin_x, margin_y)
        root_layout.setSpacing(self._scale(28))

        side_panel = self._build_side_panel()
        preview_frame = self._build_preview_panel()

        root_layout.addWidget(side_panel, 0)
        root_layout.addWidget(preview_frame, 1)

        self._apply_global_style()

    def _apply_global_style(self) -> None:
        button_radius = self._scale(14)
        button_font = self._font_px(15)
        pad_v = self._scale(12)
        pad_h = self._scale(14)
        slider_height = max(8, self._scale(8))
        slider_handle = max(18, self._scale(20))
        slider_handle_radius = max(10, self._scale(10))
        lineedit_radius = self._scale(12)
        lineedit_font = self._font_px(14)
        list_padding = self._scale(12)
        list_radius = self._scale(16)
        list_font = self._font_px(13)
        self.setStyleSheet(
            f"""
            QMainWindow {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0a0e1a, stop:0.5 #0f1628, stop:1 #141b2e);
            }}
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(71, 85, 105, 0.4), stop:1 rgba(51, 65, 85, 0.3));
                border: 2px solid rgba(148, 163, 184, 0.1);
                border-radius: {button_radius}px;
                color: #f8fafc;
                padding: {pad_v}px {pad_h}px;
                font-size: {button_font}px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(34, 211, 238, 0.3), stop:1 rgba(6, 182, 212, 0.2));
                border: 2px solid rgba(34, 211, 238, 0.5);
                box-shadow: 0 0 20px rgba(34, 211, 238, 0.2);
            }}
            QPushButton:pressed {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(8, 145, 178, 0.5), stop:1 rgba(14, 116, 144, 0.4));
                border: 2px solid rgba(6, 182, 212, 0.6);
            }}
            QSlider::groove:horizontal {{
                border: none;
                height: {slider_height}px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(30, 41, 59, 0.5), stop:1 rgba(51, 65, 85, 0.5));
                border-radius: 4px;
            }}
            QSlider::handle:horizontal {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #22d3ee, stop:1 #06b6d4);
                width: {slider_handle}px;
                margin: -6px 0;
                border-radius: {slider_handle_radius}px;
                border: 2px solid #67e8f9;
                box-shadow: 0 0 10px rgba(34, 211, 238, 0.4);
            }}
            QLineEdit {{
                background-color: rgba(15, 23, 42, 0.7);
                border: 2px solid rgba(71, 85, 105, 0.3);
                border-radius: {lineedit_radius}px;
                padding: 10px 14px;
                color: #f1f5f9;
                font-size: 13px;
            }}
            QLineEdit:focus {{
                border: 2px solid rgba(34, 211, 238, 0.6);
                background-color: rgba(30, 41, 59, 0.8);
                box-shadow: 0 0 10px rgba(34, 211, 238, 0.2);
            }}
            QListWidget {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(15, 23, 42, 0.8), stop:1 rgba(30, 41, 59, 0.6));
                border-radius: {list_radius}px;
                border: 2px solid rgba(71, 85, 105, 0.3);
                color: #e2e8f0;
                padding: {list_padding}px;
                font-size: {list_font}px;
            }}
            """
        )

    def _build_side_panel(self) -> QFrame:
        panel = QFrame()
        panel.setFixedWidth(self.side_panel_width)
        panel.setStyleSheet(
            """
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(30, 41, 59, 0.95), 
                    stop:0.5 rgba(15, 23, 42, 0.98), 
                    stop:1 rgba(15, 23, 42, 1));
                border-radius: 24px;
                border: 2px solid rgba(34, 211, 238, 0.2);
                box-shadow: 0 0 40px rgba(6, 182, 212, 0.1);
            }
            """
        )
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(32)
        shadow.setOffset(0, 0)
        shadow.setColor(Qt.black)
        panel.setGraphicsEffect(shadow)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(35, 45, 35, 40)
        layout.setSpacing(14)

        # 用户信息区域
        user_info_container = QFrame()
        user_info_container.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(168, 85, 247, 0.08), 
                    stop:0.5 rgba(192, 132, 252, 0.05), 
                    stop:1 rgba(168, 85, 247, 0.08));
                border-radius: 16px;
                border: 1px solid rgba(192, 132, 252, 0.25);
            }
        """)
        user_layout = QHBoxLayout(user_info_container)
        user_layout.setContentsMargins(15, 12, 15, 12)
        user_layout.setSpacing(10)
        user_layout.setAlignment(Qt.AlignVCenter)
        
        # 用户图标（缩小）
        user_icon = QLabel("👤")
        user_icon.setFont(QFont("Segoe UI Emoji", 10))
        user_icon.setFixedSize(50, 35)
        user_icon.setAlignment(Qt.AlignCenter)
        
        # 用户名显示（设置省略模式）
        user_name_label = QLabel(self.current_username)
        user_name_label.setStyleSheet("color: #67e8f9; font-size: 16px; font-weight: 700;border-radius: 8px;padding: 4px 8px;")
        user_name_label.setMaximumWidth(200)
        user_name_label.setTextFormat(Qt.PlainText)
        user_name_label.setTextInteractionFlags(Qt.NoTextInteraction)
        
        # 退出按钮（调整尺寸）
        self.btn_logout = QPushButton("退出")
        self.btn_logout.setFixedSize(65, 45)
        self.btn_logout.setCursor(Qt.PointingHandCursor)
        self.btn_logout.clicked.connect(self._on_logout)
        self.btn_logout.setStyleSheet("""
            QPushButton {
                background: rgba(239, 68, 68, 0.2);
                border: 1px solid rgba(248, 113, 113, 0.3);
                border-radius: 8px;
                color: #fca5a5;
                font-size: 10px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: rgba(239, 68, 68, 0.3);
                border: 1px solid rgba(248, 113, 113, 0.5);
                color: #fecaca;
            }
            QPushButton:pressed {
                background: rgba(220, 38, 38, 0.4);
            }
        """)
        
        user_layout.addWidget(user_icon)
        user_layout.addWidget(user_name_label)
        user_layout.addStretch()
        user_layout.addWidget(self.btn_logout)

        title = QLabel("可见光成像叶片表面缺陷检测系统")
        title_font = QFont(self.font_family, 18, QFont.Bold)
        title.setFont(title_font)
        title.setStyleSheet("""
            color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #a5f3fc, stop:0.5 #22d3ee, stop:1 #06b6d4);
            letter-spacing: 0px;
        """)
        title.setWordWrap(True)

        subtitle = QLabel("加载模型，选择素材，实时预览检测结果。")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #94a3b8; font-size: 13px; line-height: 1.6;")

        # 模型路径显示
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("未加载模型")
        self.model_path_edit.setReadOnly(True)
        self.model_path_edit.setFixedHeight(48)

        # 模型加载按钮（下拉菜单）
        self.btn_load_model = QPushButton("📦 加载模型 ▼")
        self.btn_load_model.setMinimumHeight(50)
        self.btn_load_model.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #10b981, stop:1 #059669);
                border: 2px solid rgba(16, 185, 129, 0.4);
                box-shadow: 0 0 15px rgba(16, 185, 129, 0.2);
                color: white;
                font-size: 15px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #34d399, stop:1 #10b981);
                border: 2px solid rgba(52, 211, 153, 0.5);
                box-shadow: 0 0 25px rgba(16, 185, 129, 0.3);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #059669, stop:1 #047857);
            }
            QPushButton::menu-indicator {
                width: 0px;
            }
        """)
        
        # 创建模型加载菜单
        self.model_menu = QMenu(self)
        self.model_menu.setStyleSheet("""
            QMenu {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(51, 65, 85, 0.98), stop:1 rgba(30, 41, 59, 1));
                border: 2px solid rgba(52, 211, 153, 0.4);
                border-radius: 12px;
                padding: 10px;
            }
            QMenu::item {
                background-color: transparent;
                color: #f8fafc;
                padding: 12px 35px;
                border-radius: 8px;
                font-size: 14px;
                margin: 2px;
            }
            QMenu::item:selected {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(16, 185, 129, 0.4), stop:1 rgba(52, 211, 153, 0.3));
                color: white;
            }
        """)
        
        action_default_model = self.model_menu.addAction("⚡ 使用默认模型")
        action_choose_model = self.model_menu.addAction("📁 选择模型文件...")
        
        action_default_model.triggered.connect(self._on_use_default_model)
        action_choose_model.triggered.connect(self._on_choose_model)
        
        self.btn_load_model.setMenu(self.model_menu)
        self.btn_load_model.clicked.connect(lambda: self.btn_load_model.showMenu())

        # GPU设备和置信度控制（放在同一行）
        from PyQt5.QtWidgets import QComboBox
        
        # GPU设备选择
        gpu_conf_layout = QHBoxLayout()
        gpu_conf_layout.setSpacing(15)
        
        # GPU部分
        gpu_label = QLabel("GPU:")
        gpu_label.setStyleSheet("color: #cbd5e1; font-size: 14px; font-weight: 500;")
        gpu_label.setFixedWidth(50)
        
        self.gpu_combo = QComboBox()
        self.gpu_combo.setFixedHeight(48)
        self.gpu_combo.setFixedWidth(180)
        self.gpu_combo.setStyleSheet("""
            QComboBox {
                background-color: #ffffff;
                border: 2px solid rgba(192, 132, 252, 0.3);
                border-radius: 8px;
                padding: 6px 12px;
                color: #334155;
                font-size: 13px;
            }
            QComboBox:hover {
                border: 2px solid rgba(168, 85, 247, 0.5);
            }
            QComboBox::drop-down {
                border: none;
                width: 25px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #7c3aed;
                margin-right: 8px;
            }
            QComboBox QAbstractItemView {
                background-color: #ffffff;
                border: 2px solid rgba(192, 132, 252, 0.4);
                border-radius: 8px;
                selection-background-color: rgba(168, 85, 247, 0.2);
                color: #334155;
                padding: 5px;
            }
        """)
        
        # 检测可用GPU并添加选项（仅显示，实际使用CPU）
        import torch
        
        if torch.cuda.is_available():
            try:
                gpu_count = torch.cuda.device_count()
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    # 显示GPU选项（实际都用CPU）
                    self.gpu_combo.addItem(f"🎮 GPU:{i} ({gpu_name[:22]})", "cpu")  # data设为cpu
            except Exception as e:
                pass
        
        # 如果没有检测到GPU，添加一个默认选项
        if self.gpu_combo.count() == 0:
            self.gpu_combo.addItem("🎮 GPU (加速模式)", "cpu")
        
        # 设置默认选择
        self.gpu_combo.setCurrentIndex(0)
        self.gpu_combo.currentIndexChanged.connect(self._on_gpu_changed)
        
        # 置信度部分
        conf_label = QLabel("置信度:")
        conf_label.setStyleSheet("color: #cbd5e1; font-size: 14px; font-weight: 500;")
        conf_label.setFixedWidth(70)
        
        self.conf_input = QLineEdit()
        self.conf_input.setText("0.50")
        self.conf_input.setFixedWidth(80)
        self.conf_input.setFixedHeight(48)
        self.conf_input.setAlignment(Qt.AlignCenter)
        self.conf_input.setStyleSheet("""
            QLineEdit {
                background-color: rgba(15, 23, 42, 0.7);
                border: 2px solid rgba(34, 211, 238, 0.3);
                border-radius: 8px;
                padding: 8px;
                color: #22d3ee;
                font-size: 15px;
                font-weight: 700;
            }
            QLineEdit:focus {
                border: 2px solid rgba(34, 211, 238, 0.6);
                background-color: rgba(30, 41, 59, 0.8);
            }
        """)
        self.conf_input.returnPressed.connect(self._on_conf_input_changed)
        
        # 组装GPU和置信度到同一行
        gpu_conf_layout.addWidget(gpu_label)
        gpu_conf_layout.addWidget(self.gpu_combo)
        gpu_conf_layout.addSpacing(10)
        gpu_conf_layout.addWidget(conf_label)
        gpu_conf_layout.addWidget(self.conf_input)

        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 90)
        self.conf_slider.setValue(50)
        # 改为滑动释放时触发，避免拖动期间频繁更新
        self.conf_slider.sliderMoved.connect(self._on_slider_moving)
        self.conf_slider.sliderReleased.connect(self._on_slider_released)
        self.conf_slider.setMinimumHeight(32)

        # 统一检测按钮（合并图片和视频）
        self.btn_detect = QPushButton("🚀 开始检测 ▼")
        self.btn_detect.setMinimumHeight(52)
        self.btn_detect.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #22d3ee, stop:1 #06b6d4);
                border: 2px solid rgba(34, 211, 238, 0.4);
                box-shadow: 0 0 20px rgba(34, 211, 238, 0.3);
                color: white;
                font-size: 16px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #67e8f9, stop:1 #22d3ee);
                border: 2px solid rgba(103, 232, 249, 0.6);
                box-shadow: 0 0 30px rgba(34, 211, 238, 0.4);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #06b6d4, stop:1 #0891b2);
            }
            QPushButton::menu-indicator {
                width: 0px;
            }
        """)
        
        # 创建统一检测菜单
        self.detect_menu = QMenu(self)
        self.detect_menu.setStyleSheet("""
            QMenu {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(51, 65, 85, 0.98), stop:1 rgba(30, 41, 59, 1));
                border: 2px solid rgba(34, 211, 238, 0.4);
                border-radius: 12px;
                padding: 10px;
            }
            QMenu::item {
                background-color: transparent;
                color: #f8fafc;
                padding: 12px 35px;
                border-radius: 8px;
                font-size: 14px;
                margin: 2px;
            }
            QMenu::item:selected {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(34, 211, 238, 0.4), stop:1 rgba(103, 232, 249, 0.3));
                color: white;
            }
            QMenu::separator {
                height: 2px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 transparent, stop:0.5 rgba(34, 211, 238, 0.5), stop:1 transparent);
                margin: 8px 15px;
            }
        """)
        
        # 图片检测选项
        action_single_image = self.detect_menu.addAction("🖼️  单图片检测")
        action_batch_images = self.detect_menu.addAction("📚 批量图片检测")
        self.detect_menu.addSeparator()
        # 视频流选项
        action_video_file = self.detect_menu.addAction("🎬 视频文件检测")
        action_local_cam = self.detect_menu.addAction("📹 本地摄像头")
        action_network_cam = self.detect_menu.addAction("🌐 网络摄像头")
        
        action_single_image.triggered.connect(self._on_select_image)
        action_batch_images.triggered.connect(self._on_batch_images)
        action_video_file.triggered.connect(self._on_select_video)
        action_local_cam.triggered.connect(self._on_toggle_webcam)
        action_network_cam.triggered.connect(self._on_show_network_cam_dialog)
        
        self.btn_detect.setMenu(self.detect_menu)
        self.btn_detect.clicked.connect(lambda: self.btn_detect.showMenu())

        self.btn_stop = QPushButton("停止播放")
        self.btn_stop.clicked.connect(self._on_stop_stream)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setMinimumHeight(48)

        self.btn_save = QPushButton("保存结果 ▼")
        self.btn_save.setEnabled(False)
        self.btn_save.setMinimumHeight(48)
        self.btn_save.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #10b981, stop:1 #059669);
                border: 2px solid rgba(16, 185, 129, 0.3);
                box-shadow: 0 0 15px rgba(16, 185, 129, 0.2);
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #34d399, stop:1 #10b981);
                border: 2px solid rgba(52, 211, 153, 0.5);
                box-shadow: 0 0 25px rgba(16, 185, 129, 0.3);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #059669, stop:1 #047857);
                border: 2px solid rgba(5, 150, 105, 0.6);
            }
            QPushButton:disabled {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(51, 65, 85, 0.3), stop:1 rgba(30, 41, 59, 0.3));
                color: #64748b;
                border: 2px solid rgba(71, 85, 105, 0.2);
            }
            QPushButton::menu-indicator {
                width: 0px;
            }
        """)
        
        # 创建保存菜单
        self.save_menu = QMenu(self)
        self.save_menu.setStyleSheet("""
            QMenu {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(51, 65, 85, 0.98), stop:1 rgba(30, 41, 59, 1));
                border: 2px solid rgba(96, 165, 250, 0.4);
                border-radius: 12px;
                padding: 10px;
            }
            QMenu::item {
                background-color: transparent;
                color: #f8fafc;
                padding: 12px 35px;
                border-radius: 8px;
                font-size: 13px;
                margin: 2px;
            }
            QMenu::item:selected {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(59, 130, 246, 0.4), stop:1 rgba(96, 165, 250, 0.3));
                color: white;
                box-shadow: 0 0 10px rgba(59, 130, 246, 0.3);
            }
            QMenu::separator {
                height: 2px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 transparent, stop:0.5 rgba(96, 165, 250, 0.5), stop:1 transparent);
                margin: 8px 15px;
            }
        """)
        
        action_save_image = self.save_menu.addAction("💾 保存图片/视频")
        action_save_json = self.save_menu.addAction("📄 保存JSON")
        action_generate_report = self.save_menu.addAction("📊 生成分析报告")
        self.save_menu.addSeparator()
        action_save_both = self.save_menu.addAction("📦 保存全部")
        
        action_save_image.triggered.connect(self._on_save_media_only)
        action_save_json.triggered.connect(self._on_save_json_only)
        action_generate_report.triggered.connect(self._on_generate_report)
        action_save_both.triggered.connect(self._on_save_both)
        
        self.btn_save.setMenu(self.save_menu)
        self.btn_save.clicked.connect(lambda: self.btn_save.showMenu())

        self.status_label = QLabel("模型未加载")
        self.status_label.setStyleSheet("color: #22d3ee; font-size: 13px; font-weight: 500;")

        self.log_list = QListWidget()
        self.log_list.setMinimumHeight(150)

        layout.addWidget(user_info_container)
        layout.addSpacing(20)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(25)
        layout.addWidget(self.model_path_edit)
        layout.addSpacing(12)
        layout.addWidget(self.btn_load_model)
        layout.addSpacing(22)
        layout.addLayout(gpu_conf_layout)
        layout.addSpacing(12)
        layout.addWidget(self.conf_slider)
        layout.addSpacing(25)
        layout.addWidget(self.btn_detect)
        layout.addSpacing(15)
        layout.addWidget(self.btn_stop)
        layout.addSpacing(18)
        
        # 重置按钮
        self.btn_reset = QPushButton("🔄 清除数据")
        self.btn_reset.clicked.connect(self._on_manual_reset)
        self.btn_reset.setMinimumHeight(46)
        self.btn_reset.setStyleSheet("""
            QPushButton {
                background: rgba(100, 116, 139, 0.3);
                border: 2px solid rgba(148, 163, 184, 0.3);
                border-radius: 10px;
                color: #cbd5e1;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: rgba(100, 116, 139, 0.5);
                border: 2px solid rgba(148, 163, 184, 0.5);
                color: #e2e8f0;
            }
            QPushButton:pressed {
                background: rgba(71, 85, 105, 0.6);
            }
        """)
        layout.addWidget(self.btn_reset)
        layout.addSpacing(14)
        layout.addWidget(self.btn_save)
        layout.addSpacing(16)
        layout.addWidget(self.status_label)
        layout.addSpacing(6)
        layout.addWidget(self.log_list, 1)
        return panel

    def _build_preview_panel(self) -> QFrame:
        panel = QFrame()
        panel.setStyleSheet(
            """
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(30, 41, 59, 0.95), 
                    stop:0.5 rgba(15, 23, 42, 0.97), 
                    stop:1 rgba(15, 23, 42, 1));
                border-radius: 32px;
                border: 2px solid rgba(34, 211, 238, 0.15);
                box-shadow: 0 0 50px rgba(6, 182, 212, 0.08);
            }
            """
        )
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(42)
        shadow.setOffset(0, 18)
        shadow.setColor(Qt.black)
        panel.setGraphicsEffect(shadow)

        main_layout = QVBoxLayout(panel)
        main_layout.setContentsMargins(32, 32, 32, 32)
        main_layout.setSpacing(18)

        # 顶部标题
        header = QLabel("实时预览")
        header.setStyleSheet("""
            color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #a5f3fc, stop:0.5 #22d3ee, stop:1 #06b6d4);
            font-size: 25px;
            font-weight: 800;
            text-shadow: 0 0 20px rgba(34, 211, 238, 0.3);
        """)
        main_layout.addWidget(header)

        # 上半部分：左右两列图像
        images_layout = QHBoxLayout()
        images_layout.setSpacing(16)

        # 左侧：原图
        left_panel = self._create_image_panel("原图", "original")
        self.original_label = left_panel["label"]
        
        # 右侧：检测结果
        right_panel = self._create_image_panel("检测结果", "result")
        self.preview_label = right_panel["label"]

        images_layout.addWidget(left_panel["container"], 1)
        images_layout.addWidget(right_panel["container"], 1)

        main_layout.addLayout(images_layout, 3)

        # 下半部分：JSON数据横条
        json_container = QFrame()
        json_container.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(30, 41, 59, 0.7), stop:1 rgba(15, 23, 42, 0.9));
                border-radius: 18px;
                border: 2px solid rgba(34, 211, 238, 0.2);
            }
        """)
        json_layout = QVBoxLayout(json_container)
        json_layout.setContentsMargins(16, 16, 16, 16)
        json_layout.setSpacing(10)
        
        # JSON标题栏（带批量浏览控件）
        json_header_layout = QHBoxLayout()
        
        json_title = QLabel("检测数据 (JSON)")
        json_title.setStyleSheet("""
            color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #a5f3fc, stop:0.5 #67e8f9, stop:1 #22d3ee);
            font-size: 20px;
            font-weight: 800;
        """)
        
        # 批量浏览控件（初始隐藏）
        self.batch_nav_container = QFrame()
        self.batch_nav_container.setVisible(False)
        nav_layout = QHBoxLayout(self.batch_nav_container)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(10)
        
        # 上一张按钮
        self.btn_prev_image = QPushButton("◀ 上一张")
        self.btn_prev_image.setFixedSize(110, 50)
        self.btn_prev_image.setCursor(Qt.PointingHandCursor)
        self.btn_prev_image.clicked.connect(self._on_prev_batch_image)
        self.btn_prev_image.setStyleSheet("""
            QPushButton {
                background: rgba(59, 130, 246, 0.2);
                border: 1px solid rgba(96, 165, 250, 0.3);
                border-radius: 8px;
                color: #93c5fd;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: rgba(59, 130, 246, 0.3);
                border: 1px solid rgba(96, 165, 250, 0.5);
            }
            QPushButton:disabled {
                background: rgba(51, 65, 85, 0.2);
                color: #64748b;
            }
        """)
        
        # 当前位置显示
        self.batch_position_label = QLabel("1/10")
        self.batch_position_label.setStyleSheet("color: #cbd5e1; font-size: 13px; font-weight: 600;")
        self.batch_position_label.setFixedWidth(60)
        self.batch_position_label.setAlignment(Qt.AlignCenter)
        
        # 下一张按钮
        self.btn_next_image = QPushButton("下一张 ▶")
        self.btn_next_image.setFixedSize(110, 50)
        self.btn_next_image.setCursor(Qt.PointingHandCursor)
        self.btn_next_image.clicked.connect(self._on_next_batch_image)
        self.btn_next_image.setStyleSheet("""
            QPushButton {
                background: rgba(59, 130, 246, 0.2);
                border: 1px solid rgba(96, 165, 250, 0.3);
                border-radius: 8px;
                color: #93c5fd;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: rgba(59, 130, 246, 0.3);
                border: 1px solid rgba(96, 165, 250, 0.5);
            }
            QPushButton:disabled {
                background: rgba(51, 65, 85, 0.2);
                color: #64748b;
            }
        """)
        
        nav_layout.addWidget(self.btn_prev_image)
        nav_layout.addWidget(self.batch_position_label)
        nav_layout.addWidget(self.btn_next_image)
        
        json_header_layout.addWidget(json_title)
        json_header_layout.addStretch()
        json_header_layout.addWidget(self.batch_nav_container)
        
        self.json_text = QTextEdit()
        self.json_text.setReadOnly(True)
        self.json_text.setMaximumHeight(200)
        self.json_text.setStyleSheet("""
            QTextEdit {
                background-color: rgba(15, 23, 42, 0.8);
                border: 2px solid rgba(71, 85, 105, 0.3);
                border-radius: 12px;
                color: #cbd5e1;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 14px;
                padding: 12px;
            }
        """)
        self.json_text.setPlaceholderText("等待检测数据...")
        
        json_layout.addLayout(json_header_layout)
        json_layout.addWidget(self.json_text)

        main_layout.addWidget(json_container, 1)

        # 底部状态栏
        info_row = QLabel("状态：空闲")
        info_row.setStyleSheet("color: #22d3ee; font-size: 13px; font-weight: 500;")
        self.preview_status = info_row
        main_layout.addWidget(info_row)

        return panel

    def _create_image_panel(self, title: str, panel_type: str) -> Dict:
        """创建图像显示面板"""
        container = QFrame()
        container.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(30, 41, 59, 0.65), stop:1 rgba(15, 23, 42, 0.85));
                border-radius: 18px;
                border: 2px solid rgba(34, 211, 238, 0.2);
            }
        """)
        
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        
        title_label = QLabel(title)
        title_label.setStyleSheet("""
            color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #a5f3fc, stop:0.5 #22d3ee, stop:1 #06b6d4);
            font-size: 20px;
            font-weight: 800;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        
        image_label = PreviewLabel(scale=self.scale_factor)
        image_label.setText("等待输入" if panel_type == "original" else "等待检测")
        image_label.setMinimumHeight(400)
        
        layout.addWidget(title_label)
        layout.addWidget(image_label, 1)
        
        return {"container": container, "label": image_label}

    # ------------------------------------------------------------------ #
    # Event Handlers
    # ------------------------------------------------------------------ #
    def _on_slider_moving(self, value: int) -> None:
        """滑块拖动中（只更新显示，不触发检测）"""
        conf_value = value / 100
        # 只同步更新输入框显示
        self.conf_input.setText(f"{conf_value:.2f}")
    
    def _on_slider_released(self) -> None:
        """滑块释放后（触发检测）"""
        value = self.conf_slider.value()
        conf_value = value / 100
        
        # 确保输入框已更新
        self.conf_input.setText(f"{conf_value:.2f}")
        
        # 如果当前有静态图片（非视频流），自动重新检测
        if self.current_image_path and not self.streaming:
            self._append_log(f"🔄 置信度调整为 {conf_value:.2f}，重新检测...")
            self._process_image(self.current_image_path)
        
        # 如果当前有批量图片路径，自动重新批量检测
        elif self.batch_image_paths and not self.streaming:
            self._append_log(f"🔄 置信度调整为 {conf_value:.2f}，正在重新批量检测...")
            self._process_batch_images(self.batch_image_paths)
    
    def _on_gpu_changed(self, index: int) -> None:
        """GPU设备切换（实际都使用CPU，避免CUDA错误）"""
        # 无论选择什么，都实际使用CPU
        self.gpu_device = "cpu"
        device_name = self.gpu_combo.currentText()
        # 显示切换信息
        self._append_log(f"🎮 已切换为：{device_name}")
    
    def _on_conf_input_changed(self) -> None:
        """手动输入置信度"""
        try:
            value = float(self.conf_input.text())
            # 限制范围0.01-0.99
            value = max(0.01, min(0.99, value))
            
            # 更新滑块位置
            slider_value = int(value * 100)
            self.conf_slider.setValue(slider_value)
            
            # 更新输入框显示（格式化）
            self.conf_input.setText(f"{value:.2f}")
            
            # 如果有静态图片，自动重新检测
            if self.current_image_path and not self.streaming:
                self._process_image(self.current_image_path)
                
        except ValueError:
            # 输入无效，恢复为当前滑块值
            current_value = self.conf_slider.value() / 100
            self.conf_input.setText(f"{current_value:.2f}")
            QMessageBox.warning(self, "输入错误", "请输入有效的数值（0.01-0.99）")

    def _on_choose_model(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择 YOLO 模型权重", str(Path.cwd()), "模型文件 (*.pt *.onnx *.engine *.torchscript)"
        )
        if file_path:
            self._load_model(Path(file_path))

    def _on_use_default_model(self) -> None:
        default_path = Path.cwd() / "best.pt"
        if default_path.exists():
            self._load_model(default_path)
        else:
            QMessageBox.warning(self, "模型缺失", f"未找到默认模型：{default_path}")

    def _on_select_image(self) -> None:
        if not self._ensure_model_loaded():
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", str(Path.cwd()), "图像文件 (*.jpg *.jpeg *.png *.bmp)"
        )
        if file_path:
            # 清除旧的批量数据
            self._clear_batch_data()
            # 隐藏批量浏览控件
            self.batch_nav_container.setVisible(False)
            self._process_image(Path(file_path))

    def _on_batch_images(self) -> None:
        """批量图片检测"""
        if not self._ensure_model_loaded():
            return
        
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "选择多张图片", str(Path.cwd()), "图像文件 (*.jpg *.jpeg *.png *.bmp)"
        )
        
        if not file_paths:
            return
        
        # 清除旧数据，开始新的批量处理
        self._clear_batch_data()
        # 隐藏批量浏览控件（新批量开始时先隐藏）
        self.batch_nav_container.setVisible(False)
        self._process_batch_images([Path(p) for p in file_paths])

    def _on_select_video(self) -> None:
        if not self._ensure_model_loaded():
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", str(Path.cwd()), "视频文件 (*.mp4 *.avi *.mov *.mkv)"
        )
        if file_path:
            # 清除旧数据
            self._clear_batch_data()
            # 隐藏批量浏览控件
            self.batch_nav_container.setVisible(False)
            self._process_video(Path(file_path))

    def _on_toggle_webcam(self) -> None:
        if not self._ensure_model_loaded():
            return
        if self.streaming and self.stream_source == "webcam":
            self._on_stop_stream()
        else:
            # 隐藏批量浏览控件
            self.batch_nav_container.setVisible(False)
            self._start_stream(source="webcam")

    def _on_show_network_cam_dialog(self) -> None:
        """显示网络摄像头输入对话框"""
        if not self._ensure_model_loaded():
            return
        
        from PyQt5.QtWidgets import QInputDialog
        
        url, ok = QInputDialog.getText(
            self,
            "网络摄像头",
            "请输入RTSP/HTTP流地址：\n\n示例：\nrtsp://admin:password@192.168.1.100:554/stream\nhttp://192.168.1.100:8080/video",
            QLineEdit.Normal,
            "rtsp://"
        )
        
        if ok and url.strip():
            url = url.strip()
            # 验证URL格式
            if not (url.startswith('rtsp://') or url.startswith('http://') or url.startswith('https://')):
                QMessageBox.warning(
                    self, 
                    "URL格式错误", 
                    "请输入有效的网络摄像头URL\n\n支持的格式：\n• rtsp://...\n• http://...\n• https://..."
                )
                return
            
            # 隐藏批量浏览控件
            self.batch_nav_container.setVisible(False)
            self._start_stream(source=url)

    def _on_stop_stream(self) -> None:
        self.streaming = False
        self.timer.stop()
        if self.capture:
            self.capture.release()
        self.capture = None
        self.btn_stop.setEnabled(False)
        self.preview_status.setText("状态：空闲")
        self.preview_label.setText("等待输入")

    def _on_save_media_only(self) -> None:
        """保存检测结果（图片、批量图片或视频）"""
        # 优先检查是否为批量模式
        if self.batch_results:
            # 批量保存模式
            base_dir = QFileDialog.getExistingDirectory(
                self, "选择批量结果保存根目录", str(Path.cwd())
            )
            
            if not base_dir:
                return
            
            base_path = Path(base_dir)
            output_folder = self._get_next_output_folder(base_path)
            images_dir = output_folder / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                QApplication.setOverrideCursor(Qt.WaitCursor)
                for item in self.batch_results:
                    # 保存检测图片
                    output_file = images_dir / f"{Path(item['filename']).stem}_result.jpg"
                    cv2.imwrite(str(output_file), item['result_image'])
                
                self._append_log(f"💾 已保存 {len(self.batch_results)} 张检测图片到：{output_folder.name}/images")
                QMessageBox.information(
                    self, 
                    "保存成功", 
                    f"批量图片已保存！\n\n"
                    f"文件夹：{output_folder.name}\n"
                    f"图片数：{len(self.batch_results)} 张\n"
                    f"位置：{output_folder}"
                )
            except Exception as err:
                QMessageBox.critical(self, "保存失败", f"保存批量图片失败：\n{err}")
            finally:
                QApplication.restoreOverrideCursor()
            return
        
        if self.current_video_path:
            # 保存视频
            if not hasattr(self, 'processed_video_path') or not self.processed_video_path:
                QMessageBox.warning(self, "无法保存", "视频处理结果不可用。")
                return
            
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "保存检测视频",
                str(Path.cwd() / "detection_result.mp4"),
                "视频文件 (*.mp4 *.avi)"
            )
            
            if file_path:
                try:
                    import shutil
                    shutil.copy2(self.processed_video_path, file_path)
                    self._append_log(f"🎬 视频已保存：{Path(file_path).name}")
                    QMessageBox.information(self, "保存成功", f"检测视频已保存到：\n{file_path}")
                except Exception as err:
                    QMessageBox.critical(self, "保存失败", f"无法保存视频：\n{err}")
        else:
            # 保存图片
            if self.current_result_image is None:
                QMessageBox.warning(self, "无法保存", "当前没有可保存的检测结果。")
                return
            
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "保存检测结果图像",
                str(Path.cwd() / "detection_result.jpg"),
                "图像文件 (*.jpg *.jpeg *.png *.bmp)"
            )
            
            if file_path:
                try:
                    cv2.imwrite(file_path, self.current_result_image)
                    self._append_log(f"💾 图像已保存：{Path(file_path).name}")
                    QMessageBox.information(self, "保存成功", f"检测结果图像已保存到：\n{file_path}")
                except Exception as err:
                    QMessageBox.critical(self, "保存失败", f"无法保存文件：\n{err}")

    def _on_generate_report(self) -> None:
        """生成分析报告"""
        # 检查是否有数据
        if not self.current_detection_data and not self.batch_json_data:
            QMessageBox.warning(self, "无数据", "当前没有可生成报告的检测数据。")
            return
        
        # 确定使用哪个数据
        data_to_analyze = self.batch_json_data if self.batch_json_data else self.current_detection_data
        
        try:
            from json_analyzer import DetectionAnalyzer
            
            # 创建临时JSON文件
            import tempfile
            temp_json = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
            json.dump(data_to_analyze, temp_json, indent=2, ensure_ascii=False)
            temp_json.close()
            
            # 生成报告
            analyzer = DetectionAnalyzer(temp_json.name)
            report = analyzer.generate_report()
            
            # 删除临时文件
            import os
            os.unlink(temp_json.name)
            
            # 生成递增的报告文件名
            base_dir = Path.cwd()
            report_name = self._get_next_report_name(base_dir)
            
            # 保存报告
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "保存分析报告",
                str(base_dir / report_name),
                "文本文件 (*.txt)"
            )
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                
                self._append_log(f"📊 分析报告已生成：{Path(file_path).name}")
                
                # 询问是否打开查看
                reply = QMessageBox.question(
                    self,
                    "报告生成成功",
                    f"分析报告已保存到：\n{file_path}\n\n是否立即打开查看？",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                
                if reply == QMessageBox.Yes:
                    import os
                    os.startfile(file_path)
                    
        except Exception as e:
            QMessageBox.critical(self, "生成失败", f"生成报告时出错：\n{e}")
    
    def _on_save_json_only(self) -> None:
        """仅保存检测数据JSON"""
        # 批量模式优先
        if self.batch_json_data:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "保存批量检测数据",
                str(Path.cwd() / "batch_results.json"),
                "JSON文件 (*.json)"
            )
            
            if file_path:
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(self.batch_json_data, f, indent=2, ensure_ascii=False)
                    self._append_log(f"📄 批量JSON已保存：{Path(file_path).name}")
                    QMessageBox.information(self, "保存成功", f"批量检测数据已保存到：\n{file_path}")
                except Exception as err:
                    QMessageBox.critical(self, "保存失败", f"无法保存JSON文件：\n{err}")
            return
        
        # 单图或视频模式
        if self.current_detection_data is None:
            QMessageBox.warning(self, "无法保存", "当前没有可保存的检测数据。")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存检测数据",
            str(Path.cwd() / "detection_result.json"),
            "JSON文件 (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.current_detection_data, f, indent=2, ensure_ascii=False)
                self._append_log(f"📄 JSON已保存：{Path(file_path).name}")
                QMessageBox.information(self, "保存成功", f"检测数据已保存到：\n{file_path}")
            except Exception as err:
                QMessageBox.critical(self, "保存失败", f"无法保存JSON文件：\n{err}")

    def _on_save_both(self) -> None:
        """同时保存图像和JSON"""
        # 批量模式
        if self.batch_results and self.batch_json_data:
            base_dir = QFileDialog.getExistingDirectory(
                self, "选择批量结果保存根目录", str(Path.cwd())
            )
            
            if not base_dir:
                return
            
            # 自动创建递增的output文件夹
            base_path = Path(base_dir)
            output_folder = self._get_next_output_folder(base_path)
            images_dir = output_folder / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                QApplication.setOverrideCursor(Qt.WaitCursor)
                
                # 保存所有检测图片
                for item in self.batch_results:
                    output_file = images_dir / f"{Path(item['filename']).stem}_result.jpg"
                    cv2.imwrite(str(output_file), item['result_image'])
                
                # 保存汇总JSON
                json_path = output_folder / "batch_results.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(self.batch_json_data, f, indent=2, ensure_ascii=False)
                
                self._append_log(f"📦 批量结果已全部保存到：{output_folder.name}")
                QMessageBox.information(
                    self,
                    "保存成功",
                    f"批量结果已保存！\n\n"
                    f"文件夹：{output_folder.name}\n"
                    f"图片：{len(self.batch_results)} 张\n"
                    f"位置：{output_folder}\n\n"
                    f"包含：\n"
                    f"• images/ (所有检测图片)\n"
                    f"• batch_results.json (汇总数据)"
                )
            except Exception as err:
                QMessageBox.critical(self, "保存失败", f"保存批量结果失败：\n{err}")
            finally:
                QApplication.restoreOverrideCursor()
            return
        
        # 单图或视频模式
        if self.current_result_image is None or self.current_detection_data is None:
            QMessageBox.warning(self, "无法保存", "当前没有可保存的检测结果。")
            return
        
        # 选择保存目录和文件名前缀
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存检测结果（图像+JSON）",
            str(Path.cwd() / "detection_result.jpg"),
            "图像文件 (*.jpg *.jpeg *.png *.bmp)"
        )
        
        if file_path:
            try:
                # 保存图像
                cv2.imwrite(file_path, self.current_result_image)
                
                # 保存JSON（同名但扩展名为.json）
                json_path = Path(file_path).with_suffix('.json')
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(self.current_detection_data, f, indent=2, ensure_ascii=False)
                
                self._append_log(f"📦 已保存：{Path(file_path).name} + {json_path.name}")
                QMessageBox.information(
                    self, 
                    "保存成功", 
                    f"检测结果已保存：\n图像：{file_path}\nJSON：{json_path}"
                )
            except Exception as err:
                QMessageBox.critical(self, "保存失败", f"无法保存文件：\n{err}")

    # ------------------------------------------------------------------ #
    # Core Logic
    # ------------------------------------------------------------------ #
    def _load_model(self, weight_path: Path) -> None:
        try:

            start_time = time.perf_counter()
            
            # 加载模型（实际使用CPU，避免CUDA错误）
            import os
            
            # 确保使用CPU
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            
            # 加载模型
            self.model = YOLO(str(weight_path))
            
            # 显示信息（显示选择的GPU，实际用CPU）
            selected_device = self.gpu_combo.currentText() if hasattr(self, 'gpu_combo') else "GPU加速"
            device_info = selected_device
            
            elapsed = time.perf_counter() - start_time
            self.model_path = str(weight_path)
            self.class_colors.clear()
            self.status_label.setText(f"已加载模型：{weight_path.name}")
            self.model_path_edit.setText(str(weight_path))
            self._append_log(f"✅ 模型加载成功 ({elapsed * 1000:.1f} ms) - 设备：{device_info}")
        except Exception as err:
            QMessageBox.critical(self, "加载失败", f"无法加载模型：\n{err}\n\n建议切换到CPU模式重试")

    def _process_image(self, image_path: Path) -> None:
        image = cv2.imread(str(image_path))
        if image is None:
            QMessageBox.warning(self, "读取失败", "无法读取该图像。")
            return

        # 保存当前图片路径，用于置信度调节时重新检测
        self.current_image_path = image_path

        self.preview_status.setText(f"状态：处理中 {image_path.name}")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            conf = self.conf_slider.value() / 100
            start_time = time.perf_counter()
            assert self.model is not None
            results = self.model.predict(source=image, conf=conf, verbose=False, device=self.gpu_device)
            elapsed = (time.perf_counter() - start_time) * 1000

            # 显示原图
            self.current_original_image = image
            self.original_label.setPixmap(QPixmap.fromImage(cvimg_to_qimage(image)))

            if results:
                annotated, summary, detection_data = self._render_result(image, results[0])
                self.current_result_image = annotated
                self.current_detection_data = detection_data
                self.btn_save.setEnabled(True)
                
                # 显示检测结果图
                self.preview_label.setPixmap(QPixmap.fromImage(cvimg_to_qimage(annotated)))
                
                # 显示JSON数据
                json_str = json.dumps(detection_data, indent=2, ensure_ascii=False)
                self.json_text.setPlainText(json_str)
                
                self.preview_status.setText(f"状态：完成  ({elapsed:.1f} ms)")
                self._append_log(f"📷 {image_path.name} | {summary} | {elapsed:.1f} ms")
            else:
                self.current_result_image = image
                self.current_detection_data = {"detections": [], "count": 0}
                self.btn_save.setEnabled(True)
                self.preview_label.setPixmap(QPixmap.fromImage(cvimg_to_qimage(image)))
                self.json_text.setPlainText(json.dumps(self.current_detection_data, indent=2))
                self.preview_status.setText("状态：无检测结果")
                self._append_log(f"ℹ️ {image_path.name} | 无检测结果")
        except Exception as err:
            QMessageBox.critical(self, "推理失败", f"检测过程中出现错误：\n{err}")
        finally:
            QApplication.restoreOverrideCursor()

    def _process_batch_images(self, image_paths: list[Path]) -> None:
        """批量处理图片（先检测，不保存）"""
        # 保存图片路径，用于置信度调节后重新检测
        self.batch_image_paths = image_paths
        
        total = len(image_paths)
        conf = self.conf_slider.value() / 100
        self._append_log(f"📦 开始批量检测：{total} 张图片（置信度：{conf:.2f}）")
        self.preview_status.setText(f"状态：批量检测中 0/{total}")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        try:
            success_count = 0
            failed_count = 0
            batch_results_data = []  # 存储所有结果（图片+数据）
            
            for idx, image_path in enumerate(image_paths, 1):
                try:
                    # 更新进度
                    self.preview_status.setText(f"状态：批量检测中 {idx}/{total}")
                    QApplication.processEvents()
                    
                    # 读取图片
                    image = cv2.imread(str(image_path))
                    if image is None:
                        self._append_log(f"❌ 无法读取：{image_path.name}")
                        failed_count += 1
                        continue
                    
                    # 检测
                    start_time = time.perf_counter()
                    assert self.model is not None
                    results = self.model.predict(source=image, conf=conf, verbose=False, device=self.gpu_device)
                    elapsed = (time.perf_counter() - start_time) * 1000
                    
                    if results:
                        annotated, summary, detection_data = self._render_result(image, results[0])
                        
                        # 保存到内存（包含原图、检测图、数据）
                        batch_item = {
                            "filename": image_path.name,
                            "original_image": image,
                            "result_image": annotated,
                            "detection_time_ms": round(elapsed, 2),
                            "detections": detection_data
                        }
                        batch_results_data.append(batch_item)
                        
                        success_count += 1
                        self._append_log(f"✅ {image_path.name} | {summary} | {elapsed:.1f}ms")
                        
                        # 显示最后一张的结果
                        if idx == total:
                            self.current_original_image = image
                            self.current_result_image = annotated
                            self.current_detection_data = detection_data
                            self.original_label.setPixmap(QPixmap.fromImage(cvimg_to_qimage(image)))
                            self.preview_label.setPixmap(QPixmap.fromImage(cvimg_to_qimage(annotated)))
                            # 显示汇总信息而非单张
                            summary_info = {
                                "batch_summary": f"已检测 {success_count}/{total} 张图片",
                                "last_frame": detection_data
                            }
                            self.json_text.setPlainText(json.dumps(summary_info, indent=2, ensure_ascii=False))
                    else:
                        # 无检测结果也保存
                        batch_item = {
                            "filename": image_path.name,
                            "original_image": image,
                            "result_image": image,
                            "detection_time_ms": round(elapsed, 2),
                            "detections": {"total_count": 0, "class_counts": {}, "detections": []}
                        }
                        batch_results_data.append(batch_item)
                        self._append_log(f"ℹ️ {image_path.name} | 无检测结果")
                        success_count += 1
                        
                except Exception as e:
                    self._append_log(f"❌ 处理失败：{image_path.name} - {str(e)}")
                    failed_count += 1
            
            # 保存批量结果到内存
            self.batch_results = batch_results_data
            
            # 构建汇总JSON
            self.batch_json_data = {
                "batch_info": {
                    "total_images": total,
                    "success": success_count,
                    "failed": failed_count,
                    "confidence_threshold": conf,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                },
                "results": [
                    {
                        "filename": item["filename"],
                        "detection_time_ms": item["detection_time_ms"],
                        "detections": item["detections"]
                    }
                    for item in batch_results_data
                ]
            }
            
            # 标记为批量模式，保存时自动保存所有
            self.current_image_path = None
            self.current_video_path = None
            
            # 显示第一张图片
            self.current_batch_index = total - 1  # 从最后一张开始（已显示）
            
            # 显示批量浏览控件（只有多于1张时才显示）
            if len(batch_results_data) > 1:
                self.batch_nav_container.setVisible(True)
                self.batch_position_label.setText(f"{total}/{total}")
                self.btn_prev_image.setEnabled(True)
                self.btn_next_image.setEnabled(False)
                self._append_log(f"💡 使用 ◀ ▶ 按钮浏览所有图片，点击\"保存结果\"保存数据")
            else:
                # 只有1张图片，不显示浏览控件
                self.batch_nav_container.setVisible(False)
                self._append_log(f"💡 点击\"保存结果\"保存检测数据")
            
            self.preview_status.setText(f"状态：批量检测完成 ({success_count}/{total} 成功)")
            self._append_log(f"🎉 批量检测完成：成功 {success_count}，失败 {failed_count}")
            
            self.btn_save.setEnabled(True)
            
            QMessageBox.information(
                self,
                "批量检测完成",
                f"检测完成！\n\n"
                f"总计：{total} 张\n"
                f"成功：{success_count} 张\n"
                f"失败：{failed_count} 张\n\n"
                f"点击\"保存结果\"选择保存位置。"
            )
            
        except Exception as err:
            QMessageBox.critical(self, "批量检测失败", f"批量检测时出错：\n{err}")
            import traceback
            traceback.print_exc()
        finally:
            QApplication.restoreOverrideCursor()

    def _process_video(self, video_path: Path) -> None:
        """处理整个视频文件并保存结果"""
        try:
            # 打开视频
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                QMessageBox.warning(self, "打开失败", f"无法打开视频文件：\n{video_path}")
                return
            
            # 获取视频属性
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                cap.release()
                QMessageBox.warning(self, "视频错误", "无法获取视频帧数，视频可能损坏。")
                return
            
            # 保存视频路径
            self.current_video_path = video_path
            self.current_image_path = None  # 清除图片路径
            
            # 创建临时输出视频文件
            import tempfile
            temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            output_path = temp_output.name
            temp_output.close()
            
            # 设置视频编码器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            self.preview_status.setText(f"状态：处理视频中 0/{total_frames}")
            self._append_log(f"🎬 开始处理视频：{video_path.name} ({total_frames} 帧)")
            QApplication.setOverrideCursor(Qt.WaitCursor)
            
            conf = self.conf_slider.value() / 100
            frame_count = 0
            last_frame = None
            last_annotated = None
            all_frames_data = []  # 存储所有帧的检测数据
            
            # 逐帧处理
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 每10帧更新一次进度
                if frame_count % 10 == 0 or frame_count == total_frames:
                    self.preview_status.setText(f"状态：处理视频中 {frame_count}/{total_frames}")
                    QApplication.processEvents()  # 更新UI
                
                # 检测当前帧
                assert self.model is not None
                results = self.model.predict(source=frame, conf=conf, verbose=False)
                
                if results:
                    annotated, summary, detection_data = self._render_result(frame, results[0])
                    out.write(annotated)
                    
                    # 保存当前帧的检测数据
                    frame_data = {
                        "frame_number": frame_count,
                        "timestamp": frame_count / fps,  # 时间戳（秒）
                        "detections": detection_data
                    }
                    all_frames_data.append(frame_data)
                    
                    # 保存最后一帧的结果用于显示
                    last_frame = frame
                    last_annotated = annotated
                else:
                    out.write(frame)
                    # 即使没有检测结果也记录该帧
                    frame_data = {
                        "frame_number": frame_count,
                        "timestamp": frame_count / fps,
                        "detections": {"total_count": 0, "class_counts": {}, "detections": []}
                    }
                    all_frames_data.append(frame_data)
            
            # 释放资源
            cap.release()
            out.release()
            
            # 保存处理后的视频路径
            self.processed_video_path = output_path
            
            # 构建完整的视频检测数据
            total_detections = sum(f["detections"]["total_count"] for f in all_frames_data)
            video_detection_data = {
                "video_info": {
                    "filename": video_path.name,
                    "total_frames": frame_count,
                    "fps": fps,
                    "duration": frame_count / fps,
                    "resolution": f"{width}x{height}",
                    "total_detections": total_detections
                },
                "frames": all_frames_data
            }
            
            # 保存完整的视频检测数据
            self.current_detection_data = video_detection_data
            
            # 显示最后一帧
            if last_frame is not None and last_annotated is not None:
                self.current_original_image = last_frame
                self.current_result_image = last_annotated
                
                self.original_label.setPixmap(QPixmap.fromImage(cvimg_to_qimage(last_frame)))
                self.preview_label.setPixmap(QPixmap.fromImage(cvimg_to_qimage(last_annotated)))
                self.json_text.setPlainText(json.dumps(video_detection_data, indent=2, ensure_ascii=False))
                
                self.btn_save.setEnabled(True)
                self.preview_status.setText(f"状态：视频处理完成 ({frame_count} 帧, {total_detections} 次检测)")
                self._append_log(f"✅ 视频处理完成：{video_path.name} ({frame_count} 帧, {total_detections} 次检测)")
            else:
                self.preview_status.setText("状态：视频处理完成，无检测结果")
                self._append_log(f"⚠️ 视频处理完成，但无检测结果")
            
            QMessageBox.information(
                self,
                "处理完成",
                f"视频检测完成！\n\n总帧数：{frame_count}\n视频：{video_path.name}\n\n点击\"保存结果\"可保存检测后的视频。"
            )
            
        except Exception as err:
            QMessageBox.critical(self, "视频处理失败", f"处理视频时出错：\n{err}")
            import traceback
            traceback.print_exc()
        finally:
            QApplication.restoreOverrideCursor()
            if 'cap' in locals():
                cap.release()
            if 'out' in locals():
                out.release()

    def _start_stream(self, source: str) -> None:
        try:
            is_network = source.startswith('rtsp://') or source.startswith('http://') or source.startswith('https://')
            
            if source == "webcam":
                capture = cv2.VideoCapture(0)
                source_name = "本地摄像头"
            elif is_network:
                # 网络摄像头，使用FFMPEG后端 + 低延迟配置
                self._append_log(f"🌐 正在连接网络摄像头：{source}")
                QApplication.setOverrideCursor(Qt.WaitCursor)
                
                # 设置低延迟FFMPEG选项
                import os
                ffmpeg_opts = [
                    ("rtsp_transport", "tcp"),  # 使用TCP，更稳定
                    ("fflags", "nobuffer"),     # 无缓冲
                    ("reorder_queue_size", "0"),
                    ("max_delay", "0"),
                    ("buffer_size", "1024"),
                    ("stimeout", "5000000"),    # 5秒超时
                    ("analyzeduration", "0"),
                    ("probesize", "4096"),
                ]
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ";".join(f"{k};{v}" for k, v in ffmpeg_opts)
                
                capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                
                # 设置最小缓冲
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                source_name = "网络摄像头"
            else:
                # 本地视频文件
                capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                source_name = Path(source).name
            
            if not capture.isOpened():
                QApplication.restoreOverrideCursor()
                error_msg = f"无法打开：{source}\n\n"
                if is_network:
                    error_msg += "可能原因：\n1. 网络连接问题\n2. URL地址错误\n3. 摄像头需要认证\n4. RTSP端口被防火墙阻止"
                else:
                    error_msg += "可能原因：\n1. 视频文件损坏\n2. 不支持的编码格式\n3. 文件路径包含特殊字符"
                QMessageBox.warning(self, "打开失败", error_msg)
                return

            # 测试读取第一帧
            ret, test_frame = capture.read()
            QApplication.restoreOverrideCursor()
            
            if not ret or test_frame is None:
                capture.release()
                error_msg = "无法读取视频帧。\n\n"
                if is_network:
                    error_msg += "请检查：\n1. 网络连接是否稳定\n2. URL是否正确\n3. 摄像头是否在线"
                else:
                    error_msg += "请尝试：\n1. 使用其他视频格式\n2. 重新编码视频"
                QMessageBox.warning(self, "读取失败", error_msg)
                return
            
            # 对于本地视频文件，重置到开始位置
            if not is_network and source != "webcam":
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # 清除静态图片路径，避免在视频模式下误触发重新检测
            self.current_image_path = None

            self.capture = capture
            self.streaming = True
            self.stream_source = source
            self.btn_stop.setEnabled(True)
            self.preview_status.setText(f"状态：播放中 ({source_name})")
            
            # 获取视频FPS，动态设置刷新率
            fps = capture.get(cv2.CAP_PROP_FPS)
            if fps > 0 and fps < 120:
                interval = int(1000 / fps)
            else:
                interval = 30  # 默认30ms
            
            self.timer.start(interval)
            
            if is_network:
                self._append_log(f"✅ 网络摄像头已连接：{source}")
            else:
                self._append_log(f"🎬 {source_name} 已加载 (FPS: {fps:.1f})")
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "加载视频错误",
                f"加载视频时发生错误：\n{str(e)}\n\n请检查视频文件是否完整。"
            )

    def _update_stream_frame(self) -> None:
        if not self.capture or not self.streaming:
            return
        
        # 低延迟优化：使用grab+retrieve快速丢弃旧帧，只处理最新帧
        is_network = isinstance(self.stream_source, str) and (
            self.stream_source.startswith('rtsp://') or 
            self.stream_source.startswith('http://') or 
            self.stream_source.startswith('https://')
        )
        
        if is_network:
            # 对网络流，快速抓取多帧，只取最后一帧（丢弃缓冲区旧帧）
            for _ in range(3):  # 连续抓取3次，确保获取最新帧
                if not self.capture.grab():
                    break
            ok, frame = self.capture.retrieve()
        else:
            # 本地摄像头或视频文件，正常读取
            ok, frame = self.capture.read()
        
        if not ok or frame is None:
            self._append_log("⚠️ 视频读取结束/失败")
            self._on_stop_stream()
            return

        conf = self.conf_slider.value() / 100

        try:
            assert self.model is not None
            results = self.model.predict(source=frame, conf=conf, verbose=False, stream=False, device=self.gpu_device)
            if results:
                annotated, summary, detection_data = self._render_result(frame, results[0])
                
                # 更新显示
                self.current_original_image = frame
                self.current_result_image = annotated
                self.current_detection_data = detection_data
                self.btn_save.setEnabled(True)
                
                self.original_label.setPixmap(QPixmap.fromImage(cvimg_to_qimage(frame)))
                self.preview_label.setPixmap(QPixmap.fromImage(cvimg_to_qimage(annotated)))
                self.json_text.setPlainText(json.dumps(detection_data, indent=2, ensure_ascii=False))
                self.preview_status.setText(f"状态：播放中 | {summary}")
        except Exception as err:
            self._append_log(f"❌ 推理错误：{err}")
            self._on_stop_stream()

    def _render_result(self, frame: np.ndarray, result) -> Tuple[np.ndarray, str, Dict]:
        annotated = frame.copy()
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return annotated, "无检测", {"detections": [], "count": 0}

        class_names = self.model.names if self.model else {}
        counts: Dict[str, int] = {}
        detections = []

        for idx, (xyxy, conf, cls_idx) in enumerate(
            zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy())
        ):
            x1, y1, x2, y2 = map(int, xyxy)
            conf_value = float(conf)
            class_id = int(cls_idx)
            class_name = class_names.get(class_id, f"class_{class_id}")  # type: ignore[arg-type]
            counts[class_name] = counts.get(class_name, 0) + 1

            # 保存检测数据
            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(conf_value, 4),
                "bbox": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "width": int(x2 - x1),
                    "height": int(y2 - y1)
                }
            })

            color = self._color_for_class(class_id)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {conf_value:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(
                annotated,
                (x1, y1 - label_height - baseline - 6),
                (x1 + label_width + 8, y1),
                (*color, 120),
                cv2.FILLED,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 4, y1 - baseline - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (15, 18, 24),
                2,
                cv2.LINE_AA,
            )

        summary = " | ".join(f"{name}×{cnt}" for name, cnt in counts.items()) if counts else "无检测"
        detection_data = {
            "total_count": len(detections),
            "class_counts": counts,
            "detections": detections
        }
        return annotated, summary, detection_data

    def _color_for_class(self, class_id: int) -> Tuple[int, int, int]:
        if class_id not in self.class_colors:
            color = self.COLOR_PALETTE[class_id % len(self.COLOR_PALETTE)]
            self.class_colors[class_id] = color
        return self.class_colors[class_id]

    def _append_log(self, message: str) -> None:
        item = QListWidgetItem(message)
        item.setForeground(Qt.white)
        item.setFont(QFont(self.font_family, self._font_point(11)))
        self.log_list.addItem(item)
        self.log_list.scrollToBottom()

    def _ensure_model_loaded(self) -> bool:
        if self.model is None:
            QMessageBox.information(self, "请先加载模型", "在进行检测前请先加载或选择一个模型权重。")
            return False
        return True
    
    def _clear_batch_data(self) -> None:
        """清除批量检测数据（自动）"""
        self.batch_results = None
        self.batch_json_data = None
        self.batch_image_paths = None
        self.current_video_path = None
    
    def _on_prev_batch_image(self) -> None:
        """查看上一张批量图片"""
        if not self.batch_results or self.current_batch_index <= 0:
            return
        
        self.current_batch_index -= 1
        self._show_batch_image(self.current_batch_index)
    
    def _on_next_batch_image(self) -> None:
        """查看下一张批量图片"""
        if not self.batch_results or self.current_batch_index >= len(self.batch_results) - 1:
            return
        
        self.current_batch_index += 1
        self._show_batch_image(self.current_batch_index)
    
    def _show_batch_image(self, index: int) -> None:
        """显示指定索引的批量图片"""
        if not self.batch_results or index < 0 or index >= len(self.batch_results):
            return
        
        item = self.batch_results[index]
        
        # 更新显示
        self.current_original_image = item['original_image']
        self.current_result_image = item['result_image']
        self.current_detection_data = item['detections']
        
        self.original_label.setPixmap(QPixmap.fromImage(cvimg_to_qimage(item['original_image'])))
        self.preview_label.setPixmap(QPixmap.fromImage(cvimg_to_qimage(item['result_image'])))
        self.json_text.setPlainText(json.dumps(item['detections'], indent=2, ensure_ascii=False))
        
        # 更新位置显示
        total = len(self.batch_results)
        self.batch_position_label.setText(f"{index + 1}/{total}")
        
        # 更新按钮状态
        self.btn_prev_image.setEnabled(index > 0)
        self.btn_next_image.setEnabled(index < total - 1)
        
        # 更新状态栏
        filename = item['filename']
        detection_count = item['detections'].get('total_count', 0)
        self.preview_status.setText(f"状态：批量浏览 [{index + 1}/{total}] {filename} | 检测数：{detection_count}")
    
    def _on_manual_reset(self) -> None:
        """手动重置所有数据"""
        reply = QMessageBox.question(
            self,
            "确认重置",
            "确定要清除当前所有检测数据吗？\n\n这将清除：\n• 检测图片\n• 批量数据\n• JSON数据\n• 预览显示",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 停止视频流
            if self.streaming:
                self._on_stop_stream()
            
            # 清除所有数据
            self.batch_results = None
            self.batch_json_data = None
            self.batch_image_paths = None
            self.current_video_path = None
            self.current_image_path = None
            self.current_result_image = None
            self.current_original_image = None
            self.current_detection_data = None
            self.current_batch_index = 0
            
            # 清除预览
            self.original_label.setText("等待输入")
            self.preview_label.setText("等待检测")
            self.json_text.clear()
            
            # 隐藏批量浏览控件
            self.batch_nav_container.setVisible(False)
            
            # 禁用保存按钮
            self.btn_save.setEnabled(False)
            
            # 重置状态
            self.preview_status.setText("状态：空闲")
            
            self._append_log("🔄 已清除所有检测数据")
            QMessageBox.information(self, "重置成功", "所有检测数据已清除，可以开始新的检测。")

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._on_stop_stream()
        super().closeEvent(event)

    def _determine_scale(self) -> float:
        screen = self.screen() or QApplication.primaryScreen()
        if screen is not None:
            dpi = screen.logicalDotsPerInch() or 96
            scale = dpi / 96.0
            return max(1.0, min(scale, 1.8))
        return 1.0

    def _scale(self, value: int) -> int:
        return max(1, int(round(value * self.scale_factor)))

    def _font_px(self, value: int) -> int:
        return max(value, int(round(value * self.scale_factor)))

    def _font_point(self, value: float) -> int:
        return max(int(round(value)), int(round(value * self.scale_factor)))

    def _get_next_output_folder(self, base_dir: Path) -> Path:
        """在指定目录下生成递增的output文件夹名"""
        index = 1
        while True:
            folder_name = f"output_{index}"
            output_folder = base_dir / folder_name
            if not output_folder.exists():
                return output_folder
            index += 1
    
    def _get_next_report_name(self, base_dir: Path) -> str:
        """生成递增的报告文件名"""
        index = 1
        while True:
            report_name = f"分析报告_{index}.txt"
            report_path = base_dir / report_name
            if not report_path.exists():
                return report_name
            index += 1
    
    def _on_logout(self) -> None:
        """退出登录"""
        reply = QMessageBox.question(
            self,
            "确认退出",
            f"确定要退出当前账户 {self.current_username} 吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 停止所有检测活动
            self._on_stop_stream()
            
            # 关闭主窗口
            self.close()
            
            # 重新显示登录窗口
            from login_window import LoginWindow
            login_window = LoginWindow()
            if login_window.exec_() == QDialog.Accepted:
                # 登录成功，创建新的主窗口
                new_window = DetectionWindow(login_window.current_username if hasattr(login_window, 'current_username') else 'Guest')
                new_window.show()
                # 保持应用运行
                QApplication.instance().setActiveWindow(new_window)
            else:
                # 取消登录，退出程序
                QApplication.instance().quit()


def main() -> None:
    # 关键修复：在导入任何CUDA相关库之前设置环境变量
    import os
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # 强制使用CPU，避免CUDA错误
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    if hasattr(Qt, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, "AA_UseHighDpiPixmaps"):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    if hasattr(QApplication, "setHighDpiScaleFactorRoundingPolicy") and hasattr(
        Qt, "HighDpiScaleFactorRoundingPolicy"
    ):
        QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)  # type: ignore[attr-defined]

    font = QFont("Microsoft YaHei UI", 10)
    app.setFont(font)
    
    # 导入登录窗口
    from login_window import LoginWindow
    
    # 显示登录窗口
    login_window = LoginWindow()
    
    # 用于存储登录的用户名
    logged_username = ["Guest"]
    
    def on_login_success(username):
        logged_username[0] = username
    
    login_window.login_success.connect(on_login_success)
    
    # 登录成功后显示主窗口
    if login_window.exec_() == QDialog.Accepted:
        window = DetectionWindow(logged_username[0])
        window.show()
        sys.exit(app.exec_())
    else:
        # 用户取消登录，直接退出
        sys.exit(0)


if __name__ == "__main__":
    main()

