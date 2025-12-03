"""
登录注册界面模块
支持用户登录、注册功能，带背景图片
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QPalette, QBrush
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFrame,
    QMessageBox,
    QTabWidget,
    QWidget,
    QGraphicsDropShadowEffect,
)


class LoginWindow(QDialog):
    """登录注册窗口"""
    
    login_success = pyqtSignal(str)  # 登录成功信号，传递用户名
    
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("太赫兹成像内部探伤检测系统 - 登录")
        
        # 获取屏幕尺寸
        from PyQt5.QtWidgets import QApplication
        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()
        
        # 自适应窗口尺寸（占屏幕的75%）
        window_width = int(screen_width * 0.75)
        window_height = int(screen_height * 0.75)
        
        # 设置最小尺寸
        min_width = max(1000, int(screen_width * 0.6))
        min_height = max(700, int(screen_height * 0.6))
        
        self.resize(window_width, window_height)
        self.setMinimumSize(min_width, min_height)
        
        # 用户数据文件
        self.users_file = Path("users.json")
        self._load_users()
        
        # 记住账号密码配置文件
        self.remember_file = Path("remember.json")
        self._load_remember()
        
        # 当前显示模式：login 或 register
        self.current_mode = "login"
        
        self._init_ui()
        self._set_background()
    
    def _load_users(self) -> None:
        """加载用户数据"""
        if self.users_file.exists():
            with open(self.users_file, 'r', encoding='utf-8') as f:
                self.users = json.load(f)
        else:
            # 默认管理员账户
            self.users = {
                "admin": {
                    "password": self._hash_password("admin123"),
                    "email": "admin@example.com"
                }
            }
            self._save_users()
    
    def _save_users(self) -> None:
        """保存用户数据"""
        with open(self.users_file, 'w', encoding='utf-8') as f:
            json.dump(self.users, f, indent=2, ensure_ascii=False)
    
    def _load_remember(self) -> None:
        """加载记住的账号密码"""
        if self.remember_file.exists():
            with open(self.remember_file, 'r', encoding='utf-8') as f:
                self.remember_data = json.load(f)
        else:
            self.remember_data = {
                "remember_username": False,
                "remember_password": False,
                "username": "",
                "password": ""
            }
    
    def _save_remember(self) -> None:
        """保存记住的账号密码"""
        with open(self.remember_file, 'w', encoding='utf-8') as f:
            json.dump(self.remember_data, f, indent=2, ensure_ascii=False)
    
    def _hash_password(self, password: str) -> str:
        """密码哈希"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _set_background(self) -> None:
        """设置背景图片"""
        # 预留背景图片位置
        bg_image_path = Path("bj3.png")
        
        if bg_image_path.exists():
            # 如果存在背景图片，使用图片作为背景
            palette = QPalette()
            pixmap = QPixmap(str(bg_image_path))
            scaled_pixmap = pixmap.scaled(
                self.size(), 
                Qt.KeepAspectRatioByExpanding, 
                Qt.SmoothTransformation
            )
            palette.setBrush(QPalette.Background, QBrush(scaled_pixmap))
            self.setPalette(palette)
        else:
            # 使用渐变背景
            self.setStyleSheet("""
                QDialog {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 #0a0e1a, stop:0.5 #1a2332, stop:1 #0f1628);
                }
            """)
    
    def _init_ui(self) -> None:
        """初始化UI - 中央单卡片布局"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setAlignment(Qt.AlignCenter)
        
        # 中央容器
        center_container = QWidget()
        center_layout = QVBoxLayout(center_container)
        center_layout.setAlignment(Qt.AlignCenter)
        #
        center_layout.setSpacing(30)
        
        # 顶部Logo和标题
        header = self._create_header()
        
        # 卡片容器（根据模式切换显示）
        self.card_container = QWidget()
        self.card_layout = QVBoxLayout(self.card_container)
        self.card_layout.setContentsMargins(0, 0, 0, 0)
        self.card_layout.setAlignment(Qt.AlignCenter)
        
        # 默认显示登录卡片
        self.login_card = self._create_login_card()
        self.register_card = self._create_register_card()
        self.register_card.hide()  # 初始隐藏注册卡片
        
        self.card_layout.addWidget(self.login_card)
        self.card_layout.addWidget(self.register_card)
        
        center_layout.addWidget(header)
        center_layout.addWidget(self.card_container)
        
        main_layout.addStretch(1)
        main_layout.addWidget(center_container)
        main_layout.addStretch(1)
    
    def _create_header(self) -> QWidget:
        """创建顶部区域"""
        header = QWidget()
        layout = QVBoxLayout(header)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(15)
        
        # Logo图标
        logo = QLabel("🔬")
        logo.setFont(QFont("Segoe UI Emoji", 80))
        logo.setAlignment(Qt.AlignCenter)
        
        # 主标题
        title = QLabel("太赫兹成像内部探伤检测系统")
        title.setFont(QFont("Microsoft YaHei UI", 30, QFont.Bold))
        title.setStyleSheet("""
            color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #e9d5ff, stop:0.5 #c084fc, stop:1 #a855f7);
        """)
        title.setAlignment(Qt.AlignCenter)
        title.setWordWrap(True)
        
        subtitle = QLabel("Terahertz Imaging Internal Flaw Detection System")
        subtitle.setFont(QFont("Arial", 15))
        subtitle.setStyleSheet("color: #94a3b8;")
        subtitle.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(logo)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        
        return header
    
    def _switch_to_register(self) -> None:
        """切换到注册页面"""
        self.login_card.hide()
        self.register_card.show()
        self.current_mode = "register"
    
    def _switch_to_login(self) -> None:
        """切换到登录页面"""
        self.register_card.hide()
        self.login_card.show()
        self.current_mode = "login"
    
    def _create_login_card(self) -> QFrame:
        """创建登录卡片"""
        card = QFrame()
        # 卡片大小自适应（最大600，最小400）
        card_width = max(400, min(600, int(self.width() * 0.4)))
        card_height = max(480, min(580, int(self.height() * 0.65)))
        card.setFixedSize(card_width, card_height)
        card.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(30, 41, 59, 0.60), 
                    stop:0.5 rgba(15, 23, 42, 0.70),
                    stop:1 rgba(15, 23, 42, 0.75));
                border-radius: 28px;
                border: 2px solid rgba(192, 132, 252, 0.35);
            }
        """)
        
        # 添加发光阴影
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(80)
        shadow.setOffset(0, 25)
        shadow.setColor(Qt.black)
        card.setGraphicsEffect(shadow)
        
        layout = QVBoxLayout(card)
        layout.setContentsMargins(70, 60, 70, 60)
        layout.setSpacing(25)
        
        # 卡片标题
        card_title = QLabel("🔑 账户登录")
        card_title.setFont(QFont("Microsoft YaHei UI", 28, QFont.Bold))
        card_title.setStyleSheet("""
            color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #e0f2fe, stop:1 #93c5fd);
            padding: 8px;
        """)
        card_title.setAlignment(Qt.AlignCenter)
        
        # 分隔线
        divider = QFrame()
        divider.setFixedHeight(2)
        divider.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 transparent, 
                stop:0.5 rgba(96, 165, 250, 0.5), 
                stop:1 transparent);
        """)
        
        layout.addWidget(card_title)
        layout.addWidget(divider)
        layout.addSpacing(20)
        layout.addLayout(self._create_login_form())
        
        return card
    
    def _create_register_card(self) -> QFrame:
        """创建注册卡片"""
        card = QFrame()
        # 卡片大小自适应
        card_width = max(400, min(600, int(self.width() * 0.4)))
        card_height = max(550, min(680, int(self.height() * 0.7)))
        card.setFixedSize(card_width, card_height)
        card.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(30, 41, 59, 0.50), 
                    stop:0.5 rgba(15, 23, 42, 0.50),
                    stop:1 rgba(15, 23, 42, 0.65));
                border-radius: 28px;
                border: 2px solid rgba(192, 132, 252, 0.35);
            }
        """)
        
        # 添加发光阴影
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(80)
        shadow.setOffset(0, 25)
        shadow.setColor(Qt.black)
        card.setGraphicsEffect(shadow)
        
        layout = QVBoxLayout(card)
        layout.setContentsMargins(70, 55, 70, 55)
        layout.setSpacing(18)
        
        # 卡片标题
        card_title = QLabel("✨ 创建账户")
        card_title.setFont(QFont("Microsoft YaHei UI", 28, QFont.Bold))
        card_title.setStyleSheet("""
            color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #6ee7b7, stop:1 #34d399);
            padding: 8px;
        """)
        card_title.setAlignment(Qt.AlignCenter)
        
        # 分隔线
        divider = QFrame()
        divider.setFixedHeight(2)
        divider.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 transparent, 
                stop:0.5 rgba(52, 211, 153, 0.5), 
                stop:1 transparent);
        """)
        
        layout.addWidget(card_title)
        layout.addWidget(divider)
        layout.addSpacing(18)
        layout.addLayout(self._create_register_form())
        
        return card
    
    def _create_login_form(self) -> QVBoxLayout:
        """创建登录表单"""
        layout = QVBoxLayout()
        layout.setSpacing(15)
        
        # 用户名输入行
        username_row = QHBoxLayout()
        username_row.setSpacing(15)
        
        username_label = QLabel("👤 用户名")
        username_label.setFixedWidth(100)
        username_label.setStyleSheet("""
            color: #e2e8f0; 
            font-size: 17px; 
            font-weight: 600;
            background: transparent;
            border: none;
        """)
        
        self.login_username = QLineEdit()
        self.login_username.setPlaceholderText("请输入用户名")
        self.login_username.setFixedHeight(50)
        self.login_username.setStyleSheet("""
            QLineEdit {
                background-color: rgba(15, 23, 42, 0.5);
                border: 2px solid rgba(71, 85, 105, 0.4);
                border-radius: 14px;
                padding: 12px 18px;
                color: #f8fafc;
                font-size: 15px;
            }
            QLineEdit:focus {
                border: 2px solid rgba(96, 165, 250, 0.7);
                background-color: rgba(30, 41, 59, 0.7);
                box-shadow: 0 0 15px rgba(96, 165, 250, 0.2);
            }
            QLineEdit:hover {
                border: 2px solid rgba(96, 165, 250, 0.5);
            }
        """)
        
        username_row.addWidget(username_label)
        username_row.addWidget(self.login_username)
        
        # 密码输入行
        password_row = QHBoxLayout()
        password_row.setSpacing(15)
        
        password_label = QLabel("🔒 密码")
        password_label.setFixedWidth(100)
        password_label.setStyleSheet("""
            color: #e2e8f0; 
            font-size: 17px; 
            font-weight: 600;
            background: transparent;
            border: none;
        """)
        
        self.login_password = QLineEdit()
        self.login_password.setPlaceholderText("请输入密码")
        self.login_password.setEchoMode(QLineEdit.Password)
        self.login_password.setFixedHeight(50)
        self.login_password.setStyleSheet("""
            QLineEdit {
                background-color: rgba(15, 23, 42, 0.5);
                border: 2px solid rgba(71, 85, 105, 0.4);
                border-radius: 14px;
                padding: 12px 18px;
                color: #f8fafc;
                font-size: 15px;
            }
            QLineEdit:focus {
                border: 2px solid rgba(96, 165, 250, 0.7);
                background-color: rgba(30, 41, 59, 0.7);
                box-shadow: 0 0 15px rgba(96, 165, 250, 0.2);
            }
            QLineEdit:hover {
                border: 2px solid rgba(96, 165, 250, 0.5);
            }
        """)
        
        password_row.addWidget(password_label)
        password_row.addWidget(self.login_password)
        
        # 登录按钮
        login_btn = QPushButton("🚀 立即登录")
        login_btn.setFixedHeight(54)
        login_btn.setCursor(Qt.PointingHandCursor)
        login_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3b82f6, stop:1 #2563eb);
                border: 2px solid rgba(59, 130, 246, 0.4);
                border-radius: 14px;
                color: white;
                font-size: 17px;
                font-weight: 700;
                letter-spacing: 1px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #60a5fa, stop:1 #3b82f6);
                border: 2px solid rgba(96, 165, 250, 0.6);
                box-shadow: 0 0 30px rgba(59, 130, 246, 0.5);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2563eb, stop:1 #1d4ed8);
                transform: scale(0.98);
            }
        """)
        login_btn.clicked.connect(self._on_login)
        
        # 记住账号密码选项
        remember_layout = QHBoxLayout()
        remember_layout.setSpacing(30)
        
        self.remember_username_cb = QCheckBox("记住账号")
        self.remember_username_cb.setStyleSheet("""
            QCheckBox {
                color: #cbd5e1;
                font-size: 14px;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border-radius: 6px;
                border: 2px solid rgba(96, 165, 250, 0.4);
                background: rgba(15, 23, 42, 0.5);
            }
            QCheckBox::indicator:checked {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3b82f6, stop:1 #2563eb);
                border: 2px solid rgba(96, 165, 250, 0.6);
            }
            QCheckBox::indicator:hover {
                border: 2px solid rgba(96, 165, 250, 0.7);
            }
        """)
        
        self.remember_password_cb = QCheckBox("记住密码")
        self.remember_password_cb.setStyleSheet("""
            QCheckBox {
                color: #cbd5e1;
                font-size: 14px;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border-radius: 6px;
                border: 2px solid rgba(96, 165, 250, 0.4);
                background: rgba(15, 23, 42, 0.5);
            }
            QCheckBox::indicator:checked {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3b82f6, stop:1 #2563eb);
                border: 2px solid rgba(96, 165, 250, 0.6);
            }
            QCheckBox::indicator:hover {
                border: 2px solid rgba(96, 165, 250, 0.7);
            }
        """)
        
        # 加载保存的状态
        self.remember_username_cb.setChecked(self.remember_data.get("remember_username", False))
        self.remember_password_cb.setChecked(self.remember_data.get("remember_password", False))
        
        # 如果勾选了记住，自动填充
        if self.remember_data.get("remember_username"):
            self.login_username.setText(self.remember_data.get("username", ""))
        if self.remember_data.get("remember_password"):
            self.login_password.setText(self.remember_data.get("password", ""))
        
        remember_layout.addWidget(self.remember_username_cb)
        remember_layout.addWidget(self.remember_password_cb)
        remember_layout.addStretch()
        
        layout.addLayout(username_row)
        layout.addSpacing(22)
        layout.addLayout(password_row)
        layout.addSpacing(20)
        layout.addLayout(remember_layout)
        layout.addSpacing(30)
        layout.addWidget(login_btn)
        layout.addSpacing(30)
        
        # 切换到注册按钮
        switch_layout = QHBoxLayout()
        switch_label = QLabel("还没有账户？")
        switch_label.setStyleSheet("color: #94a3b8; font-size: 13px;")
        
        switch_btn = QPushButton("立即注册")
        switch_btn.setCursor(Qt.PointingHandCursor)
        switch_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                color: #60a5fa;
                font-size: 13px;
                font-weight: 600;
                text-decoration: underline;
            }
            QPushButton:hover {
                color: #93c5fd;
            }
        """)
        switch_btn.clicked.connect(self._switch_to_register)
        
        switch_layout.addStretch()
        switch_layout.addWidget(switch_label)
        switch_layout.addWidget(switch_btn)
        switch_layout.addStretch()
        
        layout.addLayout(switch_layout)
        
        return layout
    
    def _create_register_form(self) -> QVBoxLayout:
        """创建注册表单"""
        layout = QVBoxLayout()
        layout.setSpacing(12)
        
        # 用户名输入
        username_label = QLabel("👤 用户名")
        username_label.setStyleSheet("color: #e2e8f0; font-size: 14px; font-weight: 600;")
        
        self.register_username = QLineEdit()
        self.register_username.setPlaceholderText("请输入用户名（3-20个字符）")
        self.register_username.setFixedHeight(50)
        input_style = """
            QLineEdit {
                background-color: rgba(15, 23, 42, 0.5);
                border: 2px solid rgba(71, 85, 105, 0.4);
                border-radius: 14px;
                padding: 12px 18px;
                color: #f8fafc;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 2px solid rgba(96, 165, 250, 0.7);
                background-color: rgba(30, 41, 59, 0.7);
                box-shadow: 0 0 15px rgba(96, 165, 250, 0.2);
            }
            QLineEdit:hover {
                border: 2px solid rgba(96, 165, 250, 0.5);
            }
        """
        
        # 用户名输入行
        username_row = QHBoxLayout()
        username_row.setSpacing(15)
        
        username_label = QLabel("👤 用户名")
        username_label.setFixedWidth(110)
        username_label.setStyleSheet("""
            color: #e2e8f0; 
            font-size: 17px; 
            font-weight: 600;
            background: transparent;
            border: none;
        """)
        
        self.register_username.setStyleSheet(input_style)
        username_row.addWidget(username_label)
        username_row.addWidget(self.register_username)
        
        # 邮箱输入行
        email_row = QHBoxLayout()
        email_row.setSpacing(15)
        
        email_label = QLabel("📧 邮箱")
        email_label.setFixedWidth(110)
        email_label.setStyleSheet("""
            color: #e2e8f0; 
            font-size: 17px; 
            font-weight: 600;
            background: transparent;
            border: none;
        """)
        
        self.register_email = QLineEdit()
        self.register_email.setPlaceholderText("请输入邮箱地址")
        self.register_email.setFixedHeight(50)
        self.register_email.setStyleSheet(input_style)
        
        email_row.addWidget(email_label)
        email_row.addWidget(self.register_email)
        
        # 密码输入行
        password_row = QHBoxLayout()
        password_row.setSpacing(15)
        
        password_label = QLabel("🔒 密码")
        password_label.setFixedWidth(110)
        password_label.setStyleSheet("""
            color: #e2e8f0; 
            font-size: 17px; 
            font-weight: 600;
            background: transparent;
            border: none;
        """)
        
        self.register_password = QLineEdit()
        self.register_password.setPlaceholderText("请输入密码（至少6位）")
        self.register_password.setEchoMode(QLineEdit.Password)
        self.register_password.setFixedHeight(50)
        self.register_password.setStyleSheet(input_style)
        
        password_row.addWidget(password_label)
        password_row.addWidget(self.register_password)
        
        # 确认密码行
        confirm_row = QHBoxLayout()
        confirm_row.setSpacing(15)
        
        confirm_label = QLabel("🔑 确认密码")
        confirm_label.setFixedWidth(110)
        confirm_label.setStyleSheet("""
            color: #e2e8f0; 
            font-size: 17px; 
            font-weight: 600;
            background: transparent;
            border: none;
        """)
        
        self.register_confirm = QLineEdit()
        self.register_confirm.setPlaceholderText("请再次输入密码")
        self.register_confirm.setEchoMode(QLineEdit.Password)
        self.register_confirm.setFixedHeight(50)
        self.register_confirm.setStyleSheet(input_style)
        
        confirm_row.addWidget(confirm_label)
        confirm_row.addWidget(self.register_confirm)
        
        # 注册按钮
        register_btn = QPushButton("✨ 创建账户")
        register_btn.setFixedHeight(54)
        register_btn.setCursor(Qt.PointingHandCursor)
        register_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #10b981, stop:1 #059669);
                border: 2px solid rgba(16, 185, 129, 0.4);
                border-radius: 14px;
                color: white;
                font-size: 17px;
                font-weight: 700;
                letter-spacing: 1px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #34d399, stop:1 #10b981);
                border: 2px solid rgba(52, 211, 153, 0.6);
                box-shadow: 0 0 30px rgba(16, 185, 129, 0.5);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #059669, stop:1 #047857);
                transform: scale(0.98);
            }
        """)
        register_btn.clicked.connect(self._on_register)
        
        layout.addLayout(username_row)
        layout.addSpacing(18)
        layout.addLayout(email_row)
        layout.addSpacing(18)
        layout.addLayout(password_row)
        layout.addSpacing(18)
        layout.addLayout(confirm_row)
        layout.addSpacing(30)
        layout.addWidget(register_btn)
        layout.addSpacing(20)
        
        # 切换到登录按钮
        switch_layout = QHBoxLayout()
        switch_label = QLabel("已有账户？")
        switch_label.setStyleSheet("color: #94a3b8; font-size: 13px;")
        
        switch_btn = QPushButton("立即登录")
        switch_btn.setCursor(Qt.PointingHandCursor)
        switch_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                color: #34d399;
                font-size: 13px;
                font-weight: 600;
                text-decoration: underline;
            }
            QPushButton:hover {
                color: #6ee7b7;
            }
        """)
        switch_btn.clicked.connect(self._switch_to_login)
        
        switch_layout.addStretch()
        switch_layout.addWidget(switch_label)
        switch_layout.addWidget(switch_btn)
        switch_layout.addStretch()
        
        layout.addLayout(switch_layout)
        
        return layout
    
    def _on_login(self) -> None:
        """处理登录"""
        username = self.login_username.text().strip()
        password = self.login_password.text()
        
        if not username or not password:
            QMessageBox.warning(self, "输入错误", "请输入用户名和密码")
            return
        
        # 验证用户
        if username not in self.users:
            QMessageBox.warning(self, "登录失败", "用户名不存在")
            return
        
        if self.users[username]["password"] != self._hash_password(password):
            QMessageBox.warning(self, "登录失败", "密码错误")
            return
        
        # 保存记住的账号密码
        self.remember_data["remember_username"] = self.remember_username_cb.isChecked()
        self.remember_data["remember_password"] = self.remember_password_cb.isChecked()
        
        if self.remember_username_cb.isChecked():
            self.remember_data["username"] = username
        else:
            self.remember_data["username"] = ""
        
        if self.remember_password_cb.isChecked():
            self.remember_data["password"] = password
        else:
            self.remember_data["password"] = ""
        
        self._save_remember()
        
        # 登录成功
        QMessageBox.information(self, "登录成功", f"欢迎回来，{username}！")
        self.login_success.emit(username)
        self.accept()
    
    def _on_register(self) -> None:
        """处理注册"""
        username = self.register_username.text().strip()
        email = self.register_email.text().strip()
        password = self.register_password.text()
        confirm = self.register_confirm.text()
        
        # 验证输入
        if not username or not email or not password:
            QMessageBox.warning(self, "输入错误", "请填写所有字段")
            return
        
        if len(username) < 3 or len(username) > 20:
            QMessageBox.warning(self, "用户名错误", "用户名长度应为3-20个字符")
            return
        
        if len(password) < 6:
            QMessageBox.warning(self, "密码错误", "密码长度至少6位")
            return
        
        if password != confirm:
            QMessageBox.warning(self, "密码错误", "两次输入的密码不一致")
            return
        
        if username in self.users:
            QMessageBox.warning(self, "注册失败", "用户名已存在")
            return
        
        # 保存新用户
        self.users[username] = {
            "password": self._hash_password(password),
            "email": email
        }
        self._save_users()
        
        QMessageBox.information(
            self, 
            "注册成功", 
            f"账户 {username} 注册成功！\n即将切换到登录页面。"
        )
        
        # 切换到登录页面并填充用户名
        self._switch_to_login()
        self.login_username.setText(username)
        self.login_password.clear()
        self.login_password.setFocus()
    
    def resizeEvent(self, event) -> None:
        """窗口大小改变时重新设置背景和调整卡片大小"""
        super().resizeEvent(event)
        self._set_background()
        
        # 动态调整卡片大小
        if hasattr(self, 'login_card') and self.login_card:
            card_width = max(400, min(600, int(self.width() * 0.4)))
            card_height = max(480, min(580, int(self.height() * 0.65)))
            self.login_card.setFixedSize(card_width, card_height)
        
        if hasattr(self, 'register_card') and self.register_card:
            card_width = max(400, min(600, int(self.width() * 0.4)))
            card_height = max(550, min(680, int(self.height() * 0.7)))
            self.register_card.setFixedSize(card_width, card_height)

