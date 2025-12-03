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
        self.setWindowTitle("猪只计数系统 - 登录")
        self.resize(1000, 700)
        self.setMinimumSize(900, 650)
        
        # 用户数据文件
        self.users_file = Path("users.json")
        self._load_users()
        
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
    
    def _hash_password(self, password: str) -> str:
        """密码哈希"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _set_background(self) -> None:
        """设置背景图片"""
        # 预留背景图片位置
        bg_image_path = Path("login_background.jpg")
        
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
        """初始化UI"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 左侧：背景展示区域（预留）
        left_spacer = QWidget()
        left_spacer.setMinimumWidth(400)
        left_spacer.setStyleSheet("background: transparent;")
        
        # 右侧：登录注册卡片
        login_card = self._create_login_card()
        
        layout.addWidget(left_spacer, 2)
        layout.addWidget(login_card, 1)
    
    def _create_login_card(self) -> QFrame:
        """创建登录注册卡片"""
        card = QFrame()
        card.setMaximumWidth(480)
        card.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(30, 41, 59, 0.95), 
                    stop:1 rgba(15, 23, 42, 0.98));
                border-radius: 24px;
                border: 2px solid rgba(96, 165, 250, 0.2);
            }
        """)
        
        # 添加阴影效果
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(50)
        shadow.setOffset(0, 10)
        shadow.setColor(Qt.black)
        card.setGraphicsEffect(shadow)
        
        layout = QVBoxLayout(card)
        layout.setContentsMargins(50, 50, 50, 50)
        layout.setSpacing(20)
        
        # 标题
        title = QLabel("猪只智能计数系统")
        title.setFont(QFont("Microsoft YaHei UI", 24, QFont.Bold))
        title.setStyleSheet("""
            color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #dbeafe, stop:0.5 #93c5fd, stop:1 #60a5fa);
        """)
        title.setAlignment(Qt.AlignCenter)
        
        subtitle = QLabel("Multi-scale Pig Counting System")
        subtitle.setFont(QFont("Arial", 11))
        subtitle.setStyleSheet("color: #94a3b8;")
        subtitle.setAlignment(Qt.AlignCenter)
        
        # 创建Tab切换（登录/注册）
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: none;
                background: transparent;
            }
            QTabBar::tab {
                background: rgba(51, 65, 85, 0.3);
                color: #cbd5e1;
                padding: 12px 40px;
                border-radius: 10px 10px 0 0;
                margin-right: 5px;
                font-size: 14px;
                font-weight: 600;
            }
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(59, 130, 246, 0.4), stop:1 rgba(37, 99, 235, 0.3));
                color: white;
            }
            QTabBar::tab:hover {
                background: rgba(71, 85, 105, 0.5);
            }
        """)
        
        # 登录Tab
        login_tab = self._create_login_tab()
        # 注册Tab
        register_tab = self._create_register_tab()
        
        self.tab_widget.addTab(login_tab, "登录")
        self.tab_widget.addTab(register_tab, "注册")
        
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(20)
        layout.addWidget(self.tab_widget)
        layout.addStretch()
        
        return card
    
    def _create_login_tab(self) -> QWidget:
        """创建登录选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 20, 0, 20)
        layout.setSpacing(18)
        
        # 用户名输入
        username_label = QLabel("用户名")
        username_label.setStyleSheet("color: #cbd5e1; font-size: 13px; font-weight: 500;")
        
        self.login_username = QLineEdit()
        self.login_username.setPlaceholderText("请输入用户名")
        self.login_username.setFixedHeight(45)
        self.login_username.setStyleSheet("""
            QLineEdit {
                background-color: rgba(15, 23, 42, 0.7);
                border: 2px solid rgba(71, 85, 105, 0.3);
                border-radius: 12px;
                padding: 10px 15px;
                color: #f1f5f9;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 2px solid rgba(96, 165, 250, 0.6);
                background-color: rgba(30, 41, 59, 0.8);
            }
        """)
        
        # 密码输入
        password_label = QLabel("密码")
        password_label.setStyleSheet("color: #cbd5e1; font-size: 13px; font-weight: 500;")
        
        self.login_password = QLineEdit()
        self.login_password.setPlaceholderText("请输入密码")
        self.login_password.setEchoMode(QLineEdit.Password)
        self.login_password.setFixedHeight(45)
        self.login_password.setStyleSheet("""
            QLineEdit {
                background-color: rgba(15, 23, 42, 0.7);
                border: 2px solid rgba(71, 85, 105, 0.3);
                border-radius: 12px;
                padding: 10px 15px;
                color: #f1f5f9;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 2px solid rgba(96, 165, 250, 0.6);
                background-color: rgba(30, 41, 59, 0.8);
            }
        """)
        
        # 登录按钮
        login_btn = QPushButton("登录")
        login_btn.setFixedHeight(50)
        login_btn.setCursor(Qt.PointingHandCursor)
        login_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3b82f6, stop:1 #2563eb);
                border: 2px solid rgba(59, 130, 246, 0.3);
                border-radius: 12px;
                color: white;
                font-size: 16px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #60a5fa, stop:1 #3b82f6);
                border: 2px solid rgba(96, 165, 250, 0.5);
                box-shadow: 0 0 25px rgba(59, 130, 246, 0.4);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2563eb, stop:1 #1d4ed8);
            }
        """)
        login_btn.clicked.connect(self._on_login)
        
        # 提示信息
        hint = QLabel("默认账户: admin / admin123")
        hint.setStyleSheet("color: #64748b; font-size: 12px;")
        hint.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(username_label)
        layout.addWidget(self.login_username)
        layout.addSpacing(5)
        layout.addWidget(password_label)
        layout.addWidget(self.login_password)
        layout.addSpacing(15)
        layout.addWidget(login_btn)
        layout.addSpacing(10)
        layout.addWidget(hint)
        layout.addStretch()
        
        return widget
    
    def _create_register_tab(self) -> QWidget:
        """创建注册选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 20, 0, 20)
        layout.setSpacing(18)
        
        # 用户名输入
        username_label = QLabel("用户名")
        username_label.setStyleSheet("color: #cbd5e1; font-size: 13px; font-weight: 500;")
        
        self.register_username = QLineEdit()
        self.register_username.setPlaceholderText("请输入用户名（3-20个字符）")
        self.register_username.setFixedHeight(45)
        self.register_username.setStyleSheet("""
            QLineEdit {
                background-color: rgba(15, 23, 42, 0.7);
                border: 2px solid rgba(71, 85, 105, 0.3);
                border-radius: 12px;
                padding: 10px 15px;
                color: #f1f5f9;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 2px solid rgba(96, 165, 250, 0.6);
                background-color: rgba(30, 41, 59, 0.8);
            }
        """)
        
        # 邮箱输入
        email_label = QLabel("邮箱")
        email_label.setStyleSheet("color: #cbd5e1; font-size: 13px; font-weight: 500;")
        
        self.register_email = QLineEdit()
        self.register_email.setPlaceholderText("请输入邮箱地址")
        self.register_email.setFixedHeight(45)
        self.register_email.setStyleSheet("""
            QLineEdit {
                background-color: rgba(15, 23, 42, 0.7);
                border: 2px solid rgba(71, 85, 105, 0.3);
                border-radius: 12px;
                padding: 10px 15px;
                color: #f1f5f9;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 2px solid rgba(96, 165, 250, 0.6);
                background-color: rgba(30, 41, 59, 0.8);
            }
        """)
        
        # 密码输入
        password_label = QLabel("密码")
        password_label.setStyleSheet("color: #cbd5e1; font-size: 13px; font-weight: 500;")
        
        self.register_password = QLineEdit()
        self.register_password.setPlaceholderText("请输入密码（至少6位）")
        self.register_password.setEchoMode(QLineEdit.Password)
        self.register_password.setFixedHeight(45)
        self.register_password.setStyleSheet("""
            QLineEdit {
                background-color: rgba(15, 23, 42, 0.7);
                border: 2px solid rgba(71, 85, 105, 0.3);
                border-radius: 12px;
                padding: 10px 15px;
                color: #f1f5f9;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 2px solid rgba(96, 165, 250, 0.6);
                background-color: rgba(30, 41, 59, 0.8);
            }
        """)
        
        # 确认密码
        confirm_label = QLabel("确认密码")
        confirm_label.setStyleSheet("color: #cbd5e1; font-size: 13px; font-weight: 500;")
        
        self.register_confirm = QLineEdit()
        self.register_confirm.setPlaceholderText("请再次输入密码")
        self.register_confirm.setEchoMode(QLineEdit.Password)
        self.register_confirm.setFixedHeight(45)
        self.register_confirm.setStyleSheet("""
            QLineEdit {
                background-color: rgba(15, 23, 42, 0.7);
                border: 2px solid rgba(71, 85, 105, 0.3);
                border-radius: 12px;
                padding: 10px 15px;
                color: #f1f5f9;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 2px solid rgba(96, 165, 250, 0.6);
                background-color: rgba(30, 41, 59, 0.8);
            }
        """)
        
        # 注册按钮
        register_btn = QPushButton("注册")
        register_btn.setFixedHeight(50)
        register_btn.setCursor(Qt.PointingHandCursor)
        register_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #10b981, stop:1 #059669);
                border: 2px solid rgba(16, 185, 129, 0.3);
                border-radius: 12px;
                color: white;
                font-size: 16px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #34d399, stop:1 #10b981);
                border: 2px solid rgba(52, 211, 153, 0.5);
                box-shadow: 0 0 25px rgba(16, 185, 129, 0.4);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #059669, stop:1 #047857);
            }
        """)
        register_btn.clicked.connect(self._on_register)
        
        layout.addWidget(username_label)
        layout.addWidget(self.register_username)
        layout.addSpacing(5)
        layout.addWidget(email_label)
        layout.addWidget(self.register_email)
        layout.addSpacing(5)
        layout.addWidget(password_label)
        layout.addWidget(self.register_password)
        layout.addSpacing(5)
        layout.addWidget(confirm_label)
        layout.addWidget(self.register_confirm)
        layout.addSpacing(15)
        layout.addWidget(register_btn)
        layout.addStretch()
        
        return widget
    
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
        
        # 登录成功
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
            f"账户 {username} 注册成功！\n请切换到登录页面进行登录。"
        )
        
        # 切换到登录页面
        self.tab_widget.setCurrentIndex(0)
        self.login_username.setText(username)
        self.login_password.clear()
    
    def resizeEvent(self, event) -> None:
        """窗口大小改变时重新设置背景"""
        super().resizeEvent(event)
        self._set_background()

