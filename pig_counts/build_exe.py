"""
打包脚本 - 将猪只计数系统打包为exe程序
使用PyInstaller打包
"""

import os
import sys
import subprocess
from pathlib import Path

def build_exe():
    """构建exe程序"""
    
    print("=" * 60)
    print("  猪只计数系统 - 打包为EXE程序")
    print("=" * 60)
    print()
    
    # 检查PyInstaller是否安装
    try:
        import PyInstaller
        print("✓ PyInstaller 已安装")
    except ImportError:
        print("⚠ PyInstaller 未安装，正在安装...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("✓ PyInstaller 安装完成")
    
    print()
    print("开始打包...")
    print()
    
    # PyInstaller 打包命令
    cmd = [
        "pyinstaller",
        "--name=猪只计数系统",
        "--windowed",  # 不显示控制台
        "--onefile",   # 单文件模式
        "--icon=NONE", # 如果有图标可以指定
        "--add-data=best.pt;.",  # 添加模型文件
        "--add-data=yolo11n.pt;.",  # 添加备用模型
        "--add-data=bg.jpg;.",  # 添加背景图片
        "--add-data=login_window.py;.",  # 添加登录模块
        "--hidden-import=PyQt5",
        "--hidden-import=cv2",
        "--hidden-import=numpy",
        "--hidden-import=ultralytics",
        "--collect-all=ultralytics",  # 收集ultralytics所有依赖
        "--noconsole",  # 无控制台
        "pyqt5_detection_viewer.py"
    ]
    
    try:
        # 执行打包
        subprocess.check_call(cmd)
        
        print()
        print("=" * 60)
        print("✓ 打包完成！")
        print("=" * 60)
        print()
        print("生成的文件位置：")
        print(f"  - dist/猪只计数系统.exe")
        print()
        print("使用方法：")
        print("  1. 将 dist/猪只计数系统.exe 复制到目标位置")
        print("  2. 双击运行即可")
        print()
        print("注意事项：")
        print("  - 首次运行会自动创建users.json和remember.json")
        print("  - 模型文件已打包在exe中")
        print("  - 背景图片已打包在exe中")
        print()
        
    except subprocess.CalledProcessError as e:
        print()
        print(f"✗ 打包失败: {e}")
        print()
        print("请检查：")
        print("  1. 所有依赖包是否已安装")
        print("  2. 文件路径是否正确")
        print("  3. 是否有足够的磁盘空间")
        return False
    
    return True


if __name__ == "__main__":
    build_exe()
    input("\n按Enter键退出...")

