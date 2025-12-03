"""
分析 PyInstaller 打包后的文件体积分布
帮助识别哪些模块占用了最多空间
"""

import os
import sys
from pathlib import Path
from collections import defaultdict


def get_size_mb(size_bytes):
    """将字节转换为 MB"""
    return size_bytes / (1024 * 1024)


def analyze_directory(directory_path, top_n=20):
    """分析目录中文件的大小分布"""
    
    if not os.path.exists(directory_path):
        print(f"错误：目录不存在 - {directory_path}")
        return
    
    # 收集所有文件信息
    file_sizes = []
    category_sizes = defaultdict(int)
    total_size = 0
    
    print(f"\n正在分析目录: {directory_path}\n")
    
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                size = os.path.getsize(file_path)
                total_size += size
                
                # 记录文件信息
                relative_path = os.path.relpath(file_path, directory_path)
                file_sizes.append((relative_path, size))
                
                # 按文件扩展名分类
                ext = os.path.splitext(file)[1].lower()
                if not ext:
                    ext = '(无扩展名)'
                category_sizes[ext] += size
                
                # 按模块名称分类
                if 'torch' in file.lower():
                    category_sizes['[torch 相关]'] += size
                elif 'cv2' in file.lower() or 'opencv' in file.lower():
                    category_sizes['[opencv 相关]'] += size
                elif 'qt' in file.lower() or 'pyqt' in file.lower():
                    category_sizes['[PyQt 相关]'] += size
                elif 'numpy' in file.lower():
                    category_sizes['[numpy 相关]'] += size
                elif 'ultralytics' in file.lower():
                    category_sizes['[ultralytics 相关]'] += size
                    
            except Exception as e:
                print(f"警告：无法读取文件 {file_path}: {e}")
    
    # 排序
    file_sizes.sort(key=lambda x: x[1], reverse=True)
    
    # 输出总体信息
    print("=" * 80)
    print(f"总体积: {get_size_mb(total_size):.2f} MB")
    print(f"文件总数: {len(file_sizes)}")
    print("=" * 80)
    
    # 输出前 N 大文件
    print(f"\n📊 前 {top_n} 大文件:")
    print("-" * 80)
    for i, (file_path, size) in enumerate(file_sizes[:top_n], 1):
        percentage = (size / total_size) * 100
        print(f"{i:2d}. {get_size_mb(size):8.2f} MB ({percentage:5.2f}%) - {file_path}")
    
    # 输出按扩展名分类的统计
    print("\n📂 按文件类型分类:")
    print("-" * 80)
    sorted_categories = sorted(category_sizes.items(), key=lambda x: x[1], reverse=True)
    for category, size in sorted_categories[:15]:
        percentage = (size / total_size) * 100
        print(f"{get_size_mb(size):8.2f} MB ({percentage:5.2f}%) - {category}")
    
    # 输出优化建议
    print("\n💡 优化建议:")
    print("-" * 80)
    
    # 分析 DLL 文件
    dll_size = sum(size for ext, size in category_sizes.items() if ext in ['.dll', '.pyd'])
    if dll_size > 100 * 1024 * 1024:  # > 100MB
        print(f"✓ DLL/PYD 文件占用 {get_size_mb(dll_size):.2f} MB")
        print("  建议：检查是否包含了不需要的库（如 MKL, CUDA 等）")
    
    # 分析 PyTorch
    torch_size = category_sizes.get('[torch 相关]', 0)
    if torch_size > 500 * 1024 * 1024:  # > 500MB
        print(f"✓ PyTorch 相关文件占用 {get_size_mb(torch_size):.2f} MB")
        print("  建议：考虑使用 CPU 版本的 PyTorch（如果不需要 GPU）")
    
    # 分析 Python 模块
    pyc_size = category_sizes.get('.pyc', 0)
    if pyc_size > 50 * 1024 * 1024:  # > 50MB
        print(f"✓ .pyc 文件占用 {get_size_mb(pyc_size):.2f} MB")
        print("  建议：使用 --exclude-module 排除不需要的 Python 模块")
    
    print("\n")


def main():
    """主函数"""
    print("=" * 80)
    print(" PyInstaller 打包体积分析工具")
    print("=" * 80)
    
    # 检查常见的打包输出目录
    possible_dirs = [
        'dist',
        'build',
    ]
    
    found_dirs = [d for d in possible_dirs if os.path.exists(d)]
    
    if not found_dirs:
        print("\n未找到 dist 或 build 目录")
        print("请先运行 PyInstaller 打包命令")
        return
    
    # 分析每个目录
    for directory in found_dirs:
        subdirs = [d for d in Path(directory).iterdir() if d.is_dir()]
        
        if subdirs:
            print(f"\n在 {directory} 中找到以下子目录:")
            for i, subdir in enumerate(subdirs, 1):
                size = sum(f.stat().st_size for f in subdir.rglob('*') if f.is_file())
                print(f"{i}. {subdir.name} ({get_size_mb(size):.2f} MB)")
            
            print("\n选择要分析的目录编号（按 Enter 分析所有）:")
            choice = input().strip()
            
            if choice.isdigit() and 1 <= int(choice) <= len(subdirs):
                analyze_directory(str(subdirs[int(choice) - 1]))
            else:
                for subdir in subdirs:
                    analyze_directory(str(subdir))
        else:
            # 直接分析目录本身
            analyze_directory(directory)


if __name__ == '__main__':
    try:
        main()
        print("\n分析完成！")
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    
    input("\n按 Enter 键退出...")

