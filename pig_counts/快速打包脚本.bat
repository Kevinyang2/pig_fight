@echo off
chcp 65001 >nul
echo ========================================
echo   PyInstaller 打包体积优化工具
echo ========================================
echo.

:menu
echo 请选择打包方式：
echo.
echo 1. 标准打包 - 目录模式（约 1.2-1.8GB，启动快 1-2秒）
echo 2. 标准打包 - 单文件模式（约 800MB-1.2GB，启动慢 5-10秒）⭐推荐
echo 3. 优化依赖（安装 CPU版PyTorch + opencv-headless）
echo 4. 优化打包 - 单文件 + CPU版（约 500-700MB）⭐⭐推荐
echo 5. 分析当前打包体积
echo 6. 清理之前的打包文件
echo 7. 查看优化指南
echo 8. 退出
echo.
set /p choice=请输入选项 (1-8): 

if "%choice%"=="1" goto standard_dir
if "%choice%"=="2" goto standard_onefile
if "%choice%"=="3" goto optimize_deps
if "%choice%"=="4" goto optimized_onefile
if "%choice%"=="5" goto analyze
if "%choice%"=="6" goto clean
if "%choice%"=="7" goto guide
if "%choice%"=="8" goto end
echo 无效选项，请重新选择。
echo.
goto menu

:standard_dir
echo.
echo [1] 开始标准打包 - 目录模式
echo 特点：启动速度快，但会生成一个文件夹
echo.
call conda activate yolov11
if errorlevel 1 (
    echo 错误：无法激活 conda 环境
    pause
    goto menu
)

pyinstaller --name=可见光成像叶片表面缺陷检测系统 ^
--windowed ^
--add-data=login_window.py;. ^
--add-data=json_analyzer.py;. ^
--add-data=best.pt;. ^
--add-data=bj2.png;. ^
--copy-metadata=torch ^
--copy-metadata=torchvision ^
--copy-metadata=ultralytics ^
--copy-metadata=numpy ^
--copy-metadata=timm ^
--collect-all=ultralytics ^
--collect-data=ultralytics ^
--collect-submodules=timm ^
--hidden-import=cv2 ^
--hidden-import=einops ^
--hidden-import=timm ^
--hidden-import=numpy.core._multiarray_tests ^
--hidden-import=numpy.core._multiarray_umath ^
--hidden-import=ultralytics ^
--hidden-import=ultralytics.models ^
--hidden-import=ultralytics.nn ^
--hidden-import=ultralytics.utils ^
--hidden-import=ultralytics.engine ^
--hidden-import=ultralytics.data ^
--exclude-module=matplotlib ^
--exclude-module=pandas ^
--exclude-module=scipy ^
--exclude-module=tkinter ^
--exclude-module=pytest ^
--exclude-module=IPython ^
--exclude-module=jupyter ^
--exclude-module=sqlite3 ^
--exclude-module=jinja2 ^
--exclude-module=onnxruntime ^
--noconfirm ^
--clean ^
pyqt5_detection_viewer.py

echo.
echo 打包完成！
echo 可执行文件位置: dist\可见光成像叶片表面缺陷检测系统\可见光成像叶片表面缺陷检测系统.exe
echo.
pause
goto menu

:standard_onefile
echo.
echo [2] 开始标准打包 - 单文件模式
echo 特点：只生成一个 exe 文件，但启动较慢
echo.
call conda activate yolov11
if errorlevel 1 (
    echo 错误：无法激活 conda 环境
    pause
    goto menu
)

pyinstaller --name=可见光成像叶片表面缺陷检测系统 ^
--windowed ^
--onefile ^
--add-data=login_window.py;. ^
--add-data=json_analyzer.py;. ^
--add-data=best.pt;. ^
--add-data=bj2.png;. ^
--copy-metadata=torch ^
--copy-metadata=torchvision ^
--copy-metadata=ultralytics ^
--copy-metadata=numpy ^
--copy-metadata=timm ^
--collect-all=ultralytics ^
--collect-data=ultralytics ^
--collect-submodules=timm ^
--hidden-import=cv2 ^
--hidden-import=einops ^
--hidden-import=timm ^
--hidden-import=numpy.core._multiarray_tests ^
--hidden-import=numpy.core._multiarray_umath ^
--hidden-import=ultralytics ^
--hidden-import=ultralytics.models ^
--hidden-import=ultralytics.nn ^
--hidden-import=ultralytics.utils ^
--hidden-import=ultralytics.engine ^
--hidden-import=ultralytics.data ^
--exclude-module=matplotlib ^
--exclude-module=pandas ^
--exclude-module=scipy ^
--exclude-module=tkinter ^
--exclude-module=pytest ^
--exclude-module=IPython ^
--exclude-module=jupyter ^
--exclude-module=sqlite3 ^
--exclude-module=jinja2 ^
--exclude-module=onnxruntime ^
--noconfirm ^
--clean ^
pyqt5_detection_viewer.py

echo.
echo 打包完成！
echo 可执行文件位置: dist\可见光成像叶片表面缺陷检测系统.exe
echo.
pause
goto menu

:optimize_deps
echo.
echo [3] 优化依赖（安装 CPU版PyTorch + opencv-headless）
echo.
echo 这将：
echo 1. 卸载 GPU 版 PyTorch，安装 CPU 版（节省 400-500MB）
echo 2. 卸载 opencv-python，安装 opencv-python-headless（节省 20-30MB）
echo.
echo ⚠️ 警告：如果需要 GPU 加速，请不要执行此操作
echo.
set /p confirm=确认继续吗？(Y/N): 
if /i not "%confirm%"=="Y" goto menu

call conda activate yolov11
echo.
echo 正在卸载旧版本...
pip uninstall -y torch torchvision torchaudio opencv-python

echo.
echo 正在安装优化版本...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python-headless

echo.
echo ✅ 优化完成！
echo 已安装：CPU 版 PyTorch + opencv-python-headless
echo 预计体积减小：420-530 MB
echo.
echo 现在选择选项 2 或 4 进行打包
echo.
pause
goto menu

:optimized_onefile
echo.
echo [4] 优化打包 - 单文件 + CPU版（推荐）
echo 特点：体积小（500-700MB）+ 单个文件 + 所有依赖优化
echo.
echo ⚠️ 注意：需要先执行选项 3 优化依赖
echo.
set /p confirm=已优化依赖？继续打包？(Y/N): 
if /i not "%confirm%"=="Y" goto menu

call conda activate yolov11
if errorlevel 1 (
    echo 错误：无法激活 conda 环境
    pause
    goto menu
)

echo.
echo 正在打包...（预计 20-30 分钟）
echo.

pyinstaller --name=可见光成像叶片表面缺陷检测系统 ^
--onefile ^
--windowed ^
--add-data=login_window.py;. ^
--add-data=json_analyzer.py;. ^
--add-data=best.pt;. ^
--add-data=bj2.png;. ^
--collect-all=ultralytics ^
--collect-data=ultralytics ^
--collect-submodules=timm ^
--hidden-import=cv2 ^
--hidden-import=einops ^
--hidden-import=timm ^
--copy-metadata=torch ^
--copy-metadata=torchvision ^
--copy-metadata=ultralytics ^
--copy-metadata=timm ^
--copy-metadata=numpy ^
--hidden-import=numpy.core._multiarray_tests ^
--hidden-import=numpy.core._multiarray_umath ^
--hidden-import=ultralytics ^
--hidden-import=ultralytics.models ^
--hidden-import=ultralytics.nn ^
--hidden-import=ultralytics.utils ^
--hidden-import=ultralytics.engine ^
--hidden-import=ultralytics.data ^
--exclude-module=matplotlib ^
--exclude-module=pandas ^
--exclude-module=scipy ^
--exclude-module=tkinter ^
--exclude-module=pytest ^
--exclude-module=IPython ^
--exclude-module=jupyter ^
--exclude-module=sqlite3 ^
--exclude-module=jinja2 ^
--exclude-module=pygments ^
--exclude-module=onnxruntime ^
--noconfirm ^
--clean ^
pyqt5_detection_viewer.py

echo.
echo ✅ 打包完成！
echo 可执行文件位置: dist\可见光成像叶片表面缺陷检测系统.exe
echo 预期体积: 500-700 MB
echo 启动速度: 3-5 秒
echo.
pause
goto menu

:analyze
echo.
echo [5] 分析打包体积
echo.
if not exist "dist" (
    echo 错误：未找到 dist 目录，请先进行打包
    pause
    goto menu
)

if not exist "分析打包体积.py" (
    echo 错误：找不到分析工具
    pause
    goto menu
)

call conda activate yolov11
python 分析打包体积.py

pause
goto menu

:clean
echo.
echo [6] 清理打包文件...
echo.
if exist "build" (
    rd /s /q build
    echo 已删除 build 目录
)
if exist "dist" (
    rd /s /q dist
    echo 已删除 dist 目录
)
if exist "*.spec" (
    del /q *.spec
    echo 已删除旧的 spec 文件
)
echo.
echo 清理完成！
echo.
pause
goto menu

:guide
echo.
echo [7] 查看优化指南
echo.
if exist "体积优化终极指南.txt" (
    notepad 体积优化终极指南.txt
) else (
    echo 错误：找不到 体积优化终极指南.txt
)
pause
goto menu

:end
echo.
echo 感谢使用！
echo.
exit /b 0

