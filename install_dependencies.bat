@echo off
chcp 65001

echo ==========================================
echo See-Through Plugin Dependency Installer
echo ==========================================
echo.

REM 获取WebUI的Python路径
set PYTHON_PATH=D:\ai\sd-webui-forge-neo-v2\system\python\python.exe

if not exist "%PYTHON_PATH%" (
    echo Error: Python not found at %PYTHON_PATH%
    echo Please check your WebUI installation path
    pause
    exit /b 1
)

echo Using Python: %PYTHON_PATH%
echo.

REM 安装psd-tools
echo Installing psd-tools...
"%PYTHON_PATH%" -m pip install psd-tools

if %errorlevel% neq 0 (
    echo Failed to install psd-tools. Please install manually:
    echo   pip install psd-tools
    echo.
    pause
    exit /b 1
)

REM 安装pycocotools
echo Installing pycocotools...
"%PYTHON_PATH%" -m pip install pycocotools

if %errorlevel% neq 0 (
    echo.
    echo Failed to install pycocotools. Trying alternative method...
    echo.
    
    REM 尝试使用conda安装（如果可用）
    where conda >nul 2>&1
    if %errorlevel% eq 0 (
        echo Found conda, trying conda install...
        conda install -c conda-forge pycocotools -y
    ) else (
        echo Conda not found. Please install pycocotools manually:
        echo   pip install pycocotools
        echo.
        echo Or if you have Visual C++ Build Tools installed:
        echo   pip install pycocotools-windows
    )
)

echo.
echo ==========================================
echo Installation completed!
echo ==========================================
echo.
echo Please restart WebUI to apply changes.
echo.
pause
