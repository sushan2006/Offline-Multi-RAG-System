@echo off
REM GPU install script for RTX 5050
REM Run: install-gpu.bat
REM Edit the CUDA line below if your nvidia-smi shows a different version (cu121, cu126, etc.)

echo ========================================
echo  GPU Setup for RAG Project (RTX 5050)
echo ========================================
echo.

echo Step 1: Installing PyTorch with CUDA 12.4...
echo (Change cu124 to cu121/cu126 if your nvidia-smi shows different)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    echo Failed to install PyTorch. Check your CUDA version with: nvidia-smi
    pause
    exit /b 1
)

echo.
echo Step 2: Installing project dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install requirements.
    pause
    exit /b 1
)

echo.
echo Step 3: Verifying GPU...
python verify_gpu.py
if errorlevel 1 (
    echo GPU verification failed. See SETUP_GPU.md for troubleshooting.
    pause
    exit /b 1
)

echo.
echo Done! Start the backend with: cd backend ^&^& python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
pause
