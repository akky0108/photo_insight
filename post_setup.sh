#!/bin/bash

# Conda環境の有効化
echo "Activating Conda environment: photo_eval_env"
source ~/anaconda3/bin/activate photo_eval_env

# プロジェクトディレクトリの追加
echo "Adding project directory to Python path: /home/mluser/photo_insight"
export PYTHONPATH=$PYTHONPATH:/home/mluser/photo_insight

# CUDAのインストール確認
echo "Checking CUDA version..."
nvcc --version

# NumPyのバージョン確認とダウングレード（必要なら）
echo "Downgrading NumPy to version <2.0..."
pip install numpy<2.0

# PyTorchのCUDA対応確認
echo "Checking PyTorch CUDA availability..."
python -c "import torch; print('CUDA is available.' if torch.cuda.is_available() else 'CUDA is not available. Using CPU.')"

# InsightFaceのGPU確認
echo "Checking if InsightFace is using GPU..."
python -c "import onnxruntime as ort; print('Using GPU for inference.' if 'CUDAExecutionProvider' in ort.get_available_providers() else 'Using CPU for inference.')"

# GPU使用状況の確認
echo "Checking GPU usage with nvidia-smi..."
nvidia-smi

# Conda環境内でインストールされているパッケージの確認
echo "Checking installed packages..."
conda list

echo "Setup checks completed."
