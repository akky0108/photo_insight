#!/bin/bash

# === 設定 ===
CUDNN_TAR_PATTERN="cudnn-linux-x86_64-9.*_cuda12-archive.tar.xz"
CUDA_PATH="/usr/local/cuda-12.2"
TMP_DIR="cudnn_tmp"

# === cuDNN アーカイブの検索 ===
echo "🔍 cuDNN アーカイブを検索中..."
TAR_FILE=$(ls $CUDNN_TAR_PATTERN 2>/dev/null | head -n 1)

if [[ -z "$TAR_FILE" ]]; then
  echo "❌ cuDNN の .tar.xz ファイルが見つかりません。スクリプトと同じディレクトリに配置してください。"
  exit 1
fi

# === 解凍 ===
echo "📦 解凍中: $TAR_FILE"
mkdir -p "$TMP_DIR"
tar -xf "$TAR_FILE" -C "$TMP_DIR"

ARCHIVE_DIR=$(find "$TMP_DIR" -type d -name "cudnn-*-archive" | head -n 1)

if [[ -z "$ARCHIVE_DIR" ]]; then
  echo "❌ 解凍に失敗しました。アーカイブの構造を確認してください。"
  exit 1
fi

# === ファイルのコピー ===
echo "📁 ファイルをコピー中..."
sudo cp -P $ARCHIVE_DIR/include/* $CUDA_PATH/include/
sudo cp -P $ARCHIVE_DIR/lib/* $CUDA_PATH/lib64/

# === ライブラリキャッシュの更新 ===
echo "🔄 ライブラリキャッシュを更新中..."
echo "$CUDA_PATH/lib64" | sudo tee /etc/ld.so.conf.d/cuda.conf > /dev/null
sudo ldconfig

# === ~/.bashrc への環境変数の追加 ===
echo "📝 ~/.bashrc に環境変数を追加中..."
BASHRC="$HOME/.bashrc"
if ! grep -q "$CUDA_PATH" "$BASHRC"; then
  echo -e "\n# cuDNN 9.x for CUDA 12.2" >> "$BASHRC"
  echo "export PATH=$CUDA_PATH/bin:\$PATH" >> "$BASHRC"
  echo "export LD_LIBRARY_PATH=$CUDA_PATH/lib64:\$LD_LIBRARY_PATH" >> "$BASHRC"
  echo "✅ ~/.bashrc に環境変数を追加しました。"
else
  echo "ℹ️ ~/.bashrc に既に環境変数が設定されています。"
fi

# === 環境変数の即時反映 ===
echo "🔄 環境変数を即時反映中..."
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

# === 確認 ===
echo "📄 インストールされた cuDNN ライブラリ:"
ls -l $CUDA_PATH/lib64/libcudnn.so*

echo "✅ cuDNN 9.x のインストールが完了しました。"
echo "⚠️ 新しいシェルを開くか、'source ~/.bashrc' を実行して環境変数を反映してください。"
