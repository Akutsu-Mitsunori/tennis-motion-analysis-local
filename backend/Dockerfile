FROM python:3.10-slim

# 作業ディレクトリ
WORKDIR /app

# システム依存ライブラリのインストール（OpenCV / PyTorch等で必要なもの）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    gcc \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# pipの更新と requirements.txt のインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# YOLOモデルの事前ダウンロード（コンテナ起動時の高速化）
RUN python3 -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

# ソースコードのコピー
COPY . .

# バックエンドポート
EXPOSE 8000

# ローカル開発用：--reload 付きで uvicorn を起動（コード変更が即時反映されます）
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]