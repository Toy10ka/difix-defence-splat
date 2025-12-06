# ============================================================
# Difix + gsplat + Difix3D 用 Dockerfile
#  - ベース: PyTorch 2.1.2 + CUDA 11.8 + cuDNN8 (公式イメージ)
#  - 目的: 公式 DIFIX リポジトリ + gsplat backend で
#          examples/gsplat/simple_trainer_difix3d.py がそのまま動く環境
# ============================================================

FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# ---------- 基本ツール ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
      git ffmpeg libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && /opt/conda/bin/python -m pip install --upgrade pip setuptools wheel

# ---------- 作業ディレクトリ ----------
WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt

# ---------- Python 依存ライブラリ ----------
# requirements.txt から torch/torchvision/torchaudio は除外して残りを入れる
RUN awk '!/^(torch|torchvision|torchaudio)(\[|==|>=|<=|~=|>|<| |$)/' \
      /workspace/requirements.txt > /tmp/req.nocuda.txt \
 && pip install --no-cache-dir -r /tmp/req.nocuda.txt

 # ソース一式をコピー（examples/gsplat/requirements.txt が使えるようになる）
 COPY . /workspace

 # gsplat環境を構築
 RUN pip install gsplat \
 && pip install --force-reinstall \
      torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
      --index-url https://download.pytorch.org/whl/cu118 \
 && pip install -r examples/gsplat/requirements.txt --no-build-isolation \
 && pip uninstall -y xformers

# ---------- Python パス設定 ----------
# examples.utils を見えるように /workspace を PYTHONPATH に追加
ENV PYTHONPATH=/workspace:${PYTHONPATH}

# ---------- デフォルトコマンド ----------
CMD ["/bin/bash"]