# CUDA 12.1 + cuDNN8 (nvcc: デスク124，ノート130，計算機サーバ122)
# FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 
# (ないとは思うけどカスタムcudaカーネルをnvccでビルドするかもしれないため,121やめてtorchの118想定にあわせる)
# -devel: -runtime の中身＋ CUDA Toolkit（nvcc・ヘッダ・静的ライブラリ・開発ツール）
# -runtime: 実行に必要なランタイム（libcuda(ドライバ側から見える), libcudart, cuBLAS/cuDNN/NCCL 等の共有ライブラリ）のみ

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1

# 基本ツール
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev git ffmpeg libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3 1 \
    && python -m pip install --upgrade pip setuptools wheel


# ---- PyTorch 2.7.1 + cu118（公式 previous-versions）----
RUN pip install --no-cache-dir \
      torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
      --index-url https://download.pytorch.org/whl/cu118

WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt

# requirements.txt から torch/torchvision/torchaudio は除外して残りを入れる
RUN awk '!/^(torch|torchvision|torchaudio)(\[|==|>=|<=|~=|>|<| |$)/' \
      /workspace/requirements.txt > /tmp/req.nocuda.txt \
 && pip install --no-cache-dir -r /tmp/req.nocuda.txt

COPY . /workspace
CMD ["/bin/bash"]