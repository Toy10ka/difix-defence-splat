# README_setup — Difix + gsplat + Difix3D 環境メモ

このファイルは **自分用の環境構築メモ** であり、

- Docker での環境構築
- gsplat / Difix3D の追加セットアップ
- こちらで加えたコード変更
- 実際に回したコマンド例

をまとめたものです。

---

## 0. やりたいこと

- 公式 Difix リポジトリをベースに、
  - **2D Difix（SD-Turbo ベースの single-step diffusion）** をそのまま使い、
  - **gsplat backend で Difix3D / Difix3D+ パイプラインを動かす**。
- まずは「既存 Difix モデルを使った Difix3D パイプラインが問題なく動くこと」を確認する。
- 将来的には
  - 自前データセットで Difix の再ファインチューニング
  - その Difix を使った Difix3D+
  を行う予定。

---

## 1. Docker イメージ構成

### 1.1 目的

- **再現性のある GPU 環境**を Docker で作る。
- ベースは **PyTorch 2.1.2 + CUDA 11.8 + cuDNN8**（PyTorch 公式イメージ）。
- 公式 Difix リポジトリには Dockerfile が無いので、**このリポジトリ側で追加**している。

### 1.2 Dockerfile 要点

- ベースイメージにすでに `torch==2.1.2+cu118` が入っているが、diffusers / transformers 周りが入るときに torch を 2.9.1 + cu128 に引き上げられるため、`force-reinstall` で明示的に固定している。
- `examples/gsplat/requirements.txt` で `fused_ssim` など C++/CUDA 拡張が入るため、  
  `--no-build-isolation` を付けて「現在の torch (2.1.2+cu118) を使ってビルド」するようにしている。
- xformers は 2.9.1 用の壊れたビルドが入っていることがあるので、一旦 `pip uninstall` しておく。  
  （必要になれば、torch 2.1.2 用に別途ビルドする）

### 1.3 イメージビルド

    # リポジトリルートで
    docker build -t difix3d-gsplat .

---

## 2. コードの変更点（IP-Adapter / xformers 周り）

### 2.1 背景

- `src/pipeline_difix.py` は diffusers の `IPAdapterMixin` を継承しており、
  - `diffusers.loaders.ip_adapter` → `xformers` → `torch.utils.checkpoint`  
    という依存がある。
- gsplat / fused_ssim 周りを試す過程で、一度 torch が **2.9.1+cu128** に上がり、
  それに合わせてビルドされた xformers が入っていた。
- その後 CUDA 11.8 と揃えるために torch を **2.1.2+cu118** に戻した結果、
  - xformers が torch の ABI と合わず import エラーを起こすようになった。
- ひとまず Difix3D パイプラインを動かすことが目的なので、  
  **IP-Adapter 依存を暫定的に切る**ことにした。

### 2.2 変更内容

`src/pipeline_difix.py` の import とクラス宣言を修正。

修正前:

```python
    from diffusers.loaders import (
        FromSingleFileMixin,
        IPAdapterMixin,
        LoraLoaderMixin,
        TextualInversionLoaderMixin,
    )

    class DifixPipeline(
        FromSingleFileMixin,
        IPAdapterMixin,
        LoraLoaderMixin,
        TextualInversionLoaderMixin,
    ):
        ...
```

修正後:
```python
    from diffusers.loaders import (
        FromSingleFileMixin,
        LoraLoaderMixin,
        TextualInversionLoaderMixin,
    )

    class DifixPipeline(
        FromSingleFileMixin,
        LoraLoaderMixin,
        TextualInversionLoaderMixin,
    ):
        ...
```

- これで `diffusers.loaders.ip_adapter` → `xformers` の import チェーンが切れる。
- DIFIX 論文で使っている「画像＋ref_image＋テキストで 1step 変換」の機能には影響しない。
- IP-Adapter を使った高度な機能（画像プロンプトによる制御）はこの環境では使えない。  
  （将来やるなら、torch 2.9 系 + xformers を別コンテナで用意する）

### 2.3 xformers について

- 現在の Docker イメージでは xformers をアンインストール済み:

```bash
    pip uninstall -y xformers
```

- 将来的に VRAM/速度のために使いたくなった場合は、  
  このイメージの中で

```bash
    export TORCH_CUDA_ARCH_LIST="8.9"   # RTX 4090
    pip install xformers --no-build-isolation
```

  のようにして **torch 2.1.2+cu118 用にビルドし直す**。

---

## 3. Python 依存（まとめ）

- 公式 `requirements.txt`（DIFIX 2D 用）
  - torch 系以外を `pip install`。
- gsplat 本体

```bash
    pip install gsplat
```

- Difix3D(gsplat) 用依存

```bash
    pip install -r examples/gsplat/requirements.txt --no-build-isolation
```

  で `pycolmap`, `viser`, `nerfview`, `fused_ssim` などが入る。

---

## 4. データ準備（MipNeRF360 の例）

### 4.1 元データ構造（例）

```text
    Data/data/360_v2/counter/
        images/
        images_2/
        images_4/
        images_4clean/
        images_4origin/
        images_8/
        sparse/0/{cameras.bin, images.bin, points3D.bin,...}
        poses_bounds.npy
        resized.txt
```

### 4.2 Difix3D(gsplat) の Parser が期待する構造

```text
    /workspace/data/360_v2/counter/
        colmap/
          sparse/
            0/
              cameras.bin
              images.bin
              points3D.bin
              ...
        images/
        images_2/
        images_4/
        images_8/
```

- `database.db` は無くてもよい（`SceneManager` は `cameras.bin`, `images.bin`, `points3D.bin` だけ読んでいる）。

### 4.3 実際にやった変換

WSL 側で`colmap/sparse/0/...`になるように配置

---

## 5. Difix3D(gsplat) 実行コマンド

### 5.1 コンテナ起動

```bash
    docker run --gpus all --rm -it \
      -v /path/to/data:/workspace/data \
      -v /path/to/outputs:/workspace/outputs \
      difix3d-gsplat \
      /bin/bash
```

ここで `/path/to/data/360_v2/counter` が上で整えたディレクトリ。

### 5.2 環境変数のセット

コンテナ内 `/workspace` で:

```bash
    export PYTHONPATH=/workspace:${PYTHONPATH}

    export SCENE_ID=counter
    export DATA_DIR=/workspace/data/360_v2
    export OUTPUT_DIR=/workspace/outputs/difix3d/gsplat/${SCENE_ID}
```

### 5.3 学習コマンド（フル設定）

```bash
    CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
        --data_dir ${DATA_DIR}/${SCENE_ID} \
        --data_factor 4 \
        --result_dir ${OUTPUT_DIR} \
        --test_every 8 \
        --no-normalize-world-space
```

- `data_dir` : `DATA_DIR/SCENE_ID`（中に `colmap/sparse/0` と `images_*` がある）
- `data_factor` : 4（`images_4` を使う想定）
- `result_dir` : ログ・ckpt・レンダ結果がこの下に出る
- `test_every` : 8 ステップごとにテストレンダ・ログ
- `--no-normalize-world-space` : COLMAP 座標系をそのまま使う

### 5.4 ステップ数を軽くしたい場合

デフォルトは `max_steps=60000` でかなり長いので、  
動作確認用に軽くしたいときは追加で:

```bash
    --max_steps 10000 \
    --steps_scaler 0.25
```

などを付ける。

```bash
- `max_steps` : 最大全体ステップ数
- `steps_scaler` : eval/save/fix/max_steps をまとめてスケーリングする係数
```

---

## 6. Devcontainer（ローカル開発用）

VSCode 用に `.devcontainer/devcontainer.json` を追加している（ローカル専用）。

構成の要点:

```json
- build:
  - `context: ".."`
  - `dockerfile: "../Dockerfile"`
- `runArgs: ["--gpus", "all"]`
- `workspaceFolder: "/workspace"`
- `workspaceMount`: ローカルのリポジトリを `/workspace` に bind mount
- Python 拡張が `/opt/conda/bin/python` を使うように設定
```
---

## 7. 今後やりたいことメモ

- 自前データで **DIFIX (2D) の再ファインチューニング**:
  - `src/train_difix.py` + JSON データセット（公式 README の形式）。
  - 学習時に `--enable_xformers_memory_efficient_attention` を使いたくなったら、  
    そのタイミングで torch 2.1.2 用 xformers をビルドする。
- ファインチューニング済み DIFIX を
  - `src/inference_difix.py` で 2D 推論
  - Difix3D+ の post-render フェーズに差し替え
- IP-Adapter を使った実験をする場合は、
  - torch 2.9 + CUDA12 系の別 Dockerfile を切って、
  - `pipeline_difix.py` の IPAdapterMixin を復活させる想定。

---

## 8. その他メモ

- `database.db` は無くても良い（Parser は sparse モデルの `.bin` を直接読む）。
- 実際に回したときの VRAM 使用量（RTX4090 24GB, counter シーン）:
  - Gaussians ~100万 + DIFIX + Fix フェーズ → 約 18GB 使用。
  - 24GB のうち 5〜6GB 余っていたので、OOM のリスクはそこまで高くない。
- 3k step 時点でもかなり見た目が良くなっているので、
  - 試行錯誤フェーズでは `max_steps` を削っても十分。


