# README_setup — Difix + gsplat + Difix3D 環境メモ

このファイルは **自分用の環境構築メモ** であり、

- Docker での環境構築
- gsplat / Difix3D / PoisonSplat 用 DIFIX 再学習 の追加セットアップ
- こちらで加えたコード変更
- 実際に回したコマンド例

をまとめたものです。

---

## 0. やりたいこと


- 公式 Difix リポジトリをベースに、2D Difix（SD-Turbo ベースの single-step diffusion）と gsplat backend を用いて Difix3D / Difix3D+ を動かす。  
- Quickstart の DIFIX（`nvidia/difix`, `nvidia/difix_ref`）は論文の 4 条件で事前学習済みモデル。  
- そのうえで PoisonSplat 汚染画像＋元画像を使って DIFIX を再FTし、PoisonSplat 専用 DIFIX（PS-DIFIX）を作る。  
- 将来的には「規定 DIFIX（4条件）＋PS」を学習した DIFIX（reg→PS 再FT）も試す予定。

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

### 2.4 Python 依存（まとめ）

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

## 3. データ準備（MipNeRF360 の例）

### 3.1 元データ構造（例）

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

### 3.2 Difix3D(gsplat) の Parser が期待する構造

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

### 3.3 実際にやった変換

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

## 7. その他メモ

- `database.db` は無くても良い（Parser は sparse モデルの `.bin` を直接読む）。
- 実際に回したときの VRAM 使用量（RTX4090 24GB, counter シーン）:
  - Gaussians ~100万 + DIFIX + Fix フェーズ → 約 18GB 使用。
  - 24GB のうち 5〜6GB 余っていたので、OOM のリスクはそこまで高くない。
- 3k step 時点でもかなり見た目が良くなっているので、
  - 試行錯誤フェーズでは `max_steps` を削っても十分。


---

## 8. PoisonSplat 用 DIFIX ファインチューニング

### 8.1 データ配置

コンテナ内 `/workspace/data` の下に、clean と poisoned を同じ構造で置く。

```text
/workspace/data/360_v2/<scene>/images/*.JPG
/workspace/data/poisoned-datasets/MIP_Nerf_360_eps16/<scene>/images/*.JPG
```

scene: bicycle, bonsai, counter, garden, kitchen, room, stump, flowers, treehill など。

### 8.2 `make_data_json.py` で `data_poison_splat.json` を生成

設定例:

```python
POISON_ROOT = "/workspace/data/poisoned-datasets/MIP_Nerf_360_eps16"
CLEAN_ROOT  = "/workspace/data/360_v2"

SCENES = [
    "bicycle",
    "bonsai",
    "counter",
    "garden",
    "kitchen",
    "room",
    "stump",
    "flowers",
    "treehill",
]

TRAIN_RATIO = 0.8
OUTPUT_JSON = "/workspace/data/data_poison_splat.json"

def to_json_path(abspath: str) -> str:
    # /workspace からの相対にして JSON に書く
    rel = os.path.relpath(abspath, "/workspace")
    return rel.replace("\", "/")  # "data/..."
```

実行:

```bash
python make_data_json.py
```

これで `data/data_poison_splat.json` ができる。  
中身は

```json
{
  "train": {
    "scene_train_xxxx": {
      "image": "data/poisoned-datasets/.../images/xxx.JPG",
      "target_image": "data/360_v2/.../images/xxx.JPG",
      "prompt": "remove degradation"
    }
  },
  "test": { ... }
}
```

カスタマイズメモ:

- パスを変えたい → `POISON_ROOT`, `CLEAN_ROOT`, `to_json_path` を自分の環境に合わせて変更。  
- train/test 比を変えたい → `TRAIN_RATIO` を 0.7 や 0.9 に変更。  
- scene ごとの枚数制限 → `collect_pairs_for_scene` の戻り値 `pairs` を途中で切る（`pairs = pairs[:200]` 等）。

### 8.3 `dataset.py` の修正（img_t 未定義バグ）

元の `PairedDataset.__getitem__` では `img_t` / `output_t` / `ref_t` が未定義のまま使われていたので修正。

修正後の該当部分:

```python
try:
    input_img = Image.open(input_img)
    output_img = Image.open(output_img)
except Exception:
    print("Error loading image:", input_img, output_img)
    return self.__getitem__((idx + 1) % len(self))  # 念のため範囲内

# 入力（汚染画像）
img_t = F.to_tensor(input_img)
img_t = F.resize(img_t, self.image_size)
img_t = F.normalize(img_t, mean=[0.5], std=[0.5])

# ターゲット（クリーン画像）
output_t = F.to_tensor(output_img)
output_t = F.resize(output_t, self.image_size)
output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

# 参照画像があれば
if ref_img is not None:
    ref_img = Image.open(ref_img)
    ref_t = F.to_tensor(ref_img)
    ref_t = F.resize(ref_t, self.image_size)
    ref_t = F.normalize(ref_t, mean=[0.5], std=[0.5])

    img_t = torch.stack([img_t, ref_t], dim=0)
    output_t = torch.stack([output_t, ref_t], dim=0)
else:
    img_t = img_t.unsqueeze(0)
    output_t = output_t.unsqueeze(0)
```

### 8.4 `train_difix.py` の val シャッフル修正

val データセットのシャッフルで存在しない属性 `img_names` を参照していたので、

```python
random.Random(42).shuffle(dataset_val.img_names)
```

を

```python
random.Random(42).shuffle(dataset_val.img_ids)
```

に変更。

### 8.5 DataLoader の shared memory 対策

`accelerate launch` 中に以下のエラーが出ることがある:

```text
ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).
RuntimeError: DataLoader worker (pid xxxx) is killed by signal: Bus error.
```

対策:

- 一番簡単 → `--dataloader_num_workers 0` を指定してシングルプロセスにする。  
- ちゃんと並列化したい → devcontainer / docker run の `--shm-size` を増やす。

devcontainer 例:

```json
"runArgs": [
  "--gpus", "all",
  "--shm-size=16g"
]
```

docker run 例:

```bash
docker run --gpus all --shm-size=16g ...
```

### 8.6 `train_difix.py` の解像度 / ステップ周りの修正

(1) Dataset の解像度を `args.resolution` に揃える。

```python
# 元
dataset_train = PairedDataset(dataset_path=args.dataset_path, split="train", tokenizer=net_difix.tokenizer)
dl_train = DataLoader(dataset_train, ...)
dataset_val = PairedDataset(dataset_path=args.dataset_path, split="test", tokenizer=net_difix.tokenizer)
```

```python
# 修正後
dataset_train = PairedDataset(
    dataset_path=args.dataset_path,
    split="train",
    height=args.resolution,
    width=args.resolution,
    tokenizer=net_difix.tokenizer,
)
dl_train = DataLoader(
    dataset_train,
    batch_size=args.train_batch_size,
    shuffle=True,
    num_workers=args.dataloader_num_workers,
)

dataset_val = PairedDataset(
    dataset_path=args.dataset_path,
    split="test",
    height=args.resolution,
    width=args.resolution,
    tokenizer=net_difix.tokenizer,
)
random.Random(42).shuffle(dataset_val.img_ids)
dl_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)
```

(2) 初期状態 `model_0.pkl` の保存。

```python
if accelerator.is_main_process:
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "model_0.pkl")
    print("Saving initial (untrained) checkpoint:", ckpt_path)
    save_ckpt(net_difix, optimizer, ckpt_path)
```

(3) `max_train_steps` でループを止める。

```python
for epoch in range(args.num_training_epochs):
    for step, batch in enumerate(dl_train):
        if global_step >= args.max_train_steps:
            break
        ...
        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1
            ...
    if global_step >= args.max_train_steps:
        break
```

### 8.7 PoisonSplat DIFIX 学習コマンド

wandb を無効化してから accelerate を起動:

```bash
export WANDB_DISABLED=true

accelerate launch --mixed_precision=bf16 src/train_difix.py     --output_dir ./outputs/difix/train_ps     --dataset_path data/data_poison_splat.json     --max_train_steps 10000     --resolution 512     --learning_rate 2e-5     --train_batch_size 1     --dataloader_num_workers 0     --checkpointing_steps 1000     --eval_freq 1000     --viz_freq 100     --lambda_lpips 1.0     --lambda_l2 1.0     --lambda_gram 1.0     --gram_loss_warmup_steps 2000     --report_to "wandb"     --tracker_project_name "difix_ps"     --tracker_run_name "train_ps"     --timestep 199
```

`--timestep 199` は

- 学習時の条件付け（LoRA がどのノイズ分布にフィットするか）  
- 推論時のノイズレベル（どの timestep から 1step で戻るか）  

の両方を決める重要パラメータ。

---

## 9. ckpt と `compare_difix.sh` による視覚比較

学習後:

```text
outputs/difix/train_ps/checkpoints/
  model_0.pkl      (初期)
  model_1.pkl
  model_1001.pkl   (~1000 step)
  model_2001.pkl   (~2000 step)
  ...
  model_15001.pkl  (~15000 step)
```

単一画像に対して各 ckpt の出力をまとめるために `compare_difix.sh` を作成。

```bash
#!/usr/bin/env bash
set -e

CKPT_DIR="outputs/difix/train_ps/checkpoints"
INPUT_IMAGE="test7979.JPG"
TIMESTEP=199

IMG_BASE=$(basename "${INPUT_IMAGE}")
IMG_NAME="${IMG_BASE%.*}"
IMG_EXT="${IMG_BASE##*.}"

ROOT_OUT="outputs/difix_199_${IMG_NAME}"
TMP_OUT="outputs/difix_tmp"

mkdir -p "${ROOT_OUT}"
mkdir -p "${TMP_OUT}"

ls "${CKPT_DIR}"/model_*.pkl | while read -r ckpt; do
  base=$(basename "${ckpt}")
  num=${base#model_}
  num=${num%.pkl}
  echo "${num} ${ckpt}"
done | sort -n | while read -r num ckpt; do
  if [[ "${num}" -eq 0 ]]; then
    tag=0
  elif [[ "${num}" -eq 1 ]]; then
    tag=1
  else
    tag=$((num - 1))
  fi

  out_file="${ROOT_OUT}/difix_199_${tag}.${IMG_EXT}"

  echo "==> ckpt: ${ckpt}  ->  ${out_file}"

  python src/inference_difix.py     --model_path "${ckpt}"     --input_image "${INPUT_IMAGE}"     --prompt "remove degradation"     --output_dir "${TMP_OUT}"     --timestep ${TIMESTEP}

  mv "${TMP_OUT}/${IMG_BASE}" "${out_file}"
done
```

さらに、規定 DIFIX (`nvidia/difix`) の出力も同じフォルダに保存:

```bash
python - << 'EOF'
from src.pipeline_difix import DifixPipeline
from diffusers.utils import load_image
import os

INPUT_IMAGE = "test7979.JPG"
TIMESTEP = 199
IMG_BASE = os.path.basename(INPUT_IMAGE)
IMG_NAME = IMG_BASE.split(".")[0]
OUT_DIR = f"outputs/difix_199_{IMG_NAME}"
OUT_PATH = os.path.join(OUT_DIR, "difix_199_reg.JPG")

pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True).to("cuda")
img = load_image(INPUT_IMAGE)
prompt = "remove degradation"

out = pipe(
    prompt,
    image=img,
    num_inference_steps=1,
    timesteps=[TIMESTEP],
    guidance_scale=0.0,
).images[0]

os.makedirs(OUT_DIR, exist_ok=True)
out.save(OUT_PATH)
print("saved:", OUT_PATH)
EOF
```

---

## 10. 現時点の観察と今後

- PS-DIFIX は脚や床の PoisonSplat 縦スジノイズを強く抑える。  
- 規定 DIFIX は構造保持が上手で、特に天板や背景の立体感が残りやすい。  
- 1000〜3000 step で大きく改善し、4000〜8000 step はほぼ plateau。10k step はオーバーキル気味だが悪化はしない。  

今後やりたいこと（メモ）:

- timestep を変えた PS-DIFIX（例: τ=100, 200, 400）を複数作り、PoisonSplat に対する構造 vs 毒除去のバランスを調べる。  
- 規定 DIFIX (`nvidia/difix`) を初期値として PS データで追加学習するパスを `train_difix.py` に作る（4条件＋PS の良いとこ取り）。  
- Difix3D / Difix3D+ パイプラインに PS-DIFIX / reg-DIFIX / reg→PS-DIFIX を組み込み、3D レンダで比較する。  
- ROI ベース混在は MVC（multi-view consistency）の観点と PoisonSplat の攻撃モデルの観点から本命にはしない（全画素処理のモデル同士で比較する）。

