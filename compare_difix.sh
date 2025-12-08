# PS用difix 0~10000ステップFTモデル + 論文の規定モデルまで段階的に実験

#!/usr/bin/env bash
# エラーが出たら修了
set -e

CKPT_DIR="outputs/difix/train_ps/checkpoints"
INPUT_IMAGE="test7979.JPG"      # 処理したい画像
TIMESTEP=199

# 入力画像名のベースと拡張子を分解
IMG_BASE=$(basename "${INPUT_IMAGE}")          # "test7979.JPG"
IMG_NAME="${IMG_BASE%.*}"                      # "test7979"
IMG_EXT="${IMG_BASE##*.}"                      # "JPG"

# 結果をまとめるフォルダ
ROOT_OUT="outputs/difix_199_${IMG_NAME}"
TMP_OUT="outputs/difix_tmp"

mkdir -p "${ROOT_OUT}"
mkdir -p "${TMP_OUT}"

# ckpt を数値ソートしてからループ
#   model_0.pkl, model_1.pkl, model_1001.pkl, ... → num=0,1,1001,... でソート
ls "${CKPT_DIR}"/model_*.pkl | while read -r ckpt; do
  base=$(basename "${ckpt}")          # e.g. model_1001.pkl
  num=${base#model_}                  # "1001.pkl"
  num=${num%.pkl}                     # "1001"

  echo "${num} ${ckpt}"
done | sort -n | while read -r num ckpt; do
  # 出力ファイル名用のステップ数タグ
  #   model_0.pkl   → tag=0
  #   model_1.pkl   → tag=1
  #   model_1001.pkl→ tag=1000
  if [[ "${num}" -eq 0 ]]; then
    tag=0
  elif [[ "${num}" -eq 1 ]]; then
    tag=1
  else
    tag=$((num - 1))
  fi

  out_file="${ROOT_OUT}/difix_199_${tag}.${IMG_EXT}"

  echo "==> ckpt: ${ckpt}  ->  ${out_file}"

  # 一時ディレクトリに出力させる
  python src/inference_difix.py \
    --model_path "${ckpt}" \
    --input_image "${INPUT_IMAGE}" \
    --prompt "remove degradation" \
    --output_dir "${TMP_OUT}" \
    --timestep ${TIMESTEP}

  # inference_difix.py は "出力先フォルダ/入力画像のbasename" で保存するので、
  # それをまとめフォルダにリネームして移動
  mv "${TMP_OUT}/${IMG_BASE}" "${out_file}"
done

echo "Done. All outputs are under: ${ROOT_OUT}"

# 論文の規定DIFIXでの評価

# …各 ckpt を回したあとに追加
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

pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
pipe = pipe.to("cuda")

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
