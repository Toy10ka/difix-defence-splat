# コンテナ内で data.json を作る用のスクリプト

#!/usr/bin/env python
import os
import glob
import json
import random

# ==== 設定ここから ====

# コンテナ内から見た clean / poison のルート
POISON_ROOT = "/workspace/data/poisoned-datasets/MIP_Nerf_360_eps16"
CLEAN_ROOT  = "/workspace/data/360_v2"

# 対象シーン（必要に応じて追加 / 削除）
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

# train/test の比率
TRAIN_RATIO = 0.8

# 出力先（コンテナ内）
OUTPUT_JSON = "/workspace/data/data_poison_splat.json"

# JSON には /workspace からの相対パスを書きたいので、その変換
def to_json_path(abspath: str) -> str:
    rel = os.path.relpath(abspath, "/workspace")
    return rel.replace("\\", "/")  # "data/..." みたいな形

# ==== 設定ここまで ====

# poison-clean のペアを作る関数
def collect_pairs_for_scene(scene: str):
    poison_dir = os.path.join(POISON_ROOT, scene, "images")
    clean_dir  = os.path.join(CLEAN_ROOT,  scene, "images")

    poison_files = sorted(glob.glob(os.path.join(poison_dir, "*.JPG")))
    clean_files  = sorted(glob.glob(os.path.join(clean_dir,  "*.JPG")))

    if not poison_files:
        print(f"[WARN] No poisoned images found for scene '{scene}' in {poison_dir}")
    if not clean_files:
        print(f"[WARN] No clean images found for scene '{scene}' in {clean_dir}")

    poison_map = {os.path.basename(p): p for p in poison_files}
    clean_map  = {os.path.basename(c): c for c in clean_files}

    common_names = sorted(set(poison_map.keys()) & set(clean_map.keys()))

    if not common_names:
        print(f"[WARN] No common filenames for scene '{scene}'")
        return []

    missing_in_poison = sorted(set(clean_map.keys()) - set(poison_map.keys()))
    missing_in_clean  = sorted(set(poison_map.keys()) - set(clean_map.keys()))
    if missing_in_poison:
        print(f"[INFO] {scene}: {len(missing_in_poison)} clean-only files (例: {missing_in_poison[:3]})")
    if missing_in_clean:
        print(f"[INFO] {scene}: {len(missing_in_clean)} poison-only files (例: {missing_in_clean[:3]})")

    pairs = [(poison_map[name], clean_map[name]) for name in common_names]
    print(f"[OK] {scene}: {len(pairs)} pairs (poison & clean)")

    return pairs


def main():
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    train = {}
    test = {}

    # 再現性のために固定 (毎回同じ分割を再現)
    random.seed(0)

    for scene in SCENES:
        pairs = collect_pairs_for_scene(scene)
        if not pairs:
            continue

        random.shuffle(pairs)
        n_train = int(len(pairs) * TRAIN_RATIO)
        train_pairs = pairs[:n_train]
        test_pairs  = pairs[n_train:]

        # キーは "scene_0000" 形式にしておく
        for i, (poison_path, clean_path) in enumerate(train_pairs):
            key = f"{scene}_train_{i:04d}"
            train[key] = {
                "image":        to_json_path(poison_path),
                "target_image": to_json_path(clean_path),
                "prompt":       "remove degradation",
            }

        for i, (poison_path, clean_path) in enumerate(test_pairs):
            key = f"{scene}_test_{i:04d}"
            test[key] = {
                "image":        to_json_path(poison_path),
                "target_image": to_json_path(clean_path),
                "prompt":       "remove degradation",
            }

    data = {"train": train, "test": test}

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] Wrote JSON to {OUTPUT_JSON}")
    print(f"  train samples: {len(train)}")
    print(f"  test  samples: {len(test)}")


if __name__ == "__main__":
    main()