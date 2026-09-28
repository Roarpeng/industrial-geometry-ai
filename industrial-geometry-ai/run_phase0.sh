#!/usr/bin/env bash
# Phase 0 一键启动（V4 执行顺序 [3][4]，需 GPU 已恢复）
# 用法：bash run_phase0.sh [num_images]   默认 5000
set -e
cd "$(dirname "$0")"

NUM=${1:-5000}

echo "=== [1/3] 生成 Phase 0 数据（clean ${NUM} + industrial $((NUM/6))） ==="
python3 -m dataset.synthetic_generator.generate --num "$NUM" --mode clean --out data/phase0_clean --seed 100
python3 -m dataset.synthetic_generator.generate --num "$((NUM/6))" --mode industrial --out data/phase0_ind --seed 101

echo "=== [2/3] 训练（configs/train.yaml，checkpoint 按 object_recall 选优） ==="
python3 -m training.train --config configs/train.yaml \
    --data-dir data/phase0_clean,data/phase0_ind --out out/phase0

echo "=== [3/3] 校准 + 导出 + 验证可视化 ==="
python3 tools/calibrate.py --ckpt out/phase0/best.pth --data data/phase0_clean
python3 -m deployment.export_onnx --ckpt out/phase0/best.pth --out out/phase0/model.onnx
python3 tools/visualize_prediction.py --ckpt out/phase0/best.pth --data data/phase0_ind --num 8 --out out/phase0/vis
python3 -m deployment.benchmark --ckpt out/phase0/best.pth

echo "Phase 0 完成。验收对照 V4 第 81 节 V0.1 目标（Recall>95% / ROI Hit>95% / No-target>95% / <100ms）。"
