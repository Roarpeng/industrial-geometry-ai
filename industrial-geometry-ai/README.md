# industrial-geometry-ai

工业自然语言几何点粗定位模型 —— V4 方案实现骨架。

对应文档：`../工业几何点粗定位模型_训练实施方案v4.md`

## 快速开始

```bash
cd industrial-geometry-ai

# 1. 端到端 smoke 测试（生成 32 张图 → 训练 2 epoch → 解码/分组/补全 → 指标 → 可视化）
python tools/run_smoke.py

# 2. 生成 Stage A 干净合成数据 5K
python -m dataset.synthetic_generator.generate --num 5000 --mode clean

# 3. 生成 Stage B 工业背景数据
python -m dataset.synthetic_generator.generate --num 15000 --mode industrial --out data/synthetic_b

# 4. 训练（RTX 4060 8GB）
python -m training.train --config configs/train.yaml

# 5. Phase 0 一键启动（需 GPU；生成 5K → 训练 → 校准 → ONNX → benchmark）
bash run_phase0.sh

# 6. 伪标注真实图（V4 §39.2；复核 spotcheck.json 后入训练集）
python3 tools/self_train.py --ckpt out/train/best.pth --images data/real_raw --out data/pseudo
```

## 结构 ↔ 方案章节对照

| 模块 | V4 方案章节 |
|---|---|
| `task/` vocabulary / task_spec / normalizer | 第 6/7/8 节（含 clarify 兜底） |
| `dataset/synthetic_generator/` | 第 30–34 节（硬约束 + 笔触渲染） |
| `dataset/dataset.py` | 第 26/27/56 节（targets on-the-fly） |
| `models/` | 第 11/12/14 节（backbone + FPN + head fusion + per-type heads） |
| `losses/` | 第 25/27/28 节（focal dense 归一化 + masked offset/size） |
| `postprocess/peak_decode.py` | 第 21 节（NMS + 亚像素细化） |
| `postprocess/grouping.py` | 第 22 节（K 约束分组） |
| `postprocess/completion.py` | 第 24 节（缺角几何补全） |
| `training/` | 第 43 节（EMA / warmup / cosine / AMP / 指标选 checkpoint） |
| `training/metrics.py` | 第 49 节（Point Recall@10px / 带约束 ROI Hit / 系统指标） |
| `inference/pipeline.py` | 第 51/52 节（输出 API schema） |

## 最终验收结果（2026-09-27，V0.1 全量正式终评）

模型：`out/phase0/best.pth`（ep104，12.3K 训练图，两轮迭代）。工作点 τ=0.10（全量 val 扫定）。

| 指标 | 实测（1303 张全量 val） | V0.1 门槛 | 结果 |
|---|---|---|---|
| Point Recall（10px 固定阈值） | **0.9728** | > 0.95 | PASS |
| ROI Hit（带上界约束） | **1.0000** | > 0.95 | PASS |
| No-target | **0.9917** | > 0.95 | PASS |
| GPU 延迟（模型前向 P50） | **2.97 ms** | < 100 ms | PASS |
| Object Recall | 0.9550 | （V1.0 参考 0.95） | 已达 V1.0 线 |

分类型 point_recall：triangle 0.960 / line 0.963 / rectangle 0.972 / star 0.995。

迭代记录：
1. 第一轮 5.85K×100ep → point 0.92（τ 扫描后 0.936），弱点 triangle 0.89 / line 0.91
2. 第二轮 +6.5K（含 1.5K triangle+line 弱类偏采样）+ warm-restart 续训 → point 0.975 / obj 0.955

已知留档问题（V1.0 迭代项）：跨类型 FP——同图查 triangle 时矩形角点偶被误认为三角形
（可视化 `out/phase0/vis/` 有实例）；对应 V4 exp005 Hard Negative 轮与 FP<2% 指标。

## 已完成验证（2026-09-26，RTX 4060 + CPU 复验）

- smoke 端到端 PASS；2K×100ep 收敛验证（ep31：point_recall 0.79 / roi_hit 1.0 / roi_ratio≈0.10）
- 温度校准（T=0.35，raw hit 0.986）；ONNX 导出 + schema 校验 + 数值一致性 ≤4e-5
- 100+ 点密集压力（机制验证）；self_train 伪标注（500 图 / 1129 对象 / 55 抽检）
- 实现期修复：EMA warmup、生成器退化对象（9.6%→0）、文档 §44 语义勘误

## 已知 TODO（按 V4 执行顺序）

- [ ] `dataset/stroke_stats/`：实测 ≥50 段真实笔迹统计，替换渲染先验（第 32 节要求，当前为占位参数）
- [ ] 真实数据 300–500 张标注（Phase 1）
- [ ] Phase 0 正式 5K 训练（`bash run_phase0.sh`，需先重启恢复 GPU）
- [ ] Stage C 密集数据训练后复测 100+ 点精度（当前仅机制验证）
- [ ] TensorRT engine 构建与实测延迟（`deployment/build_engine.py`，需 GPU）
