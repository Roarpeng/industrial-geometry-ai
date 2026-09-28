# 工业几何点粗定位模型（Industrial Geometry Coarse Point Localization）

自然语言驱动的工业几何角点粗定位：输入「找出图中的三角形」这类指令 + 图像，输出目标图形的角点坐标。

## 仓库结构

| 路径 | 说明 |
|---|---|
| `工业几何点粗定位模型_训练实施方案v4.md` | **执行基准文档**（81 节）。v3 为历史版本，保留供对照。 |
| `industrial-geometry-ai/` | V4 方案的可运行实现骨架（Python，约 2700 行） |
| `industrial-geometry-ai/out/*/*.json` | 训练与评测的文字指标记录 |
| ~~`industrial-geometry-ai/data/`~~ | 合成数据集（约 4.6 G），已 gitignore，按 seed 复现 |
| ~~`industrial-geometry-ai/out/**/*.pth`~~ | 训练权重（`.pth`/`.onnx`），已 gitignore |
| ~~`test_imgs/`~~ | 本地真实测试图，已 gitignore |

## 快速开始

```bash
pip install -r industrial-geometry-ai/requirements.txt
cd industrial-geometry-ai
python tools/run_smoke.py      # 端到端 smoke 测试
bash run_phase0.sh             # Phase 0 一键：生成数据 → 训练 → 校准 → ONNX → benchmark
```

模块划分、训练细节与验收指标见 [`industrial-geometry-ai/README.md`](industrial-geometry-ai/README.md)。

## 当前状态

V0.1 全量终评（1303 张 val）四项门槛全部 PASS：

| 指标 | 实测 | 门槛 |
|---|---|---|
| Point Recall @10px | 0.9728 | > 0.95 |
| ROI Hit（带上界约束） | 1.0000 | > 0.95 |
| No-target | 0.9917 | > 0.95 |
| GPU 前向 P50 延迟 | 2.97 ms | < 100 ms |

Object Recall 0.9550，已达 V1.0 参考线。待办：真实数据标注、笔触统计实测、跨类型 FP 治理（V1.0 迭代项）。
