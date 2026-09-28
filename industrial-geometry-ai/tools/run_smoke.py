"""端到端 smoke 测试：

生成（clean + industrial）→ 训练 2 epoch（smoke 档）→ 推理 pipeline →
可视化 → 正则化 normalizer 冒烟。全部通过打印 SMOKE PASS。
"""

import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dataset.synthetic_generator.generate import main as gen_main  # noqa: E402
from inference.pipeline import InferencePipeline  # noqa: E402
from task.normalizer import ClarifyNeeded, normalize  # noqa: E402
from tools.visualize import draw_prediction, load_pair  # noqa: E402
from tools.visualize_prediction import load_pipeline  # noqa: E402
from training.train import main as train_main  # noqa: E402


def check_normalizer():
    cases = [
        ("帮我找到画面里所有五角星的角点", "star", "all"),
        ("找出所有三角形", "triangle", "all"),
        ("找到矩形四个角", "rectangle", "all"),
        ("找到这条直线两个端点", "line", "single"),
    ]
    for text, target, scope in cases:
        spec = normalize(text)
        assert spec.target == target, f"{text} -> {spec.target} != {target}"
        assert spec.scope == scope, f"{text} scope {spec.scope} != {scope}"
    try:
        normalize("把那个东西找出来")
        raise AssertionError("expected ClarifyNeeded")
    except ClarifyNeeded as e:
        assert e.candidates
    print("[1/5] normalizer ok (含 clarify 兜底)")


def main():
    dir_a = os.path.join(ROOT, "data", "smoke_a")
    dir_b = os.path.join(ROOT, "data", "smoke_b")
    out_dir = os.path.join(ROOT, "out", "smoke_train")
    vis_dir = os.path.join(ROOT, "out", "smoke_vis")
    for d in (dir_a, dir_b, out_dir, vis_dir):
        shutil.rmtree(d, ignore_errors=True)

    check_normalizer()

    gen_main(["--num", "20", "--mode", "clean", "--out", dir_a, "--seed", "7"])
    gen_main(["--num", "12", "--mode", "industrial", "--out", dir_b, "--seed", "8"])
    print("[2/5] data generated")

    train_main(["--config", os.path.join(ROOT, "configs", "train.yaml"),
                "--data-dir", f"{dir_a},{dir_b}", "--out", out_dir, "--smoke"])
    print("[3/5] smoke train done")

    ckpt = os.path.join(out_dir, "best.pth")
    assert os.path.exists(ckpt), "best.pth missing"
    pipe = load_pipeline(ckpt)

    names = sorted(f for f in os.listdir(dir_a) if f.endswith(".json") and f != "manifest.json")
    n_vis = 0
    for fn in names[:4]:
        img, ann = load_pair(os.path.join(dir_a, fn))
        for target in ("star", "triangle"):
            result = pipe.run(img, target)
            draw_prediction(img, result, ann,
                            os.path.join(vis_dir, f"{fn.replace('.json','')}_{target}.png"))
            n_vis += 1
    print(f"[4/5] pipeline + visualization ok ({n_vis} images)")

    with open(os.path.join(out_dir, "history.json"), encoding="utf-8") as f:
        hist = json.load(f)
    assert hist and "metrics" in hist[-1]
    print(f"[5/5] metrics recorded: "
          f"point_recall={hist[-1]['metrics'].get('point_recall')} "
          f"object_recall={hist[-1]['metrics'].get('object_recall')}")
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
