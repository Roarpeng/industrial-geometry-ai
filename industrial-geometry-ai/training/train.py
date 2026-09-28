"""训练主循环（V4 第 43 节）：

AdamW + warmup + cosine + AMP（bf16 优先）+ EMA + 梯度累积 +
按 Object Recall 选 checkpoint（不以 loss 为准）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.dataset import GeoPointDataset, collate_fn
from losses import compute_loss
from models.build import build_model
from task.vocabulary import vocabulary_dict
from .ema import ModelEMA
from .metrics import evaluate


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _split(names_hashable, val_ratio_bucket: int = 10):
    """按文件名哈希分桶（同背景/同 seed 的数据不跨 train/val 泄漏）。"""
    train_idx, val_idx = [], []
    for i, name in enumerate(names_hashable):
        h = int(hashlib.md5(os.path.basename(name).encode()).hexdigest(), 16) % 10
        (val_idx if h < 1 else train_idx).append(i)
    return train_idx, val_idx


def make_lr_fn(warmup_steps: int, total_steps: int):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        t = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))
    return fn


def main(argv=None):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(root, "configs", "train.yaml"))
    ap.add_argument("--data-dir", default=None, help="逗号分隔多个数据目录")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="从 out_dir/last.pth 断点续训（GPU 故障后恢复）")
    args = ap.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        import yaml
        tcfg_full = yaml.safe_load(f)
    with open(os.path.join(root, "configs", "model.yaml"), encoding="utf-8") as f:
        mcfg = yaml.safe_load(f)
    with open(os.path.join(root, "configs", "dataset.yaml"), encoding="utf-8") as f:
        dcfg = yaml.safe_load(f)

    data_dirs = (args.data_dir or tcfg_full.get("data_dir", "data/synthetic")).split(",")
    data_dirs = [os.path.join(root, d) if not os.path.isabs(d) else d for d in data_dirs]
    out_dir = args.out or tcfg_full.get("out_dir", "out/train")
    out_dir = os.path.join(root, out_dir) if not os.path.isabs(out_dir) else out_dir
    os.makedirs(out_dir, exist_ok=True)

    epochs = tcfg_full["epochs"]
    batch_size = tcfg_full["batch_size"]
    num_workers = tcfg_full.get("num_workers", 4)
    if args.smoke:
        epochs, batch_size, num_workers = 2, 8, 0
        data_dirs = data_dirs[:2]

    _set_seed(tcfg_full.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tcfg = {**dcfg["target"], "roi": dcfg["roi"], **dcfg["decode"]}
    decode_cfg = {**dcfg["decode"], "roi": dcfg["roi"]}
    group_cfg = dcfg["grouping"]

    base = GeoPointDataset(data_dirs, tcfg=dcfg["target"], train=True)
    limit = 24 if args.smoke else 0
    if limit:
        base = GeoPointDataset(data_dirs, tcfg=dcfg["target"], train=True, limit=limit)
    train_idx, val_idx = _split(base.names)
    val_limit = tcfg_full.get("val_limit", 0)
    if val_limit and len(val_idx) > val_limit:
        val_idx = val_idx[:val_limit]  # 长跑时限制 val 规模控制开销
    train_ds = GeoPointDataset(data_dirs, tcfg=dcfg["target"], train=True, indices=train_idx)
    val_ds = GeoPointDataset(data_dirs, tcfg=dcfg["target"], train=False, indices=val_idx)
    print(f"dataset: train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn,
                              drop_last=len(train_ds) >= batch_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, collate_fn=collate_fn)

    model = build_model(mcfg).to(device)
    ema = ModelEMA(model, decay=tcfg_full.get("ema_decay", 0.9999))

    decay, nd = tcfg_full["weight_decay"], 0.0
    params = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad
                    and ("norm" in n or n.endswith(".bias"))], "weight_decay": nd},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad
                    and not ("norm" in n or n.endswith(".bias"))], "weight_decay": decay},
    ]
    optimizer = torch.optim.AdamW(params, lr=tcfg_full["lr"])
    accum = tcfg_full.get("grad_accumulation", 1)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / accum))
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * tcfg_full.get("warmup_epochs", 5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_fn(warmup_steps, total_steps))

    use_amp = tcfg_full.get("amp", True) and device.type == "cuda" and not args.smoke
    bf16 = use_amp and torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and not bf16)

    def amp_ctx():
        if bf16:
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if use_amp:
            return torch.amp.autocast("cuda", dtype=torch.float16)
        import contextlib
        return contextlib.nullcontext()

    weights = tcfg_full.get("loss_weights", {})
    focal = tcfg_full.get("focal", {})
    W, H = mcfg["input_size"][1], mcfg["input_size"][0]

    with open(os.path.join(out_dir, "task_vocabulary.json"), "w", encoding="utf-8") as f:
        json.dump(vocabulary_dict(), f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "decode_params.json"), "w", encoding="utf-8") as f:
        json.dump({**decode_cfg, "grouping": group_cfg}, f, ensure_ascii=False, indent=2)

    best_metric = -1.0
    history = []
    start_epoch = 0

    resume_path = os.path.join(out_dir, "last.pth")
    if args.resume and os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.module.load_state_dict(ck["ema"])
        if "optimizer" in ck:
            optimizer.load_state_dict(ck["optimizer"])
        if "scheduler" in ck:
            scheduler.load_state_dict(ck["scheduler"])
        start_epoch = ck.get("epoch", -1) + 1
        best_metric = ck.get("best_metric", -1.0)
        hist_path = os.path.join(out_dir, "history.json")
        if os.path.exists(hist_path):
            with open(hist_path, encoding="utf-8") as f:
                history = json.load(f)
        print(f"resumed from {resume_path}: start_epoch={start_epoch} best={best_metric:.4f}")
    global_step = start_epoch * steps_per_epoch
    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        running = {}
        optimizer.zero_grad(set_to_none=True)
        for it, (images, targets, _metas) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
            with amp_ctx():
                outputs = model(images)
                losses = compute_loss(outputs, targets, weights, focal)
                loss = losses["total"] / accum
            scaler.scale(loss).backward()
            if (it + 1) % accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               tcfg_full.get("gradient_clip", 1.0))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
                scheduler.step()
                global_step += 1
            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + float(v.detach())
            if it % 20 == 0:
                avg = {k: v / (it + 1) for k, v in running.items()}
                print(f"ep{epoch} it{it}/{len(train_loader)} "
                      + " ".join(f"{k}={v:.4f}" for k, v in avg.items()))
        if not math.isfinite(running.get("total", 0.0)):
            print("loss not finite, stopping")
            break

        if (epoch + 1) % tcfg_full.get("val_interval", 1) == 0:
            metrics = evaluate(ema.module, val_loader, device, decode_cfg, group_cfg, W, H)
            key_metric = metrics.get(tcfg_full.get("checkpoint_metric", "object_recall"), float("nan"))
            print(f"[epoch {epoch}] {time.time()-t0:.1f}s "
                  f"point_recall={metrics['point_recall']:.4f} "
                  f"roi_hit={metrics['roi_hit']:.4f} "
                  f"object_recall={metrics['object_recall']:.4f} "
                  f"no_target={metrics['no_target']:.4f}")
            history.append({"epoch": epoch, "metrics": {k: (None if v != v else round(v, 6))
                                                        for k, v in metrics.items()}})
            is_best = key_metric == key_metric and key_metric > best_metric
            if is_best:
                best_metric = key_metric
                print(f"new best {tcfg_full.get('checkpoint_metric')}: {best_metric:.4f}")
            ck = {"model": model.state_dict(), "ema": ema.module.state_dict(),
                  "epoch": epoch, "mcfg": mcfg, "decode": decode_cfg, "grouping": group_cfg,
                  "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                  "best_metric": best_metric}
            torch.save(ck, os.path.join(out_dir, "last.pth"))
            if is_best:
                torch.save(ck, os.path.join(out_dir, "best.pth"))

    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"train done. best={best_metric:.4f} out={out_dir}")


if __name__ == "__main__":
    main()
