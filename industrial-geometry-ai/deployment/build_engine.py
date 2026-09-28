"""TensorRT engine 构建（V4 第 46/50 节）。

需要：pip install tensorrt（且 GPU 可用）。
用法：python -m deployment.build_engine --onnx out/model.onnx --out out/model.engine
"""

from __future__ import annotations

import argparse
import os


def build(onnx_path: str, out_path: str, fp16: bool = True, workspace_gb: int = 4):
    try:
        import tensorrt as trt
    except ImportError:
        raise SystemExit(
            "tensorrt 未安装：pip install tensorrt（需 GPU 环境；"
            "当前主机 GPU 驱动恢复后执行）")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise SystemExit("ONNX 解析失败")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 enabled")

    engine = builder.build_engine(network, config)
    with open(out_path, "wb") as f:
        f.write(engine.serialize())
    print(f"engine saved: {out_path}")


def main():
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=os.path.join(ROOT, "out", "model.onnx"))
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "model.engine"))
    ap.add_argument("--no-fp16", action="store_true")
    args = ap.parse_args()
    build(args.onnx, args.out, fp16=not args.no_fp16)


if __name__ == "__main__":
    main()
