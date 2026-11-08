"""
在【完整】验证集上评估一个训练好的模型，得到可以写进报告的最终 val loss。

为什么需要它：train.py 训练时的 val loss 是随机抽 eval_iters 个 batch 算的，只覆盖验证集的一小部分
（默认 20×32×256 ≈ 16 万 token，TinyStories 验证集有 546 万 token），会有 ±0.01 左右的随机波动。
这里把整个验证集切成不重叠的 context_length 长度的窗口，全部算一遍取平均，结果是确定的、可复现的。

用法：
    uv run python -m cs336_basics.eval_full runs/ts_gpu_lr1e-3
    uv run python -m cs336_basics.eval_full runs/ts_gpu_lr1e-3 runs/ts_gpu_lr3e-4 --device cuda
    # 只想快速检查脚本能不能跑：只评估前 50 万个 token
    uv run python -m cs336_basics.eval_full runs/ts_mps --max_tokens 500000

结果会打印出来，并保存到每个实验目录下的 full_val.json。
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from cs336_basics.modules import TransformerLM
from cs336_basics.utils import cross_entropy


def parse_args():
    parser = argparse.ArgumentParser(description="在完整验证集上评估训练好的模型")
    parser.add_argument("run_dirs", nargs="+", help="一个或多个实验目录（里面要有 config.json 和 ckpt.pt）")
    parser.add_argument("--device", type=str, default=None, help="默认自动选择 cuda > mps > cpu")
    parser.add_argument("--batch_size", type=int, default=64, help="评估时每批多少个窗口（只影响速度，不影响结果）")
    parser.add_argument("--max_tokens", type=int, default=None, help="只评估验证集的前这么多 token（调试用）")
    return parser.parse_args()


def get_device(name):
    if name is not None:
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def evaluate_run(run_dir: str, device: str, batch_size: int, max_tokens: int | None) -> dict:
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)

    # 用训练时保存的超参数重建模型（一律用关键字参数，避免位置写错），再加载权重
    model = TransformerLM(
        vocab_size=cfg["vocab_size"], context_length=cfg["context_length"], num_layers=cfg["num_layers"],
        d_model=cfg["d_model"], num_heads=cfg["num_heads"], d_ff=cfg["d_ff"], rope_theta=cfg["rope_theta"],
        attn_gate=cfg.get("attn_gate", "none"),   # 旧实验的 config.json 里没有这一项，默认不加门控
        device=device,
    )
    ckpt = torch.load(os.path.join(run_dir, "ckpt.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 验证集切成不重叠的窗口：第 k 个窗口的输入是 [kT, (k+1)T)，目标是整体右移一位
    val = np.load(cfg["val_data"], mmap_mode="r")
    if max_tokens is not None:
        val = val[: max_tokens + 1]
    T = cfg["context_length"]
    n = (len(val) - 1) // T                     # 最后不够一个窗口的零头丢掉
    x_all = torch.from_numpy(val[: n * T].astype(np.int64)).view(n, T)
    y_all = torch.from_numpy(val[1: n * T + 1].astype(np.int64)).view(n, T)

    # 一律用 FP32、不编译来评估：这是“最准确”的数值，和训练时开了哪些加速选项无关
    total, count, t0 = 0.0, 0, time.time()
    for i in range(0, n, batch_size):
        x, y = x_all[i: i + batch_size].to(device), y_all[i: i + batch_size].to(device)
        loss = cross_entropy(model(x), y)       # 这一批每个 token 的平均 loss
        total += loss.item() * y.numel()        # 乘回 token 数，最后按 token 总数平均
        count += y.numel()

    result = {
        "full_val_loss": total / count,
        "tokens": count,
        "step": ckpt.get("iteration"),
        "seconds": round(time.time() - t0, 1),
        "max_tokens": max_tokens,
    }
    # 只有评估了整个验证集时才保存，避免调试用的部分结果覆盖正式结果
    if max_tokens is None:
        with open(os.path.join(run_dir, "full_val.json"), "w") as f:
            json.dump(result, f, indent=2)
    return result


def main():
    args = parse_args()
    device = get_device(args.device)
    for run_dir in args.run_dirs:
        r = evaluate_run(run_dir, device, args.batch_size, args.max_tokens)
        part = "" if args.max_tokens is None else f"（只评估了前 {args.max_tokens:,} token）"
        print(f"{run_dir}: full val loss {r['full_val_loss']:.4f}  "
              f"({r['tokens']:,} token, step {r['step']}, {r['seconds']}s){part}", flush=True)


if __name__ == "__main__":
    main()
