"""
生成速度测速：对比用 / 不用 KV cache 时，decoding.py 的 decode() 每秒能生成多少个 token，
以及加速比怎样随生成长度变化。

为什么要测不同的生成长度：
    不用缓存时，生成第 t 个 token 要把前面全部 ~t 个 token 重算一遍，越往后每一步越慢；
    用缓存时，每一步只算 1 个新 token（attention 要读的缓存会变长，但计算量小得多）。
    所以生成得越长，两者差距应该越大。

做法：
    - 用训练好的模型（默认 FP32、不编译），batch = 1，贪心解码（temperature=0），关掉 <|endoftext|> 提前停止，
      保证每次都恰好生成 N 个 token。
    - prompt 长度 + N 不超过 context_length，所以带缓存时一直走“只算 1 个新 token”的路径，不会触发超长重算。
    - 每个长度、每种模式先跑一次预热（GPU 第一次跑要分配显存、加载 kernel），不计时；
      之后重复 --repeats 次取中位数，两种模式交错执行，减少机器状态变化带来的偏差。
    - 计时前后都做设备同步（CUDA 是异步的，不同步会测到“提交任务”的时间而不是“算完”的时间）。

用法（服务器上，配合 gpuhold 拿到 GPU3）：
    /usr/bin/python3 ~/workspace/gpuhold/gpuhold.py run --gpus 3 -- \
        uv run --no-sync python -m cs336_basics.bench_decode runs/ts_gpu_lr1e-3 --device cuda --out results/kv_cache_l40

输出：<out>.csv（每次计时一行）、<out>.md（汇总表）、<out>.png（tok/s 和加速比随生成长度变化的图）
"""

import argparse
import csv
import json
import os
import statistics
import time

import matplotlib

matplotlib.use("Agg")   # 不需要显示器，直接保存图片
import matplotlib.pyplot as plt
import torch

from cs336_basics.decoding import decode
from cs336_basics.modules import TransformerLM
from cs336_basics.tokenizer import Tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="对比用 / 不用 KV cache 的生成速度")
    parser.add_argument("run_dir", help="实验目录（里面要有 config.json 和 ckpt.pt）")
    parser.add_argument("--vocab", default="data/tokenizers/tinystories_vocab.json")
    parser.add_argument("--merges", default="data/tokenizers/tinystories_merges.txt")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--lengths", type=int, nargs="+", default=[16, 32, 64, 128, 192, 250],
                        help="要测的生成长度（新生成的 token 数）")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default=None, help="默认自动选择 cuda > mps > cpu")
    parser.add_argument("--out", default="results/kv_cache", help="结果文件前缀（生成 .csv/.md/.png）")
    return parser.parse_args()


def get_device(name):
    if name is not None:
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device: str):
    """等设备把已经提交的计算全部做完，计时才准。"""
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device.startswith("mps"):
        torch.mps.synchronize()


def time_decode(model, prompt_ids, n, ctx, device, use_kv_cache) -> tuple[float, list[int]]:
    """生成恰好 n 个 token，返回 (用时秒数, 生成的 token)。"""
    sync(device)
    t0 = time.perf_counter()
    out = decode(model, prompt_ids, n, ctx, temperature=0, eot_id=None, device=device, use_kv_cache=use_kv_cache)
    sync(device)
    return time.perf_counter() - t0, out


def main():
    args = parse_args()
    device = get_device(args.device)

    with open(os.path.join(args.run_dir, "config.json")) as f:
        cfg = json.load(f)
    ctx = cfg["context_length"]
    model = TransformerLM(
        vocab_size=cfg["vocab_size"], context_length=ctx, num_layers=cfg["num_layers"], d_model=cfg["d_model"],
        num_heads=cfg["num_heads"], d_ff=cfg["d_ff"], rope_theta=cfg["rope_theta"],
        attn_gate=cfg.get("attn_gate", "none"), device=device,
    )
    model.load_state_dict(torch.load(os.path.join(args.run_dir, "ckpt.pt"), map_location=device)["model"])
    model.eval()

    tokenizer = Tokenizer.from_files(args.vocab, args.merges, special_tokens=["<|endoftext|>"])
    prompt_ids = tokenizer.encode(args.prompt)
    for n in args.lengths:
        if len(prompt_ids) + n > ctx:
            raise ValueError(f"prompt {len(prompt_ids)} + 生成 {n} 个 token 超过了 context_length {ctx}，"
                             f"带缓存时会退回整段重算，测出来的不是纯粹的 KV cache 速度")

    modes = [("no_cache", False), ("kv_cache", True)]
    records = []
    print(f"模型 {args.run_dir}，设备 {device}，prompt {len(prompt_ids)} 个 token，重复 {args.repeats} 次取中位数")
    for n in args.lengths:
        # 预热：每种模式先跑一次，不计时；顺便确认两种模式生成的内容完全一样
        _, a = time_decode(model, prompt_ids, n, ctx, device, False)
        _, b = time_decode(model, prompt_ids, n, ctx, device, True)
        if a != b:
            raise RuntimeError(f"生成 {n} 个 token 时，用和不用缓存的结果不一致")
        for rep in range(1, args.repeats + 1):
            for mode, use_cache in modes:   # 两种模式交错执行
                sec, _ = time_decode(model, prompt_ids, n, ctx, device, use_cache)
                records.append({"length": n, "mode": mode, "repeat": rep, "seconds": sec, "tok_per_sec": n / sec})
        med = {m: statistics.median(r["tok_per_sec"] for r in records if r["length"] == n and r["mode"] == m)
               for m, _ in modes}
        print(f"  生成 {n:4d} 个 token: 不用缓存 {med['no_cache']:8.1f} tok/s   用缓存 {med['kv_cache']:8.1f} tok/s   "
              f"加速 {med['kv_cache'] / med['no_cache']:.2f}×", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["length", "mode", "repeat", "seconds", "tok_per_sec"])
        writer.writeheader()
        writer.writerows(records)

    # 汇总：每个长度两种模式的 tok/s 中位数和加速比
    rows = []
    for n in args.lengths:
        nc = statistics.median(r["tok_per_sec"] for r in records if r["length"] == n and r["mode"] == "no_cache")
        kv = statistics.median(r["tok_per_sec"] for r in records if r["length"] == n and r["mode"] == "kv_cache")
        rows.append((n, nc, kv, kv / nc))
    lines = [
        f"# KV cache 生成速度（{args.run_dir}，设备 {device}）",
        "",
        f"batch 1，贪心解码，prompt {len(prompt_ids)} 个 token，FP32，重复 {args.repeats} 次取中位数。",
        "",
        "| 生成 token 数 | 不用缓存 tok/s | 用 KV cache tok/s | 加速比 |",
        "|---|---|---|---|",
    ] + [f"| {n} | {nc:,.1f} | {kv:,.1f} | {sp:.2f}× |" for n, nc, kv, sp in rows]
    with open(args.out + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")

    # 图：左边 tok/s 随生成长度变化，右边加速比
    lengths = [r[0] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(lengths, [r[1] for r in rows], marker="o", label="no cache (recompute full sequence)")
    axes[0].plot(lengths, [r[2] for r in rows], marker="o", label="KV cache")
    axes[0].set_xlabel("generated tokens")
    axes[0].set_ylabel("tokens / second")
    axes[0].legend()
    axes[1].plot(lengths, [r[3] for r in rows], marker="o", color="C2")
    axes[1].set_xlabel("generated tokens")
    axes[1].set_ylabel("speedup (KV cache / no cache)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Decoding speed with vs without KV cache ({device}, batch 1, greedy)")
    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=150)

    print("\n" + "\n".join(lines))
    print(f"\n结果：{args.out}.csv / .md / .png")


if __name__ == "__main__":
    main()
