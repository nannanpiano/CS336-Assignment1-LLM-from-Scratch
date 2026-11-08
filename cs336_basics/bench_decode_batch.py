"""
批量生成测速：同时为 B 条 prompt 生成文本，对比用 / 不用 KV cache 的吞吐（每秒一共生成多少 token），
看 KV cache 的加速比怎样随 batch 变大而变化。

背景（见 results/kv_cache_l40.md）：batch 1 时，一次前向的耗时几乎全是 Python 执行和 kernel 启动的固定开销
（约 4.3 ms，真正计算只有约 0.5 ms），所以不管每步算 1 个还是 256 个 token 都差不多慢，KV cache 几乎没用。
batch 变大后，每一步的计算量变大，固定开销被摊薄，“重算整段”和“只算 1 个新 token”的差别才会显现出来。

做法：
    - 从验证集里随机取 B 段文本，每段的前 --prompt_len 个 token 作为 prompt（所有 prompt 一样长，不需要 padding）。
    - 贪心解码，每条都恰好生成 --new_tokens 个 token（不因 <|endoftext|> 提前停）；prompt_len + new_tokens ≤ context_length，
      所以带缓存时一直是“每步只算 1 个新 token”。
    - 不用缓存：每一步把整段（B, 当前长度）送进模型；用缓存：先 prefill 整段 prompt，之后每步只送 (B, 1)。
      这里直接调用模型（decoding.py 的 decode() 只支持单条序列），逻辑和 decode() 的两条路径一样。
    - 先跑一次预热（不计时），并检查两种方式生成的 token 完全一致；之后重复 --repeats 次取中位数，两种方式交错执行。

用法（服务器上，配合 gpuhold 拿到 GPU3）：
    /usr/bin/python3 ~/workspace/gpuhold/gpuhold.py run --gpus 3 -- \
        uv run --no-sync python -m cs336_basics.bench_decode_batch runs/ts_gpu_lr1e-3 --device cuda --out results/kv_cache_batch_l40

输出：<out>.csv（每次计时一行）、<out>.md（汇总表）、<out>.png（吞吐和加速比随 batch 变化的图）
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
import numpy as np
import torch

from cs336_basics.modules import TransformerLM


def parse_args():
    parser = argparse.ArgumentParser(description="批量生成时，对比用 / 不用 KV cache 的吞吐")
    parser.add_argument("run_dir", help="实验目录（里面要有 config.json 和 ckpt.pt）")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 4, 16, 32, 64, 128])
    parser.add_argument("--prompt_len", type=int, default=4, help="每条 prompt 的 token 数")
    parser.add_argument("--new_tokens", type=int, default=252, help="每条生成多少个新 token")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="默认自动选择 cuda > mps > cpu")
    parser.add_argument("--out", default="results/kv_cache_batch", help="结果文件前缀（生成 .csv/.md/.png）")
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


@torch.no_grad()
def generate_batch(model, prompts: torch.Tensor, new_tokens: int, use_kv_cache: bool) -> torch.Tensor:
    """
    批量贪心生成。prompts: (B, prompt_len)，返回新生成的 token: (B, new_tokens)。
    长度不会超过 context_length（调用方保证），所以不需要像 decode() 那样处理窗口滑动。
    """
    ids = prompts
    caches = None
    for _ in range(new_tokens):
        if not use_kv_cache:
            logits = model(ids)[:, -1]                       # 整段重算，只取最后一个位置 (B, V)
        elif caches is None:
            out, caches = model(ids, use_cache=True)         # prefill：整段 prompt 送进去一次
            logits = out[:, -1]
        else:
            out, caches = model(ids[:, -1:], kv_caches=caches, use_cache=True)   # 每条只送最新的 1 个 token
            logits = out[:, -1]
        next_ids = torch.argmax(logits, dim=-1, keepdim=True)   # 贪心 (B, 1)
        ids = torch.cat([ids, next_ids], dim=1)
    return ids[:, prompts.shape[1]:]


def timed(model, prompts, new_tokens, use_kv_cache, device):
    sync(device)
    t0 = time.perf_counter()
    out = generate_batch(model, prompts, new_tokens, use_kv_cache)
    sync(device)
    return time.perf_counter() - t0, out


def main():
    args = parse_args()
    device = get_device(args.device)

    with open(os.path.join(args.run_dir, "config.json")) as f:
        cfg = json.load(f)
    ctx = cfg["context_length"]
    if args.prompt_len + args.new_tokens > ctx:
        raise ValueError(f"prompt_len + new_tokens = {args.prompt_len + args.new_tokens} 超过了 context_length {ctx}")
    model = TransformerLM(
        vocab_size=cfg["vocab_size"], context_length=ctx, num_layers=cfg["num_layers"], d_model=cfg["d_model"],
        num_heads=cfg["num_heads"], d_ff=cfg["d_ff"], rope_theta=cfg["rope_theta"],
        attn_gate=cfg.get("attn_gate", "none"), device=device,
    )
    model.load_state_dict(torch.load(os.path.join(args.run_dir, "ckpt.pt"), map_location=device)["model"])
    model.eval()

    # prompt 取自验证集的随机位置：每条 prompt 都是真实文本的开头几个 token，而且彼此不同
    val = np.load(cfg["val_data"], mmap_mode="r")
    rng = np.random.default_rng(args.seed)
    max_b = max(args.batch_sizes)
    starts = rng.integers(0, len(val) - args.prompt_len, size=max_b)
    all_prompts = torch.from_numpy(np.stack([val[s: s + args.prompt_len] for s in starts]).astype(np.int64)).to(device)

    modes = [("no_cache", False), ("kv_cache", True)]
    records = []
    print(f"模型 {args.run_dir}，设备 {device}，prompt {args.prompt_len} 个 token，每条生成 {args.new_tokens} 个，"
          f"重复 {args.repeats} 次取中位数")
    for b in args.batch_sizes:
        prompts = all_prompts[:b]
        # 预热（不计时），同时检查两种方式的生成结果完全一致
        _, a = timed(model, prompts, args.new_tokens, False, device)
        _, c = timed(model, prompts, args.new_tokens, True, device)
        mismatch = (a != c).any(dim=1).sum().item()
        for rep in range(1, args.repeats + 1):
            for mode, use_cache in modes:   # 两种方式交错执行
                sec, _ = timed(model, prompts, args.new_tokens, use_cache, device)
                records.append({"batch": b, "mode": mode, "repeat": rep, "seconds": sec,
                                "tok_per_sec": b * args.new_tokens / sec})
        med = {m: statistics.median(r["tok_per_sec"] for r in records if r["batch"] == b and r["mode"] == m)
               for m, _ in modes}
        print(f"  batch {b:4d}: 不用缓存 {med['no_cache']:10.0f} tok/s   用缓存 {med['kv_cache']:10.0f} tok/s   "
              f"加速 {med['kv_cache'] / med['no_cache']:5.2f}×   结果不一致的条数 {mismatch}/{b}", flush=True)
        records[-1]["mismatch"] = mismatch   # 记在这个 batch 的最后一行，写进汇总表

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["batch", "mode", "repeat", "seconds", "tok_per_sec", "mismatch"])
        writer.writeheader()
        writer.writerows(records)

    rows = []
    for b in args.batch_sizes:
        sel = [r for r in records if r["batch"] == b]
        nc = statistics.median(r["tok_per_sec"] for r in sel if r["mode"] == "no_cache")
        kv = statistics.median(r["tok_per_sec"] for r in sel if r["mode"] == "kv_cache")
        mm = next(r["mismatch"] for r in reversed(sel) if "mismatch" in r)
        rows.append((b, nc, kv, kv / nc, mm))
    lines = [
        f"# 批量生成：KV cache 加速比随 batch 的变化（{args.run_dir}，设备 {device}）",
        "",
        f"贪心解码，prompt {args.prompt_len} 个 token（取自验证集），每条生成 {args.new_tokens} 个，FP32，"
        f"重复 {args.repeats} 次取中位数。吞吐 = batch × 生成 token 数 / 用时。",
        "",
        "| batch | 不用缓存 tok/s | 用 KV cache tok/s | 加速比 | 两种方式结果不一致的条数 |",
        "|---|---|---|---|---|",
    ] + [f"| {b} | {nc:,.0f} | {kv:,.0f} | {sp:.2f}× | {mm}/{b} |" for b, nc, kv, sp, mm in rows]
    with open(args.out + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")

    # 图：左边两种方式的吞吐（对数坐标，batch 跨两个数量级），右边加速比
    bs = [r[0] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(bs, [r[1] for r in rows], marker="o", label="no cache (recompute full sequence)")
    axes[0].plot(bs, [r[2] for r in rows], marker="o", label="KV cache")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("batch size")
    axes[0].set_ylabel("generated tokens / second (whole batch)")
    axes[0].legend()
    axes[1].plot(bs, [r[3] for r in rows], marker="o", color="C2")
    axes[1].set_xscale("log", base=2)
    axes[1].axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    axes[1].set_xlabel("batch size")
    axes[1].set_ylabel("speedup (KV cache / no cache)")
    for ax in axes:
        ax.grid(True, alpha=0.3, which="both")
    fig.suptitle(f"KV cache speedup vs batch size ({device}, greedy, {args.new_tokens} new tokens)")
    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=150)

    print("\n" + "\n".join(lines))
    print(f"\n结果：{args.out}.csv / .md / .png")


if __name__ == "__main__":
    main()
