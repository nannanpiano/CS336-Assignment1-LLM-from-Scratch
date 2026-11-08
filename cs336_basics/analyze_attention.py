"""
分析训练好的模型内部：attention sink、massive activation、门控值，用来对比 baseline 和 Gated Attention。

测的三个量（定义尽量和 Gated Attention 论文 4.3 节一致）：
    first_token_attn（论文里的 F-Attn）：每个 query 分给窗口里第 1 个 token（位置 0）的注意力权重，
        对所有 query（不含位置 0 自己，它只能看自己，权重必然是 1）、所有 head、所有样本取平均。逐层报告，再对层取平均。
        如果没有 sink，注意力大致均匀分布在能看到的位置上，期望值只有约 (1/t) 的平均 ≈ 2%（context 256 时）。
    max_activation（论文里的 M-Act）：每一层 TransformerBlock 输出（残差流）里绝对值最大的那个数，逐层报告，再对层取平均。
    gate_mean / gate_small_frac：门控 sigmoid 输出的平均值，以及小于 0.1 的比例（衡量门控有多“稀疏”），只有门控模型才有。

做法：从验证集随机取 --num_seqs 段长 context_length 的文本（和训练时取数据的方式一样），整段送进模型一次，
     用替换函数和 forward hook 把中间结果记下来。不修改模型代码。

用法（在服务器上，用 CPU 就够了）：
    uv run --no-sync python -m cs336_basics.analyze_attention runs/ts_gpu_lr1e-3 runs/ts_gpu_lr1e-3_gateelementwise_seed0 \
        --device cpu --out results/attention_sink
输出：<out>.json（每个模型、每一层的数值）、<out>.md（汇总表）
"""

import argparse
import json
import os

import numpy as np
import torch

import cs336_basics.modules as M


def parse_args():
    parser = argparse.ArgumentParser(description="分析 attention sink、massive activation 和门控值")
    parser.add_argument("run_dirs", nargs="+", help="一个或多个实验目录（里面要有 config.json 和 ckpt.pt）")
    parser.add_argument("--num_seqs", type=int, default=64, help="用多少段验证集文本")
    parser.add_argument("--seed", type=int, default=0, help="抽取文本的随机种子（所有模型用同一批文本）")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="results/attention_sink", help="结果文件前缀（生成 .json 和 .md）")
    return parser.parse_args()


def load_model(run_dir, device):
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    model = M.TransformerLM(
        vocab_size=cfg["vocab_size"], context_length=cfg["context_length"], num_layers=cfg["num_layers"],
        d_model=cfg["d_model"], num_heads=cfg["num_heads"], d_ff=cfg["d_ff"], rope_theta=cfg["rope_theta"],
        attn_gate=cfg.get("attn_gate", "none"), device=device,
    )
    model.load_state_dict(torch.load(os.path.join(run_dir, "ckpt.pt"), map_location=device)["model"])
    model.eval()
    return model, cfg


@torch.no_grad()
def analyze(model, x: torch.Tensor) -> dict:
    num_layers = len(model.layers)
    first_token_attn = []                       # 按调用顺序记录每一层的 F-Attn（每次前向依次调用第 0,1,2,... 层）
    max_act = [0.0] * num_layers
    gate_mean = [None] * num_layers
    gate_small = [None] * num_layers

    # 1. 替换 attention 函数：算法和原来一样，只是顺便把注意力权重记下来
    original_sdpa = M.scaled_dot_product_attention

    def recording_sdpa(Q, K, V, mask=None):
        d_k = Q.shape[-1]
        scores = torch.einsum("...qd,...kd->...qk", Q, K) / (d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(~mask, -torch.inf)
        weights = M.softmax(scores, -1)          # (num_heads, batch, T, T)
        # 位置 >= 1 的每个 query 分给 key 位置 0 的权重，再对 head、batch、query 取平均
        first_token_attn.append(weights[..., 1:, 0].mean().item())
        return torch.einsum("...qk,...kd->...qd", weights, V)

    # 2. hook：记录每一层输出（残差流）的最大绝对值，以及门控 sigmoid 之后的统计
    handles = []
    for i, layer in enumerate(model.layers):
        def block_hook(module, inputs, output, i=i):
            max_act[i] = output.abs().max().item()
        handles.append(layer.register_forward_hook(block_hook))
        if getattr(layer.attn, "attn_gate", "none") != "none":
            def gate_hook(module, inputs, output, i=i):
                g = torch.sigmoid(output)
                gate_mean[i] = g.mean().item()
                gate_small[i] = (g < 0.1).float().mean().item()
            handles.append(layer.attn.gate_proj.register_forward_hook(gate_hook))

    M.scaled_dot_product_attention = recording_sdpa
    try:
        model(x)
    finally:
        M.scaled_dot_product_attention = original_sdpa   # 无论如何都要换回原函数
        for h in handles:
            h.remove()

    return {
        "first_token_attn": first_token_attn,
        "max_activation": max_act,
        "gate_mean": gate_mean,
        "gate_small_frac": gate_small,
    }


def main():
    args = parse_args()
    results = {}
    x = None
    for run_dir in args.run_dirs:
        model, cfg = load_model(run_dir, args.device)
        if x is None:
            # 所有模型用同一批验证集文本，比较才公平
            val = np.load(cfg["val_data"], mmap_mode="r")
            T = cfg["context_length"]
            rng = np.random.default_rng(args.seed)
            starts = rng.integers(0, len(val) - T, size=args.num_seqs)
            x = torch.from_numpy(np.stack([val[s: s + T] for s in starts]).astype(np.int64)).to(args.device)
        r = analyze(model, x)
        r["attn_gate"] = cfg.get("attn_gate", "none")
        results[run_dir] = r
        fa = ", ".join(f"{v * 100:.1f}%" for v in r["first_token_attn"])
        ma = ", ".join(f"{v:.0f}" for v in r["max_activation"])
        print(f"{run_dir}\n  第一个 token 的注意力（逐层）: {fa}\n  最大激活（逐层）: {ma}")
        if r["attn_gate"] != "none":
            print(f"  门控平均值（逐层）: {', '.join(f'{v:.3f}' for v in r['gate_mean'])}   "
                  f"<0.1 的比例: {', '.join(f'{v * 100:.0f}%' for v in r['gate_small_frac'])}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump(results, f, indent=2)

    lines = [
        "# Attention sink / massive activation 分析",
        "",
        f"每个模型都用同一批 {args.num_seqs} 段验证集文本（每段 {T} 个 token，随机位置截取）。",
        "F-Attn = 位置 ≥1 的 query 分给窗口第一个 token 的平均注意力；M-Act = 每层输出残差流的最大绝对值。",
        "",
        "| 模型 | 门控 | F-Attn 逐层 | F-Attn 平均 | M-Act 逐层 | M-Act 平均 | 门控均值 | 门控 <0.1 比例 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for run_dir, r in results.items():
        fa = r["first_token_attn"]
        ma = r["max_activation"]
        if r["attn_gate"] != "none":
            gm = f"{np.mean(r['gate_mean']):.3f}"
            gs = f"{np.mean(r['gate_small_frac']) * 100:.0f}%"
        else:
            gm = gs = "—"
        lines.append(f"| {os.path.basename(run_dir)} | {r['attn_gate']} | "
                     f"{' / '.join(f'{v * 100:.1f}%' for v in fa)} | {np.mean(fa) * 100:.1f}% | "
                     f"{' / '.join(f'{v:.0f}' for v in ma)} | {np.mean(ma):.0f} | {gm} | {gs} |")
    with open(args.out + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
