"""
训练吞吐量测速：对比不同精度/加速选项（FP32、TF32、BF16、torch.compile）下 train.py 每秒处理多少 token。

做法（尽量让数字可信）：
    1. 每种配置都调用真正的 train.py 跑 --steps 步（模型、数据、优化器和正式训练完全一样），
       吞吐量直接读 train.py 写的 log.csv 里的 tokens_per_sec 列。
    2. 去掉前 --warmup_steps 步的日志：这段包含 GPU 预热，以及 torch.compile 的编译时间（编译发生在
       第一次前向/反向时，不在 torch.compile(model) 那一行），会把数字拉低。
    3. 每种配置重复 --repeats 次，并且按“轮”交错执行（第 1 轮把所有配置各跑一次，再第 2 轮……），
       而不是同一配置连着跑 3 次。这样机器状态随时间的变化（温度、别人的任务）会平均摊到每个配置上。
    4. 单次的吞吐取稳定段的中位数；多次重复之间也取中位数，并报告最小/最大值，看波动有多大。
    5. 每轮开始前记录 GPU 上有没有别人的进程，写进结果，方便判断数字有没有被干扰。

用法（在服务器上，配合 gpuhold 拿到 GPU3）：
    /usr/bin/python3 ~/workspace/gpuhold/gpuhold.py run --gpus 3 -- \
        uv run --no-sync python -m cs336_basics.bench_throughput --device cuda --out results/throughput_l40

本地快速试跑（只检查脚本能不能跑通，数字没有意义）：
    uv run python -m cs336_basics.bench_throughput --steps 40 --warmup_steps 20 --repeats 1 \
        --configs fp32 bf16 --out /tmp/bench_test

输出：
    <out>.csv  每次运行一行（配置、第几轮、稳定段吞吐中位数……），原始数据，方便以后重新分析
    <out>.md   汇总表（每种配置的中位数、最小/最大、相对 FP32 的倍数），可以直接贴进 README/报告
"""

import argparse
import csv
import os
import shutil
import statistics
import subprocess
import sys
import time

# 要对比的配置：名字 -> 传给 train.py 的额外参数。顺序就是表格里的顺序，每一项只比上一项多开一个开关，
# 这样相邻两行的差别就是这个开关的效果。
CONFIGS = {
    "fp32": [],
    "tf32": ["--tf32"],
    "bf16": ["--bf16"],
    "bf16+compile": ["--bf16", "--compile"],
    "bf16+compile+tf32": ["--bf16", "--compile", "--tf32"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="对比不同精度/加速选项下的训练吞吐量")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS),
                        help="要测的配置，默认全部")
    parser.add_argument("--repeats", type=int, default=3, help="每种配置重复几次")
    parser.add_argument("--steps", type=int, default=300, help="每次跑多少步")
    parser.add_argument("--warmup_steps", type=int, default=50, help="去掉前多少步的日志（预热+编译）")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--device", type=str, default=None, help="传给 train.py 的 --device")
    parser.add_argument("--out", type=str, default="results/throughput", help="结果文件前缀（会生成 .csv 和 .md）")
    parser.add_argument("--work_dir", type=str, default="runs/bench_tmp", help="每次运行的临时输出目录")
    return parser.parse_args()


def gpu_snapshot() -> str:
    """
    记录当前所有 GPU 上的计算进程，格式如 "GPU1: pid 123 (13520MiB)"。
    nvidia-smi 不受 CUDA_VISIBLE_DEVICES 影响，看到的是整台机器的物理卡号。
    没有 nvidia-smi（比如在 Mac 上）就返回说明文字。
    """
    if shutil.which("nvidia-smi") is None:
        return "no nvidia-smi"

    def query(q, kind):
        r = subprocess.run(["nvidia-smi", f"--query-{kind}={q}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True)
        return [[c.strip() for c in line.split(",")] for line in r.stdout.strip().splitlines() if line.strip()]

    index_of = {uuid: idx for idx, uuid in query("index,uuid", "gpu")}
    procs = [f"GPU{index_of.get(uuid, '?')}: pid {pid} ({mb}MiB)"
             for uuid, pid, mb in query("gpu_uuid,pid,used_memory", "compute-apps")]
    return " ; ".join(procs) or "no compute apps"


def git_commit() -> str:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or "unknown"


def run_once(name: str, extra: list[str], args) -> dict:
    """跑一次 train.py，返回这次的吞吐统计。"""
    out_dir = os.path.join(args.work_dir, name)
    shutil.rmtree(out_dir, ignore_errors=True)   # 只删本脚本自己的临时目录，保证 log.csv 是这次新写的
    cmd = [sys.executable, "-m", "cs336_basics.train",
           "--max_iters", str(args.steps),
           "--log_interval", str(args.log_interval),
           "--eval_interval", str(args.steps),       # 只在最后一步评估一次；评估时间本来就不计入吞吐
           "--ckpt_interval", str(args.steps),
           "--batch_size", str(args.batch_size),
           "--context_length", str(args.context_length),
           "--out_dir", out_dir] + extra
    if args.device:
        cmd += ["--device", args.device]

    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:], sep="\n")
        raise RuntimeError(f"{name} 运行失败（返回码 {r.returncode}）")

    # 读 train.py 写的 log.csv，只保留预热之后的吞吐
    with open(os.path.join(out_dir, "log.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    stable = [float(row["tokens_per_sec"]) for row in rows if int(row["step"]) > args.warmup_steps]
    if not stable:
        raise RuntimeError(f"{name}: 预热之后没有日志行，请调大 --steps 或调小 --warmup_steps")
    final_loss = float(rows[-1]["train_loss"])

    # checkpoint 有两百多 MB，测速用不到，删掉省磁盘
    ckpt = os.path.join(out_dir, "ckpt.pt")
    if os.path.exists(ckpt):
        os.remove(ckpt)

    return {
        "tok_per_sec": statistics.median(stable),
        "n_points": len(stable),
        "final_train_loss": final_loss,
        "wall_seconds": elapsed,       # 整次运行的真实时间，包括启动、编译和最后的评估
    }


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    commit = git_commit()
    print(f"代码版本 {commit}；每次 {args.steps} 步，去掉前 {args.warmup_steps} 步；"
          f"batch {args.batch_size} × context {args.context_length}；重复 {args.repeats} 轮\n")

    records = []
    for rep in range(1, args.repeats + 1):
        snapshot = gpu_snapshot()
        print(f"--- 第 {rep} 轮  GPU 进程: {snapshot}")
        for name in args.configs:
            res = run_once(name, CONFIGS[name], args)
            res.update(config=name, repeat=rep, gpu_snapshot=snapshot, commit=commit)
            records.append(res)
            print(f"  {name:<20} {res['tok_per_sec']:>10.0f} tok/s   loss {res['final_train_loss']:.4f}   "
                  f"整次 {res['wall_seconds']:.0f}s", flush=True)

    # 原始数据：每次运行一行
    fields = ["config", "repeat", "tok_per_sec", "n_points", "final_train_loss", "wall_seconds", "commit", "gpu_snapshot"]
    with open(args.out + ".csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in records)

    # 汇总表：每种配置多次重复的中位数、最小、最大，以及相对 FP32（或第一种配置）的倍数
    base_name = "fp32" if "fp32" in args.configs else args.configs[0]
    summary = {}
    for name in args.configs:
        vals = [r["tok_per_sec"] for r in records if r["config"] == name]
        summary[name] = (statistics.median(vals), min(vals), max(vals))
    base = summary[base_name][0]

    lines = [
        f"# 训练吞吐量对比（代码版本 {commit}）",
        "",
        f"batch {args.batch_size} × context {args.context_length}，每次 {args.steps} 步，去掉前 {args.warmup_steps} 步，"
        f"重复 {args.repeats} 轮取中位数。",
        "",
        f"| 配置 | tok/s（中位数） | 最小 | 最大 | 相对 {base_name} |",
        "|---|---|---|---|---|",
    ]
    for name, (med, lo, hi) in summary.items():
        lines.append(f"| {name} | {med:,.0f} | {lo:,.0f} | {hi:,.0f} | {med / base:.2f}× |")
    lines += ["", "每轮开始时 GPU 上的进程："]
    for rep in range(1, args.repeats + 1):
        snap = next(r["gpu_snapshot"] for r in records if r["repeat"] == rep)
        lines.append(f"- 第 {rep} 轮：{snap}")
    with open(args.out + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines))
    print(f"\n原始数据：{args.out}.csv\n汇总表：{args.out}.md")


if __name__ == "__main__":
    main()
