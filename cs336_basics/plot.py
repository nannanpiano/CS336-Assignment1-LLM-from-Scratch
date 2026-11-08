"""
画学习曲线：把一个或多个实验目录（train.py 的 --out_dir）的 log.csv 画在同一张图上。

一张图有左右两个子图：
    左：横轴是训练步数（step）
    右：横轴是真实时间（wall-clock，单位分钟）
handout §7.1 要求两种横轴的曲线都要有。纵轴默认是 val loss。

用法示例：
    # 画一个实验
    uv run python -m cs336_basics.plot runs/ts_mps --out figures/ts_mps_baseline.png

    # 对比多个实验（每个实验一条线），自定义图例名字，限制纵轴上限（防止发散的实验把纵轴撑得太大）
    uv run python -m cs336_basics.plot runs/lr1e-3 runs/lr3e-3 runs/lr1e-2 \
        --labels "lr=1e-3" "lr=3e-3" "lr=1e-2" --ymax 4 --out figures/lr_sweep.png

    # 同时画上平滑后的 train loss（虚线），用来看 train 和 val 的差距
    uv run python -m cs336_basics.plot runs/ts_mps --train --out figures/ts_mps_train_val.png

除了画图，还会在终端打印一张汇总表（最终/最好的 val loss、总时间、吞吐），方便直接抄进实验记录。
"""

import argparse
import csv
import math
import os
import statistics

import matplotlib

# 使用不需要显示器的后端：只把图保存成文件，不弹窗。
# 这样在没有图形界面的服务器（比如 L40）上也能运行。必须在 import pyplot 之前设置。
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="把多个实验的 log.csv 画成学习曲线")
    parser.add_argument("run_dirs", nargs="+", help="一个或多个实验目录（train.py 的 --out_dir），每个目录里要有 log.csv")
    parser.add_argument("--labels", nargs="+", default=None, help="图例名字，数量要和 run_dirs 一样；默认用目录名")
    parser.add_argument("--out", type=str, default="figures/plot.png", help="图片保存路径（目录不存在会自动创建）")
    parser.add_argument("--title", type=str, default=None, help="整张图的标题")
    parser.add_argument("--train", action="store_true", help="额外画出平滑后的 train loss（虚线）")
    parser.add_argument("--smooth", type=float, default=0.9,
                        help="train loss 的指数滑动平均系数，越接近 1 越平滑（只在 --train 时有用）")
    parser.add_argument("--ymin", type=float, default=None, help="纵轴下限")
    parser.add_argument("--ymax", type=float, default=None, help="纵轴上限，对比里有发散的实验时很有用")
    parser.add_argument("--logy", action="store_true", help="纵轴用对数刻度")
    return parser.parse_args()


def to_float(s: str) -> float | None:
    """把 CSV 里的一格转成 float。空字符串（这一步没有评估）返回 None；"nan"/"inf" 会正常转成 float 的 nan/inf。"""
    s = s.strip()
    if s == "":
        return None
    return float(s)


def read_log(run_dir: str) -> dict[str, list[float]]:
    """
    读取一个实验目录下的 log.csv，返回按列整理好的数据：
        train_step, train_time, train_loss:  每一行都有（每 log_interval 步一行）
        val_step,   val_time,   val_loss:    只有做了评估的那几行才有
        tokens_per_sec:                      吞吐（旧版 log.csv 没有这一列，就是空列表）
    """
    path = os.path.join(run_dir, "log.csv")
    data = {k: [] for k in ["train_step", "train_time", "train_loss",
                            "val_step", "val_time", "val_loss", "tokens_per_sec"]}
    with open(path, newline="") as f:
        # DictReader 把每一行读成字典，键是表头的列名，比如 row["val_loss"]
        for row in csv.DictReader(f):
            step = int(row["step"])
            minutes = float(row["wall_time"]) / 60   # 秒 -> 分钟，横轴更好读
            data["train_step"].append(step)
            data["train_time"].append(minutes)
            data["train_loss"].append(to_float(row["train_loss"]))

            val = to_float(row["val_loss"])
            if val is not None:          # 只有评估的那一步才有 val loss
                data["val_step"].append(step)
                data["val_time"].append(minutes)
                data["val_loss"].append(val)

            # 旧的 log.csv（比如 runs/ts_mps）没有这一列，用 .get 读取就不会报错
            tps = row.get("tokens_per_sec")
            if tps:
                data["tokens_per_sec"].append(float(tps))
    return data


def ema(values: list[float], alpha: float) -> list[float]:
    """
    指数滑动平均（EMA），用来平滑噪声很大的 train loss：
        smoothed_t = alpha * smoothed_{t-1} + (1 - alpha) * value_t
    alpha 越接近 1，越依赖过去的值，曲线越平滑（但对变化的反应也越慢）。
    遇到 nan/inf（训练发散）时，从这里开始都输出 nan，让曲线在发散处断开，而不是被平均“掩盖”掉。
    """
    out = []
    smoothed = None
    for v in values:
        if not math.isfinite(v) or (smoothed is not None and not math.isfinite(smoothed)):
            smoothed = float("nan")    # 当前值发散了，或者之前已经发散过：之后一直是 nan
        elif smoothed is None:
            smoothed = v               # 第一个值直接作为起点
        else:
            smoothed = alpha * smoothed + (1 - alpha) * v
        out.append(smoothed)
    return out


def summarize(label: str, data: dict[str, list[float]], width: int) -> str:
    """生成汇总表里的一行：最终 val loss、最好的 val loss（及其步数）、总步数、总时间、吞吐中位数。"""
    vals = data["val_loss"]
    finite_vals = [(v, s) for v, s in zip(vals, data["val_step"]) if math.isfinite(v)]
    final_val = f"{vals[-1]:.4f}" if vals else "-"
    if finite_vals:
        best_val, best_step = min(finite_vals)
        best = f"{best_val:.4f} @ {best_step}"
    else:
        best = "-"
    steps = data["train_step"][-1] if data["train_step"] else 0
    minutes = data["train_time"][-1] if data["train_time"] else 0
    # 吞吐用中位数而不是平均数：第一段包含编译/预热，数值偏低，中位数不受这种个别值影响
    tps = f"{statistics.median(data['tokens_per_sec']):.0f}" if data["tokens_per_sec"] else "-"
    diverged = any(not math.isfinite(v) for v in data["train_loss"] + vals)
    note = "  (diverged)" if diverged else ""
    return f"{label:<{width}} {final_val:>10} {best:>18} {steps:>8} {minutes:>9.1f} {tps:>12}{note}"


def main():
    args = parse_args()
    labels = args.labels or [os.path.basename(os.path.normpath(d)) for d in args.run_dirs]
    if len(labels) != len(args.run_dirs):
        raise ValueError(f"--labels 有 {len(labels)} 个，但实验目录有 {len(args.run_dirs)} 个，数量要一致")

    # 一行两列的子图：axes[0] 按步数，axes[1] 按时间；sharey=True 让两边纵轴刻度一致，方便对照
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    # 图上的文字用英文：matplotlib 默认字体不含中文，中文会显示成方框；而且以后放进 README/报告也更通用
    x_keys = [("step", "step"), ("time", "wall-clock time (min)")]

    # 汇总表第一列的宽度：按最长的实验名字来定，保证各列对齐
    width = max(len(label) for label in labels) + 2
    print(f"{'run':<{width}} {'final val':>10} {'best val @ step':>18} {'steps':>8} {'minutes':>9} {'median tok/s':>12}")

    plotted_y = []   # 记录画上去的所有有限 y 值，后面只给了 --ymax 时用来确定纵轴下限
    for i, (run_dir, label) in enumerate(zip(args.run_dirs, labels)):
        data = read_log(run_dir)
        color = f"C{i}"   # matplotlib 默认色板的第 i 个颜色；同一个实验的 train/val 用同一种颜色
        smoothed_train = ema(data["train_loss"], args.smooth) if args.train else []
        for ax, (x_key, _) in zip(axes, x_keys):
            # val loss：实线 + 小圆点（评估点比较稀疏，圆点能看出具体在哪几步评估过）
            ax.plot(data[f"val_{x_key}"], data["val_loss"], color=color, marker="o", markersize=3,
                    linewidth=1.5, label=f"{label} (val)" if args.train else label)
            if args.train:
                # train loss：先平滑，再用同色虚线画，半透明，不抢 val 的视觉重点
                ax.plot(data[f"train_{x_key}"], smoothed_train, color=color,
                        linestyle="--", linewidth=1, alpha=0.7, label=f"{label} (train, EMA {args.smooth})")
        plotted_y += [v for v in data["val_loss"] + smoothed_train if math.isfinite(v)]
        print(summarize(label, data, width))

    # 纵轴范围：
    #   只给 --ymax 时，matplotlib 会按“所有数据”的范围自动算下限。发散的实验里有上千的 loss，
    #   自动算出的下限会变成负几百，正常曲线被压成一条线。所以这时用画出来的最小值往下留 5% 当下限。
    ymin, ymax = args.ymin, args.ymax
    if ymax is not None and ymin is None and plotted_y:
        lowest = min(plotted_y)
        ymin = lowest - 0.05 * (ymax - lowest)

    for ax, (_, xlabel) in zip(axes, x_keys):
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)
        if args.logy:
            ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)   # 传 None 表示这一边不限制，由 matplotlib 自动决定
    axes[0].set_ylabel("loss (per-token cross-entropy)")
    axes[1].legend(fontsize=8)
    if args.title:
        fig.suptitle(args.title)
    fig.tight_layout()   # 自动调整子图间距，避免标签互相重叠

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"\n图片已保存到 {args.out}")


if __name__ == "__main__":
    main()
