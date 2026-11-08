#!/bin/sh
# 学习率扫描：在当前可见的那张 GPU 上，按顺序把给定的几个学习率各训练一次。
# 数据集用环境变量 DATASET 选：tinystories（默认）或 owt（OpenWebText，词表 32000）。
#
# 用法（在仓库根目录下运行；用哪张卡由调用方通过 CUDA_VISIBLE_DEVICES 决定，进程内都是 cuda:0）：
#   GPU3（被 gpuhold 占着，要用 run 包装，跑完自动占回）：
#     /usr/bin/python3 ~/workspace/gpuhold/gpuhold.py run --gpus 3 -- sh scripts/lr_sweep.sh 1e-3 3e-4
#   其他空闲的卡（比如 GPU2）：
#     CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 sh scripts/lr_sweep.sh 3e-3 1e-2
#   OpenWebText：前面加 DATASET=owt，比如
#     DATASET=owt CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 sh scripts/lr_sweep.sh 2e-3
#
# 固定的设置（只有学习率在变，其余都和 baseline 一样）：
#   batch 32 × 40000 步 × context 256 = 327,680,000 token（handout 要求的 GPU 训练量）
#   bf16 + torch.compile（测速结果：约为 fp32 的 3.07 倍）
#   min_lr = lr / 10：余弦衰减的终点按比例跟着最大学习率走，不同 lr 之间才公平
#   每 1000 步评估一次，每次 100 个 batch（约 82 万 token），曲线比默认的 20 个 batch 稳
#   模型结构（4 层、d_model 512……）两个数据集完全一样，只有词表大小跟着各自的 tokenizer 变
#   输出目录 runs/ts_gpu_lr<学习率> 或 runs/owt_gpu_lr<学习率>，比如 runs/ts_gpu_lr1e-3
#
# 另外两个可选的环境变量（Gated Attention 对比实验用）：
#   ATTN_GATE=none|elementwise|headwise   attention 门控，默认 none；不是 none 时目录名加上 _gate<类型>
#   SEED=<整数>                           随机种子，默认 0（和 train.py 一样）；设置了就在目录名加上 _seed<种子>
#   例：ATTN_GATE=elementwise SEED=1 sh scripts/lr_sweep.sh 1e-3   ->  runs/ts_gpu_lr1e-3_gateelementwise_seed1
#   不设置这两个变量时，行为和目录名都和以前完全一样

set -u
if [ $# -eq 0 ]; then
  echo "用法: [DATASET=tinystories|owt] sh scripts/lr_sweep.sh <lr1> [lr2 ...]"
  exit 2
fi

DATASET=${DATASET:-tinystories}
case "$DATASET" in
  tinystories) data_name=tinystories; vocab_size=10000; prefix=ts ;;
  owt)         data_name=owt;         vocab_size=32000; prefix=owt ;;
  *) echo "未知的 DATASET: $DATASET（只支持 tinystories 或 owt）"; exit 2 ;;
esac

ATTN_GATE=${ATTN_GATE:-none}
SEED=${SEED:-}
suffix=""
[ "$ATTN_GATE" != "none" ] && suffix="${suffix}_gate${ATTN_GATE}"
[ -n "$SEED" ] && suffix="${suffix}_seed${SEED}"

# 服务器上非交互 shell 不会自动加载 uv 的 PATH，代理变量会让下载卡住，这里统一处理
export PATH="$HOME/.local/bin:$PATH"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
# 输出重定向到文件时 Python 默认会攒一大块再写，加这个让日志实时出现，方便随时查看进度
export PYTHONUNBUFFERED=1

for lr in "$@"; do
  min_lr=$(awk "BEGIN { print $lr / 10 }")
  out_dir="runs/${prefix}_gpu_lr${lr}${suffix}"
  # 目录里已经有日志，说明这个实验跑过或者正在别的卡上跑：跳过，避免重复训练、往旧日志里追加
  if [ -e "$out_dir/log.csv" ]; then
    echo "===== $(date '+%m-%d %H:%M:%S')  跳过 $out_dir（已经存在 log.csv）"
    continue
  fi
  echo "===== $(date '+%m-%d %H:%M:%S')  $DATASET  lr=$lr  min_lr=$min_lr  gate=$ATTN_GATE  seed=${SEED:-0}  ->  $out_dir  (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-未设置})"
  uv run --no-sync python -m cs336_basics.train \
    --train_data "data/tokenized/${data_name}_train.npy" --val_data "data/tokenized/${data_name}_valid.npy" \
    --vocab_size "$vocab_size" \
    --attn_gate "$ATTN_GATE" --seed "${SEED:-0}" \
    --device cuda --bf16 --compile \
    --lr "$lr" --min_lr "$min_lr" --warmup_iters 200 \
    --batch_size 32 --max_iters 40000 \
    --log_interval 50 --eval_interval 1000 --eval_iters 100 --ckpt_interval 5000 \
    --out_dir "$out_dir"
  rc=$?    # 必须紧跟在训练命令后面取，否则 $? 会被下一条命令覆盖
  # 发散的实验 train.py 会自己停下并正常退出，这里不中断队列，继续跑下一个学习率
  echo "===== $(date '+%m-%d %H:%M:%S')  lr=$lr 结束（返回码 $rc）"
done
