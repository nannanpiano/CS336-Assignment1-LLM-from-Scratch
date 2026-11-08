import argparse 
import csv 
import json 
import math
import os 
import time 

import numpy as np 
import torch 

from cs336_basics.modules import TransformerLM 
from cs336_basics.optimizer import AdamW, learning_rate_schedule 
from cs336_basics.utils import cross_entropy, gradient_clipping, data_loading, save_checkpoint, load_checkpoint 


# 解析命令行参数，所有超参数都从命令行传入
def parse_args():
    parser = argparse.ArgumentParser(description="训练 Transformer 语言模型")
    # 数据
    parser.add_argument("--train_data", type=str, default="data/tokenized/tinystories_train.npy")
    parser.add_argument("--val_data", type=str, default="data/tokenized/tinystories_valid.npy")
    # 模型
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--attn_gate", type=str, default="none", choices=["none", "elementwise", "headwise"],
                        help="attention 输出门控（Gated Attention）：none 不加，elementwise 逐元素，headwise 每个头一个")
    # 优化器与学习率调度
    parser.add_argument("--lr", type=float, default=1e-3, help="最大学习率：alpha_max")
    parser.add_argument("--min_lr", type=float, default=1e-4, help="最小学习率：alpha_min")
    parser.add_argument("--warmup_iters", type=int, default=200)
    parser.add_argument("--cosine_iters", type=int, default=None, help="余弦退化结束的步数，默认等于max_iters")
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    # 训练过程
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--device", type=str, default=None, help="默认自动选择 cuda > mps > cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile", action="store_true", help="用 torch.compile 编译模型，融合小算子加速训练")
    parser.add_argument("--tf32", action="store_true", help="将矩阵乘法改为 TF32，命令里写了 --tf32，值就是True，反之False")  
    parser.add_argument("--bf16", action="store_true", help="将矩阵乘法改为 BF16")
    # 日志、评估与检查点
    parser.add_argument("--out_dir", type=str, default="runs/default", help="保存config、log和checkpoint的目录")
    parser.add_argument("--log_interval", type=int, default=10, help="每隔多少步记录一次训练loss")
    parser.add_argument("--eval_interval", type=int, default=200, help="每隔多少步在验证集上评估一次")
    parser.add_argument("--eval_iters", type=int, default=20, help="每次评估采样多少个batch后取平均")
    parser.add_argument("--ckpt_interval", type=int, default=1000, help="每隔多少步保存一次checkpoint")
    parser.add_argument("--resume", type=str, default=None, help="从这个检查点恢复训练")
    return parser.parse_args()

# 自动选择设备
def get_device(name: str | None) -> str:
    if name is not None:
        return name 
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# 在验证集上评估：随机采样 eval_iters 个 batch，取平均 loss
@ torch.no_grad()
def evaluate(model, val_data, args, device, device_type) -> float:
    model.eval()
    losses = []
    for _ in range(args.eval_iters):
        x, y = data_loading(val_data, args.batch_size, args.context_length, device)
        # 和训练时用同样的精度计算，train loss和 val loss 才能直接比较
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=args.bf16):
            logits = model(x)
            loss = cross_entropy(logits, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

def main():
    args = parse_args()
    if args.cosine_iters is None:
        args.cosine_iters = args.max_iters 
    device = get_device(args.device)   # device有时候是 "cuda:2"
    device_type = device.split(":")[0]   # "cuda:2" -> "cuda"，autocast 只接受不带编号的设备类型
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.tf32:
        if device_type == "cuda":
            torch.set_float32_matmul_precision('high')   # 全局开关
        else:
            print("TF32 只在 CUDA 上开启，这次忽略。")

    # 保存这次实验的全部超参数，之后对比曲线时知道每条线对应什么配置
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args) | {"device": device}, f, indent=2)

    # 用内存映射加载数据：只在采样时按需从磁盘读取，不把整个数据集读进内存
    train_data = np.load(args.train_data, mmap_mode="r")
    val_data = np.load(args.val_data, mmap_mode="r")
    print(f"训练集 {len(train_data):,} tokens, 验证集 {len(val_data):,} tokens，设备 {device}")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
        attn_gate=args.attn_gate,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2), 
        eps=args.eps,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量 {num_params / 1e6:.2f}M")

    # 如果指定了检查点，就恢复模型和优化器，并从保存时的步数继续
    start_iter = 0
    if args.resume is not None:
        start_iter = load_checkpoint(args.resume, model, optimizer)
        print(f"从 {args.resume} 恢复，继续第 {start_iter} 步")
    
    # torch.compile 返回一个包装对象，它的 state_dict 参数名会多出 "_orig_mod." 前缀。
    # 所以保留未编译模型的引用 raw_model：训练/评估用编译后的 model，保存 checkpoint 用 raw_model。
    # 两者共享同一组参数张量，优化器不用改。
    raw_model = model 
    if args.compile:
        print("编译模型中，前几步会比较慢")
        # 调用 torch.compile(model) 时并没有编译，真正的编译是在第一次运行时才发生的。
        model = torch.compile(model)

    # 日志文件：每行一条记录，之后可以读进来画学习曲线
    log_path = os.path.join(args.out_dir, "log.csv")
    log_file = open(log_path, "a", newline="")
    writer = csv.writer(log_file)
    if os.path.getsize(log_path) == 0:
        # tokens_per_sec 是吞吐率
        writer.writerow(["step", "wall_time", "lr", "train_loss", "val_loss", "tokens_per_sec"])

    start_time = time.time()
    # 上一次打印日志时的时间和步数，用来计算这段时间内的训练吞吐量
    last_log_time = start_time 
    last_log_step = start_iter 
    for it in range(start_iter, args.max_iters):
        # 1. 按学习率调度设置这一步的学习率
        lr = learning_rate_schedule(it, args.lr, args.min_lr, args.warmup_iters, args.cosine_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # 2. 采样一个 batch，前向计算 loss
        x, y = data_loading(train_data, args.batch_size, args.context_length, device)
        # autocast：矩阵乘法自动用 BF16 计算，参数仍是 FP32；enabled=False 时这个 with 块不起作用
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=args.bf16):
            logits = model(x)     # (batch_size, context_length, vocab_size)
            loss = cross_entropy(logits, y)

        # 3. 反向传播，梯度裁剪，更新参数
        optimizer.zero_grad()
        loss.backward() 
        gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step() 

        # 4. 记录日志
        step = it + 1   # 已完成的步数
        is_eval = step % args.eval_interval == 0 or step == args.max_iters 
        if step % args.log_interval == 0 or is_eval:
            # loss.item() 会等 GPU 算完，所以必须先调用它，再取时间
            train_loss = loss.item() 
            now = time.time()
            wall_time = now - start_time 
            # 吞吐量 = 这段时间处理的 token 数 / 这段时间的秒数
            tokens = (step - last_log_step) * args.batch_size * args.context_length
            tokens_per_sec = tokens / (now - last_log_time)

            val_loss = evaluate(model, val_data, args, device, device_type) if is_eval else None 
            writer.writerow([step, f"{wall_time:.2f}", lr, train_loss,
                             "" if val_loss is None else val_loss, f"{tokens_per_sec:.0f}"])
            log_file.flush()
            msg = (f"step {step:6d} | time {wall_time:8.1f}s | lr {lr:.2e} | "
                   f"train_loss {train_loss:.4f} | {tokens_per_sec:8.0f} tok/s")
            if val_loss is not None:
                msg += f" | val_loss {val_loss:.4f}"
            print(msg)

            # 发散检查：loss 变成 NaN 或 inf 就停止训练
            # 放在写完日志之后（log.csv 里能看到发散的那一步），放在保存 checkpoint 之前（不会用坏参数覆盖好的 ckpt.pt）
            diverged = not math.isfinite(train_loss) or (val_loss is not None and not math.isfinite(val_loss))
            # math.isfinite(num) 判断 num 是否有限
            if diverged:
                print(f"第 {step} 步 loss 变成了 {train_loss}, 训练发散，停止训练（不保存 checkpoint）")
                break
            
            # 下一段的计时起点放在评估之后，这样评估时间不会算进下一段
            last_log_time = time.time()
            last_log_step = step


        # 5. 定期保存检查点
        if step % args.ckpt_interval == 0 or step == args.max_iters:
            # 保存 checkpoint 时使用原始模型
            save_checkpoint(raw_model, optimizer, step, os.path.join(args.out_dir, "ckpt.pt"))

    log_file.close()

if __name__ == "__main__":
    main()

