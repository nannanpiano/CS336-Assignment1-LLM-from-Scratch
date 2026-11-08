import torch 
import torch.nn as nn
from cs336_basics.modules import softmax
import numpy as np
import os
import typing


# 交叉熵损失函数
def cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # logits: [..., vocab_size]    target: [...]   每个位置是正确 token 的下标
    logits = logits.to(torch.float32)
    maximal = torch.amax(logits, dim=-1, keepdim=True).detach()   # 从计算图中摘掉，减小计算量
    # 减掉的最大值只是为了防止 exp 溢出，不参与求导
    modified = logits - maximal    # 减去最大值，防止 exp 溢出  [..., vocab_size]
    exponent = torch.exp(modified)   # [..., vocab_size]
    # torch.gather表示沿dim指定的维度按下标取值
    loss = - torch.gather(modified, dim=-1, index=target[..., None]).squeeze(dim=-1) + torch.log(torch.sum(exponent, dim=-1))
    avg_loss = loss.mean()
    return avg_loss


# 梯度裁剪
def gradient_clipping(params, M: float, eps: float = 1e-6):
    # params 可能是只能遍历一次的生成器（如 model.parameters()），先转成 list
    grads = [p.grad for p in params if p.grad is not None]
    if len(grads) == 0:
        return
    # 所有参数的梯度合在一起的第二范数
    total_norm = torch.sqrt(sum(torch.sum(g.detach() ** 2) for g in grads))  # torch.sum返回的是 0 维张量
    if total_norm > M:
        scale = M / (total_norm + eps)
        for g in grads:
            g.mul_(scale)




# 实现数据加载
def data_loading(x: np.array, 
                 batch_size: int,
                 context_length: int, 
                 device: str
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    total_len = x.shape[0] 
    # 随机采样 batch_size 个起点，范围 [0, total_len - context_length - 1]
    starts = np.random.randint(0, total_len - context_length, size=(batch_size, 1)) # 输入序列的起点, (batch_size, 1)
    # 广播得到下标矩阵，每行是：起点 + [0, 1, ..., context_length-1]
    idx = starts + np.arange(context_length)
    # 只在 numpy (x)上切除这一小块，再传承 tenser 放到 device 上
    inputs = torch.from_numpy(x[idx].astype(np.int64)).to(device)  # 转成int64，因为embedding索引需要
    targets = torch.from_numpy(x[idx + 1].astype(np.int64)).to(device)
    return inputs, targets


# 保存检查点 save checkpoint
def save_checkpoint(model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    iteration: int,
                    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
                    ):
    # 把模型参数、优化器状态和当前迭代步数打包成一个字典
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoint, out)   # 序列化写入文件

# 加载检查点 load checkpoint
def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
                    model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    ) -> int:
    checkpoint = torch.load(src)   # 读回保存的checkpoint字典
    model.load_state_dict(checkpoint["model"])    
    optimizer.load_state_dict(checkpoint["optimizer"]) 
    return checkpoint["iteration"]
