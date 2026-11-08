from collections.abc import Callable, Iterable 
from typing import Optional 
import torch 
import math 


# 示例：SGD 优化器
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}    # 将额外的参数以字典形式传给基类的构造函数
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure() 
        for group in self.param_groups:
            lr = group["lr"]    # 获取学习率
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]    # 获取与p关联的状态
                t = state.get("t", 0)   # 从状态中获取迭代次数，若无则为0
                grad = p.grad.data     # 获取损失关于 p 的梯度
                p.data -= lr / math.sqrt(t + 1) * grad    # 就地修改权重
                state["t"] = t + 1    # 迭代次数 +1
        return loss 

# 实现 AdamW 优化器
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure() 
        for group in self.param_groups:
            lr, weight_decay, betas, eps = group["lr"], group["weight_decay"], group["betas"], group["eps"]
            beta1, beta2 = betas 
            for p in group["params"]:
                if p.grad is None:
                    continue 
                state = self.state[p]
                # 这里迭代次数 t 要初始化为 1
                # t, m, v = state.get("t", 1), state.get("m", 0), state.get("v", 0)
                t = state.get("t", 1)
                # m, v 初始化为与 p形状相同的zero tensor
                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                m, v = state["m"], state["v"]
                alpha_t = lr * ((1 - beta2**t)**0.5) / (1 - beta1**t)
                p.data -= lr * weight_decay * p.data 
                # m = beta1 * m + (1 - beta1) * p.grad 
                # v = beta2 * v + (1 - beta2) * (p.grad**2)
                # 就地修改 m 和 v
                m.mul_(beta1).add_((1-beta1)*p.grad)
                v.mul_(beta2).add_((1-beta2)*(p.grad**2))
                p.data -= alpha_t * m / (v**0.5 + eps)
                state["t"] = t + 1
        return loss 

# 实现带预热的余弦学习率调度
def learning_rate_schedule(t: int, alpha_max: float, alpha_min: float, T_w: int, T_c: int):
    if t < T_w:
        return alpha_max * t / T_w 
    elif T_w <= t <= T_c:
        return alpha_min + 1/2 * (1 + math.cos(math.pi * (t-T_w)/(T_c-T_w))) * (alpha_max - alpha_min)
    else:
        return alpha_min