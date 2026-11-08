import torch 
import torch.nn as nn 
import math
from einops import rearrange, reduce, einsum

# 线性操作
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None=None, dtype: torch.dtype | None=None):
        super().__init__()
        self.in_features = in_features 
        self.out_features = out_features
        std = (2/(in_features+out_features))**0.5
        self.weight = nn.Parameter(nn.init.trunc_normal_(
            torch.empty((out_features, in_features), dtype=dtype, device=device),
            mean=0.0,
            std=std,
            a=-3*std,
            b=3*std,)
        )
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")  # x_out = W @ x_in

# print(Linear(1, 2).state_dict().keys())


# 词嵌入
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device: torch.device | None=None, dtype: torch.dtype | None=None):
        super().__init__() 
        self.num_embeddings = num_embeddings    # 词表大小
        self.embedding_dim = embedding_dim     # 嵌入维度 d
        self.weight = nn.Parameter(nn.init.trunc_normal_(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype),
            mean=0.0,
            std=1,
            a=-3,
            b=3,
        ))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]    # 取出第 token_ids 行，得到 (token_ids.shape, embedding_dim)


# RMS Norm
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model 
        self.eps = eps    # epsilon
        self.weight = nn.Parameter(torch.ones(d_model, dtype=dtype, device=device))   # 缩放系数 gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype 
        x = x.to(torch.float32)
        result = (x / ((reduce(x ** 2, "... d -> ... ()", "sum") / self.d_model + self.eps) ** 0.5)) * self.weight 
        return result.to(in_dtype)


# SiLU
class SiLU(nn.Module):
    def __init__(self):
        super().__init__() 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


# SwiGLU
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None=None,  device=None, dtype=None):
        super().__init__()
        self.d_model = d_model 
        self.d_ff = d_ff if d_ff is not None else max(64, round(8/3 * d_model/64) * 64)   # 必须是int类型；加一个max防止d_ff=0
        self.w1 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)        
        self.w3 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)  
        self.silu = SiLU()  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.w2(self.silu(self.w1(x)) * self.w3(x))
        return result




# RoPE 
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__() 
        self.theta = theta    
        self.d_k = d_k       # query 和 key 的维度
        self.max_seq_len = max_seq_len 
        position_i = torch.arange(max_seq_len, device=device, dtype=torch.float32)   # 0, 1, ..., max_seq_len-1
        frequency_j = 1 / theta ** (torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k)  # 指数取0/d_k, 2/d_k, ..., (d-2)/d_k
        angle = einsum(position_i, frequency_j, "len, d -> len d")    # (max_seq_len, d_k//2)
        self.register_buffer("cosine", torch.cos(angle), persistent=False)    # 创建缓冲区，cos表
        self.register_buffer("sine", torch.sin(angle), persistent=False)    # 创建缓冲区，sin表

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cosine[token_positions]     # (..., seq_len, d_k//2)
        sin = self.sine[token_positions]      # (..., seq_len, d_k//2)
        x_even = x[..., 0::2]    # 取出偶数维度    (..., seq_len, d_k//2)
        x_odd = x[..., 1::2]     # 取出奇数维度    (..., seq_len, d_k//2)
        out_even = cos * x_even - sin * x_odd 
        out_odd = sin * x_even + cos * x_odd 
        # 在嵌入维度上拼接
        result = rearrange(torch.stack([out_even, out_odd], dim=-1), "... d two -> ... (d two)")   # 交错拼回
        # 也可以写：result = rearrange(torch.cat([out_even, out_odd], dim=-1), "... (two d) -> ... (d two)", two=2)
        return result.to(x.dtype)    # 转回x原本的dtype（maybe bf16）

# 对输入张量 x 的第 i 维施加softmax
def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    type = x.dtype
    x = x.to(torch.float32)  # 先转成float32
    # torch.max会返回（values，indices）
    # maximal = torch.max(x, dim=i, keepdim=True).values
    # 也可以：
    maximal = torch.amax(x, dim=i, keepdim=True).detach()   # 从计算图中摘掉，减小计算量
    # 减掉的最大值只是为了防止 exp 溢出，不参与求导
    modified = x - maximal
    exponent = torch.exp(modified)
    return (exponent / torch.sum(exponent, dim=i, keepdim=True)).to(type)

         
# 缩放点积注意力
def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = Q.shape[-1]   # 嵌入维度
    # 计算attention scores (..., len_queries, len_keys)
    scores = einsum(Q, K, "... len_queries d_k, ... len_keys d_k -> ... len_queries len_keys") / (d_k**0.5)
    masked_scores = scores.masked_fill(~mask, -torch.inf) if mask is not None else scores
    weights = softmax(masked_scores, i=-1)
    return einsum(weights, V, "... len_queries len_keys, ... len_keys d_v -> ... len_queries d_v")


# 因果多头自注意力（包含RoPE）
class MultiHeadSelfAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 max_seq_len: int | None=None,
                 theta: float | None=None, 
                 device: torch.device | None=None,
                 dtype: torch.dtype | None=None,
                 attn_gate: str = "none",           # 注意力门控机制
                 ):
        super().__init__()
        self.d_model = d_model 
        self.num_heads = num_heads
        # self.d_head = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        if (max_seq_len is not None) and (theta is not None):
            self.rope = RotaryPositionalEmbedding(theta, d_model//num_heads, max_seq_len, device)
        else:
            self.rope = None 
        # Gated Attention（Qiu et al., 2025）：在 attention 输出之后、output_proj 之前乘一个 sigmoid 门控
        #   "none"：不加门控（原始模型）
        #   "elementwise"：每个头的每一维都有自己的门控值，gate_proj: d_model -> d_model（论文效果最好的形式）
        #   "headwise"：每个头只有一个门控值，gate_proj: d_model -> num_heads（参数少很多）
        if attn_gate not in ("none", "elementwise", "headwise"):
            raise ValueError(f"attn_gate 只能是 none / elementwise / headwise，收到 {attn_gate}")
        self.attn_gate = attn_gate
        if attn_gate == "elementwise":
            self.gate_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        elif attn_gate == "headwise":
            self.gate_proj = Linear(d_model, num_heads, device=device, dtype=dtype)
        

    # 不带 KV Cache 的 forward 函数
    """def forward(self, x: torch.Tensor,
                token_positions: torch.Tensor | None=None):
        seq_len = x.shape[-2]
        # x: [..., len, d_model]    Q,K,V: [..., len, d_model]
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # Q,K,V: [num_heads, ..., len, d_head]
        Q = rearrange(Q, "... (num_heads d_head) -> num_heads ... d_head", num_heads = self.num_heads)
        K = rearrange(K, "... (num_heads d_head) -> num_heads ... d_head", num_heads = self.num_heads)
        V = rearrange(V, "... (num_heads d_head) -> num_heads ... d_head", num_heads = self.num_heads)
        if self.rope is not None:
            if token_positions is None:    # 如果没有传 token_positions
                # 也可以简写，因为 cos * x会自动广播：
                # token_positions = torch.arange(seq_len, device=x.device)
                token_positions = torch.arange(seq_len)[None, :][torch.zeros(Q.shape[:-2], dtype=torch.int32)].to(x.device)
            Q, K = self.rope(Q, token_positions), self.rope(K, token_positions)

        helper = torch.arange(seq_len)
        mask = helper <= helper[:, None]    # 构建因果掩码
        mask = mask.to(x.device)     # 将掩码移到 x.device
        # 做attention运算，并讲 head 维度拼回来
        mid = rearrange(scaled_dot_product_attention(Q, K, V, mask), "num_heads ... d_head -> ... (num_heads d_head)")
        output = self.output_proj(mid)   # 最后的线性层
        return output """

    # 带 KV Cache 的 forward 函数
    def forward(self, x: torch.Tensor,
                token_positions: torch.Tensor | None=None,
                kv_cache: tuple[torch.Tensor, torch.Tensor] | None=None,
                use_cache: bool = False):
        """
        x: (..., seq_len, d_model)
        kv_cache: 之前所有位置的 (K, V)，形状都是 (num_heads, ..., past_len, d_head)，其中 K 已经加过了 RoPE
        use_cache：True 时返回 (output, (K, V))；False 时只返回 output
        """
        seq_len = x.shape[-2]
        # 缓存里已经有多少个位置 = 这次输入的第一个 token 的绝对位置
        past_len = kv_cache[0].shape[-2] if kv_cache is not None else 0

        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # Q,K,V: [num_heads, ..., len, d_head]
        Q = rearrange(Q, "... (num_heads d_head) -> num_heads ... d_head", num_heads = self.num_heads)
        K = rearrange(K, "... (num_heads d_head) -> num_heads ... d_head", num_heads = self.num_heads)
        V = rearrange(V, "... (num_heads d_head) -> num_heads ... d_head", num_heads = self.num_heads)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(past_len, past_len + seq_len, device=x.device)
            # 先加 RoPE 再进缓存
            Q, K = self.rope(Q, token_positions), self.rope(K, token_positions)

        if kv_cache is not None:
            # 在序列那一维把旧的 K/V 和新的拼起来
            K = torch.cat([kv_cache[0], K], dim=-2)
            V = torch.cat([kv_cache[1], V], dim=-2)

        # 因果掩码：query 的绝对位置是 past_len...past_len+seq_len-1，key 的位置是 0 ... past_len+seq_len-1
        q_pos = torch.arange(past_len, past_len + seq_len, device=x.device)
        k_pos = torch.arange(0, past_len + seq_len, device=x.device)
        mask = k_pos[None, :] <= q_pos[:, None]

        mid = rearrange(scaled_dot_product_attention(Q, K, V, mask), "num_heads ... d_head -> ... (num_heads d_head)")

        if self.attn_gate != "none":
            # 门控值由这一层 attention 的输入 x 算出，sigmoid 后在 0~1 之间
            gate = torch.sigmoid(self.gate_proj(x))
            # elementwise: (..., seq_len, d_model);   headwise: (..., seq_len, num_heads)

            if self.attn_gate == "elementwise":
                mid = mid * gate 
            else:
                # 每个头一个门控值 (headwise)
                mid = rearrange(mid, "... (num_heads d_head) -> ... num_heads d_head", num_heads=self.num_heads)
                mid = rearrange(mid * gate[..., None], "... num_heads d_head -> ... (num_heads d_head)")

        output = self.output_proj(mid)
        if use_cache:
            return output, (K, V)
        return output


# 完整的 Transformer block
class TransformerBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int, 
                 d_ff: int | None=None,
                 max_seq_len: int | None=None, 
                 theta: float | None=None,
                 device: torch.device | None=None, 
                 dtype: torch.dtype | None=None,
                 attn_gate: str = "none",
                 ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype, attn_gate=attn_gate)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None=None,
                kv_cache: tuple[torch.Tensor, torch.Tensor] | None=None,
                use_cache: bool = False
    ):
        if use_cache:
            attn_out, new_cache = self.attn(self.ln1(x), token_positions, kv_cache=kv_cache, use_cache=True)
        else:
            attn_out = self.attn(self.ln1(x), token_positions)
        x = x + attn_out
        output = x + self.ffn(self.ln2(x))
        if use_cache:
            return output, new_cache
        return output



# Transformer 语言模型
class TransformerLM(nn.Module):
    def __init__(self, 
                 vocab_size: int,       # 词表大小，用于确定 token embedding矩阵的维度
                 context_length: int,   # 最大上下文长度，用于确定 RoPE 的 sin/cos buffer的维度
                 num_layers: int,      # transformer block块的数量
                 d_model: int,
                 num_heads: int,
                 d_ff: int | None=None,
                 rope_theta: float | None=None,
                 device: torch.device | None=None,
                 dtype: torch.dtype | None=None,
                 attn_gate: str = "none", 
                 ):
        super().__init__() 
        self.context_length = context_length 
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model,
                             num_heads,
                             d_ff,
                             max_seq_len=context_length,
                             theta=rope_theta,
                             device=device,
                             dtype=dtype,
                             attn_gate=attn_gate,
                             )
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor,
                token_positions: torch.Tensor | None=None,
                kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
                use_cache: bool = False,
                ) -> torch.Tensor:
        # kv_caches: 长度为 num_layers 的列表，第 i 个是第 i 层的 (K, V)；第一次调用传 None
        # use_cache: True 时返回 (logits, 新的缓存列表); False 时只返回 logits（和原来一样）
        past_len = kv_caches[0][0].shape[-2] if kv_caches is not None else 0
        total_len = past_len + token_ids.shape[-1]
        if total_len > self.context_length:
            raise ValueError(f"KV cache 总长度 {total_len} 超过了 context_length {self.context_length}")

        # token_ids: (batch, seq_len)  ->  x: (batch, seq_len, d_model)
        x = self.token_embeddings(token_ids)
        new_caches = []
        for i, layer in enumerate(self.layers):
            if use_cache:
                layer_cache = kv_caches[i] if kv_caches is not None else None 
                x, cache = layer(x, token_positions, kv_cache=layer_cache, use_cache=True)
                new_caches.append(cache)
            else:
                x = layer(x, token_positions)
        x = self.ln_final(x)
        logits = self.lm_head(x)    # (batch, seq_len, vocab_size)
        if use_cache:
            return logits, new_caches
        return logits



