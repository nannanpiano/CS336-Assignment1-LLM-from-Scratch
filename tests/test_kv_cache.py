"""
KV cache 的正确性测试：带缓存逐个 token 计算的结果，必须和一次性把整段序列送进去的结果一致。

官方测试只覆盖“不用缓存”的路径，所以 TransformerBlock 漏写 `return output, new_cache` 时，
官方测试照样全部通过。这里专门测缓存路径，防止以后改模型时把它悄悄弄坏。

运行：uv run pytest tests/test_kv_cache.py -q
"""

import pytest
import torch

from cs336_basics.decoding import decode
from cs336_basics.modules import MultiHeadSelfAttention, TransformerLM


def make_lm(context_length=32):
    """一个很小的随机初始化模型：测的是“缓存 vs 不缓存是否一致”，和模型训没训好无关。"""
    torch.manual_seed(0)
    return TransformerLM(vocab_size=300, context_length=context_length, num_layers=3, d_model=64,
                         num_heads=4, d_ff=128, rope_theta=10000.0)


def test_attention_incremental_matches_full():
    """单个 attention：先送前 6 个 token 拿缓存，再逐个送剩下的，每个位置的输出应和一次全送一致。"""
    torch.manual_seed(0)
    attn = MultiHeadSelfAttention(64, 4, max_seq_len=32, theta=10000.0)
    x = torch.randn(2, 10, 64)
    full = attn(x)

    out, cache = attn(x[:, :6], use_cache=True)
    outs = [out]
    for t in range(6, 10):
        o, cache = attn(x[:, t:t + 1], kv_cache=cache, use_cache=True)
        outs.append(o)

    assert cache[0].shape == (4, 2, 10, 16)   # (num_heads, batch, 已缓存的位置数, d_head)
    torch.testing.assert_close(torch.cat(outs, dim=1), full, atol=1e-5, rtol=1e-5)


def test_lm_incremental_matches_full():
    """整个模型：prefill 前 12 个 token，再逐个送第 13~20 个，每个位置的 logits 应和一次全送一致。"""
    lm = make_lm()
    ids = torch.randint(0, 300, (2, 20))
    full = lm(ids)

    logits, caches = lm(ids[:, :12], use_cache=True)
    outs = [logits]
    for t in range(12, 20):
        lg, caches = lm(ids[:, t:t + 1], kv_caches=caches, use_cache=True)
        outs.append(lg)

    assert len(caches) == 3                     # 每一层一份缓存
    torch.testing.assert_close(torch.cat(outs, dim=1), full, atol=1e-4, rtol=1e-4)


def test_default_call_unchanged():
    """不传缓存参数时，返回值仍然只是 logits 张量（官方测试和训练代码都依赖这一点）。"""
    lm = make_lm()
    out = lm(torch.randint(0, 300, (1, 5)))
    assert isinstance(out, torch.Tensor) and out.shape == (1, 5, 300)


def test_cache_overflow_raises():
    """缓存总长度超过 context_length 时，要给出清楚的报错，而不是 RoPE 查表越界。"""
    lm = make_lm(context_length=16)
    _, caches = lm(torch.randint(0, 300, (1, 12)), use_cache=True)
    with pytest.raises(ValueError):
        lm(torch.randint(0, 300, (1, 5)), kv_caches=caches, use_cache=True)


@pytest.mark.parametrize("max_new_tokens", [10, 40])   # 40 > context_length，覆盖“超长后整段重算”的路径
def test_greedy_decode_same_with_and_without_cache(max_new_tokens):
    """贪心解码：用不用 KV cache，生成的 token 序列必须一模一样。"""
    lm = make_lm(context_length=16)
    prompt = [5, 17, 42]
    a = decode(lm, prompt, max_new_tokens, context_length=16, temperature=0, use_kv_cache=False)
    b = decode(lm, prompt, max_new_tokens, context_length=16, temperature=0, use_kv_cache=True)
    assert len(a) == max_new_tokens
    assert a == b
