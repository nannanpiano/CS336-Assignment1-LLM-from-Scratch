import argparse 
import json 
import os 
import torch 
from cs336_basics.modules import TransformerLM, softmax 
from cs336_basics.tokenizer import Tokenizer

# 根据最后一个位置的 logits (V, ) 选出下一个 token
def sample_next(logits: torch.Tensor,
                temperature: float,
                top_p: float) -> int:
    # 贪心解码，直接取概率最大的 token
    if temperature == 0:
        return int(torch.argmax(logits))
    # 温度缩放后转成概率
    probs = softmax(logits.float() / temperature, -1)
    # 按概率从大到小排序，保留累计概率刚好到达 top_p 的最小集合
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)
    # 用“加上自己之前”的累计和判断：之前还没达到 top_p，自己就要保留（所以第一个 token 一定保留）
    remove = (cum - sorted_probs) >= top_p
    sorted_probs[remove] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum()   # 概率归一化
    # 在排序后的分布里采样，再通过 sorted_idx 映射回真正的 token_id
    k = torch.multinomial(sorted_probs, num_samples=1)
    return int(sorted_idx[k])

# 从 prompt 开始自回归生成，返回新生成的 token_id，直到遇到 eot
@torch.no_grad()
def decode(
    model: TransformerLM,      # languange model
    prompt_ids: list[int],      # prompt 编码成的 token id 序列
    max_new_tokens: int,       # 用户指定的最大生成 token 数
    context_length: int,        # 最大上下文
    temperature: float = 1.0,    # 温度参数
    top_p: float = 1.0,       # top_p
    eot_id: int | None = None,      # endoftext 的 token_id
    device: str = "cpu", 
    use_kv_cache: bool = False,     # True：用 KV cache，每步只算新 token；False：每步把整段序列重算一遍
) -> list[int]:
    model.eval()
    ids = list(prompt_ids)     # 为了不改变原始prompt_ids
    new_ids = []
    caches = None        # 每一层的 (K, V)，只在 use_kv_cache=True 时使用
    for _ in range(max_new_tokens):
        if not use_kv_cache:
            # 不用缓存：每一步都把最后 context_length 个 token 整段送进模型
            x = torch.tensor(ids[-context_length:], dtype=torch.long, device=device)[None, :]   # (1, T)
            logits = model(x)[0, -1]    # 只取最后一个位置的 logits   (V,)
        elif caches is None or len(ids) > context_length:
            # prefill: 第一步把 prompt 整段送进去，拿到每一层的缓存
            # 序列超过 context_length 后，窗口每步都要往右滑，所有 token 的位置编号都变了，缓存失效
            # 只能像不用 cache 时一样整段重算
            x = torch.tensor(ids[-context_length:], dtype=torch.long, device=device)[None, :]   # (1, T)
            out, caches = model(x, use_cache=True)
            logits = out[0, -1]
        else:
            # decode：只把最新的那 1 个 token 送进去，旧位置的 K/V 都在 cache 里
            x = torch.tensor([ids[-1]], dtype=torch.long, device=device)[None, :]   # (1, 1)
            out, caches = model(x, kv_caches=caches, use_cache=True)
            logits = out[0, -1]

        next_id = sample_next(logits, temperature, top_p)
        # 追加，遇到结束符就停下来
        ids.append(next_id)
        new_ids.append(next_id)
        if eot_id is not None and next_id == eot_id:
            break
    return new_ids

    

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="用训练好的 TransformerLM 生成文本")
    parser.add_argument("--run_dir", type=str, default="runs/ts_mps")       # 训练输出目录
    parser.add_argument("--vocab", type=str, default="data/tokenizers/tinystories_vocab.json")
    parser.add_argument("--merges", type=str, default="data/tokenizers/tinystories_merges.txt")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_samples", type=int, default=1)    # 用同一个prompt生成多少条
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--kv_cache", action="store_true", help="用 KV cache 加速生成（结果和不用时一样）")
    return parser.parse_args()


def main():
    args = parse_args()

    device = args.device if args.device is not None else get_device() 
    torch.manual_seed(args.seed)

    # 读训练时保存的超参
    with open(os.path.join(args.run_dir, "config.json")) as f:
        cfg = json.load(f)

    model = TransformerLM(
        vocab_size=cfg["vocab_size"],
        context_length=cfg["context_length"],
        num_layers=cfg["num_layers"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        rope_theta=cfg["rope_theta"],
        device=device,
        attn_gate=cfg.get("attn_gate", "none"),
    )
    ckpt = torch.load(os.path.join(args.run_dir, "ckpt.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])

    tokenizer = Tokenizer.from_files(args.vocab, args.merges, special_tokens=["<|endoftext|>"])
    eot_id = tokenizer.vocab_reverse["<|endoftext|>".encode("utf-8")]
    prompt_ids = tokenizer.encode(args.prompt)

    for i in range(args.num_samples):
        new_ids = decode(model, prompt_ids, args.max_new_tokens, cfg["context_length"],
                         args.temperature, args.top_p, eot_id, device, use_kv_cache=args.kv_cache)
        print(f"===== sample {i} ({len(new_ids)} tokens) =====")
        print(args.prompt + tokenizer.decode(new_ids))



if __name__ == "__main__":
    main()