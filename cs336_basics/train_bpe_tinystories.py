import json 
import time 
from functools import lru_cache 
from pathlib import Path 

from cs336_basics.train_bpe_v1 import train_bpe 

# 项目根目录：本文件在 cs336_basics/ 下，往上一级就是根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent 

@lru_cache 
def gpt2_bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]

    n = 0 
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d 

# ------------ ⬆️来自Assignment/assignment1-basics/tests/common.py --------------

bytes_to_unicode = gpt2_bytes_to_unicode()      # bytes 到 unicode字符 映射表，如 32 --> 'G'
unicode_to_bytes = {ch: b for b, ch in bytes_to_unicode.items()}   # 反过来：unicode字符 到 bytes，如 'G' --> 32

# 一个 token (bytes) --> 可读字符串
def token_to_str(token: bytes) -> str:
    return "".join(bytes_to_unicode[b] for b in token)   

# 可读字符串 --> 一个 token（bytes）
def str_to_token(s: str) -> bytes:
    return bytes([unicode_to_bytes[ch] for ch in s])

# 把 vocab 和 merges 写到磁盘（和 tests/fixtures 里 gpt2_vocab.json / gpt2_merges.txt 同一种格式）
# vocab 文件：JSON，{可读字符串: id}
# merges 文件：每行一个 merge，两个可读字符串之间用一个空格隔开，按创建顺序
def save_vocab_merges(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], vocab_path, merges_path):
    Path(vocab_path).parent.mkdir(parents=True, exist_ok=True)

    vocab_json = {token_to_str(token): token_id for token_id, token in vocab.items()}
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)

    with open(merges_path, "w", encoding="utf-8") as f:
        for left, right in merges:
            f.write(f"{token_to_str(left)} {token_to_str(right)}\n")

# 从磁盘读回 vocab 和 merges，是 save_vocab_merges 的逆操作
def load_vocab_merges(vocab_path, merges_path) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(vocab_path, encoding="utf-8") as f:
        vocab_json = json.load(f) 
    # 文件里是 {字符串: id}，要还原成 {id: bytes}
    vocab = {token_id: str_to_token(s) for s, token_id in vocab_json.items()}

    merges = []
    with open(merges_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")   # 去掉行尾换行符
            if not line:     # 跳过空行
                continue
            left, right = line.split(" ")
            merges.append((str_to_token(left), str_to_token(right)))
    return vocab, merges

if __name__ == "__main__":
    input_path = PROJECT_ROOT / "data" / "TinyStoriesV2-GPT4-train.txt"
    special_tokens = ['<|endoftext|>']
    vocab_size = 10000

    output_dir = PROJECT_ROOT / "data" / "tokenizers"
    vocab_path = output_dir / "tinystories_vocab.json"
    merges_path = output_dir / "tinystories_merges.txt"

    # 训练
    start = time.time() 
    vocab_dict, merge_list = train_bpe(str(input_path), vocab_size, special_tokens)
    print(f"训练完成：词表 {len(vocab_dict)}项，{len(merge_list)} 次合并，用时 {time.time() - start:.1f}s")

    # 保存
    save_vocab_merges(vocab_dict, merge_list, vocab_path, merges_path) 
    print(f"已保存到 {vocab_path} 和 {merges_path}")

    # 往返验证：读出必须和写入一模一样
    loaded_vocab, loaded_merges = load_vocab_merges(vocab_path, merges_path)
    assert loaded_vocab == vocab_dict, "vocab 读回后不一致"
    assert loaded_merges == merge_list, "merges 读回后不一致"
    print("往返验证通过！✅")

    # 最长的 token（排除特殊 token）
    special_bytes = {token.encode("utf-8") for token in special_tokens}
    longest = max((token for token in vocab_dict.values() if token not in special_bytes), key=len)
    print(f"最长的 token：{longest} ({len(longest)} 字节)")