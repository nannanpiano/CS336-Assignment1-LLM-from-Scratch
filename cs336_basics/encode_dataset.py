import argparse 
import time 
from array import array 
from collections.abc import Iterator 
from pathlib import Path

import numpy as np 

from cs336_basics.tokenizer import Tokenizer 

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent 
DATA_DIR = PROJECT_ROOT / "data"
TOKENIZER_DIR = DATA_DIR / "tokenizers"
OUTPUT_DIR = DATA_DIR / "tokenized" 

# 每个数据集的原始文本文件：tokenizer 用同名的那一份
DATASETS = {
    "tinystories": {
        "train": DATA_DIR / "TinyStoriesV2-GPT4-train.txt",
        "valid": DATA_DIR / "TinyStoriesV2-GPT4-valid.txt"
    },
    "owt": {
        "train": DATA_DIR / "owt_train.txt",
        "valid": DATA_DIR / "owt_valid.txt",
    },    
}
SPECIAL_TOKENS = ["<|endoftext|>"]

# 加载tokenizer（用 Tokenizer.from_files）
def load_tokenizer(name: str) -> Tokenizer:
    return Tokenizer.from_files(
        str(TOKENIZER_DIR / f"{name}_vocab.json"),
        str(TOKENIZER_DIR / f"{name}_merges.txt"),
        special_tokens=SPECIAL_TOKENS,
    )

# 包一层文件迭代器：把每一行原样交出去，顺便每隔一段打印一次进度
def lines_with_progress(f, total_bytes: int, report_every: int = 100_000_000) -> Iterator[str]:
    seen = 0 
    next_report = report_every 
    start = time.time() 
    for line in f:
        seen += len(line)    # 用字符数近似字节数
        if seen >= next_report:
            print(f"  已处理约 {seen / 1e6:.0f}MB / {total_bytes / 1e6:.0f}MB，用时 {time.time() - start:.0f}s", flush=True)
            next_report += report_every
        yield line

# 把一个文本文件编码成 token id，存成 uint16 的 .npy 文件
def encode_file(tokenizer: Tokenizer, input_path: Path, output_path: Path) -> np.ndarray:
    # uint16 只能表示 0~65535，词表必须装得下
    assert len(tokenizer.vocab) <= 2**16, "词表超过 65536，uint16 装不下"

    total_bytes = input_path.stat().st_size
    ids = array("H")   # "H" 是 2 字节的无符号整数，比 Python 列表省十几倍内存
    with open(input_path, encoding="utf-8") as f:
        # encode_iterable 逐行编码、逐个产出 id，array 边收边存，全程不需要把整个文件读进内存
        ids.extend(tokenizer.encode_iterable(lines_with_progress(f, total_bytes)))

    arr = np.frombuffer(ids, dtype=np.uint16)   # 直接复用 array 的内存，不复制
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, arr)
    return arr

# 读回保存的文件做检查：类型、id 范围、开头一段能否解码回原文
def check_saved(tokenizer: Tokenizer, input_path: Path, output_path: Path):
    arr = np.load(output_path, mmap_mode="r")   # 内存映射：按需读取，不把整个文件读进内存
    assert arr.dtype == np.uint16, f"dtype 应该是 uint16，实际是 {arr.dtype}"
    assert int(arr.max()) < len(tokenizer.vocab), "出现了超出词表范围的 id"

    with open(input_path, encoding="utf-8") as f:
        head = f.read(200)
    decoded = tokenizer.decode(arr[:1000].tolist())
    assert decoded.startswith(head), "开头一段解码后和原文不一致"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="把文本数据集编码成 uint16 的 token id 数组")
    parser.add_argument("--dataset", choices=list(DATASETS), default="tinystories")
    parser.add_argument("--split", choices=["valid", "train"], default="valid")
    args = parser.parse_args()

    input_path = DATASETS[args.dataset][args.split]
    output_path = OUTPUT_DIR / f"{args.dataset}_{args.split}.npy"

    tokenizer = load_tokenizer(args.dataset)
    print(f"编码 {input_path.name} → {output_path}")

    start = time.time()
    arr = encode_file(tokenizer, input_path, output_path)
    elapsed = time.time() - start
    total_bytes = input_path.stat().st_size
    print(f"完成：{len(arr):,} 个 token，用时 {elapsed:.0f}s（{total_bytes / 1e6 / elapsed:.2f} MB/s），"
          f"平均 {total_bytes / len(arr):.2f} 字节/token")

    check_saved(tokenizer, input_path, output_path)
    print("检查通过：dtype、id 范围、开头解码都正确")