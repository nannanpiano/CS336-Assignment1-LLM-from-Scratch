import json 
import regex as re 
from collections.abc import Iterable, Iterator

from pathlib import Path 

from cs336_basics.train_bpe_tinystories import *

# 项目根目录：本文件在 cs336_basics/ 下，往上一级就是根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent 

# GPT-2用的正则表达式
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""   )

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab     # 词表
        self.merges = merges    # 合并列表
        self.special_tokens=special_tokens   
        self.word_to_token = {}    # 缓存，建立word: str -> token_ids: list[int]的映射，处理过的word直接转换成token
        # 将新的 special_tokens加入到词表最后
        current_idx = len(vocab)
        if special_tokens:
            for special_token in self.special_tokens:
                if special_token.encode("utf-8") not in vocab.values():
                    vocab[current_idx] = special_token.encode("utf-8")
                    current_idx += 1    
        # 构建从 bytes到 idx的映射
        self.vocab_reverse = {bytes:idx for idx, bytes in self.vocab.items()}

    @classmethod   # 表示类方法，cls相当于self
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        vocab, merges = load_vocab_merges(vocab_filepath, merges_filepath)
        return cls(vocab, merges, special_tokens)

    def split_on_special_tokens(self, text: str, special_tokens: list[str]) -> list[str]:
        # 如果没有特殊字符
        if not special_tokens:
            return [text]
        # 按长度从长到短排序：重叠时优先匹配更长的 token
        tokens = sorted(special_tokens, key=len, reverse=True)
        # 每个 token 先转义，在用 "|"（即正则里的“或”）连起来
        # pattern = "|".join(re.escape(token) for token in tokens)
        pattern = "("+ "|".join(re.escape(token) for token in tokens) + ")"
        return re.split(pattern, text)   # 按特殊字符切分

    # 将一串字符编码成 token
    def encode(self, text: str) -> list[int]:
        token_list = []
        # 按特殊字符切分成块
        chunks = self.split_on_special_tokens(text, self.special_tokens)
        for chunk in chunks:
            if self.special_tokens and (chunk in self.special_tokens):
                token_list.append(self.vocab_reverse[chunk.encode("utf-8")])
            else:
                for m in PAT.finditer(chunk):
                    word = m.group() 
                    # 如果这个词之前处理过，直接映射成token序列
                    if word in self.word_to_token:
                        word_ids = self.word_to_token[word]
                        token_list += word_ids
                        continue
                    word_bytes = word.encode("utf-8")
                    start_bytes = [word_bytes[i:i+1] for i in range(len(word_bytes))]
                    for left, right in self.merges:
                        if (left + right) not in word_bytes:
                            continue 
                        else:
                            i = 0
                            while i < len(start_bytes) - 1:
                                if start_bytes[i] == left and start_bytes[i+1] == right:
                                    start_bytes = start_bytes[:i] + [left + right] + start_bytes[i+2:]
                                i += 1
                    word_ids = [self.vocab_reverse[token] for token in start_bytes]
                    self.word_to_token[word] = word_ids
                    token_list += word_ids
        return token_list

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for string in iterable:
            for token in self.encode(string):
                yield token

    def decode(self, ids: list[int]) -> str:
        return b''.join(self.vocab[id] for id in ids).decode("utf-8", errors="replace")


if __name__ == "__main__":
    output_dir = PROJECT_ROOT / "data" / "tokenizers"
    vocab_path = output_dir / "tinystories_vocab.json"
    merges_path = output_dir / "tinystories_merges.txt"

    tokenizer_test = Tokenizer.from_files(str(vocab_path), str(merges_path))

    example_dir = PROJECT_ROOT / "tests" / "fixtures" / "tinystories_sample.txt"

    with open(example_dir, 'r', encoding="utf-8") as f:
        text = f.read()
        tokens_full = tokenizer_test.encode(text)

    with open(example_dir, 'r', encoding="utf-8") as f:
        generator = tokenizer_test.encode_iterable(f)
        tokens_iter = list(generator)

    print(tokens_full == tokens_iter)


