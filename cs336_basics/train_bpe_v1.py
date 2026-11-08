import regex as re
import os 
from multiprocessing import Pool
from typing import BinaryIO

# 用来找到若干个切分点，每个切分点都落在特殊 token 的开头
from cs336_basics.pretokenization_example import find_chunk_boundaries  # 从 pretokenization_example.py导入

# GPT-2用的正则表达式
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""   )

# 小于这个大小的文件不开多进程（开进程本身要花时间，小文件反而更慢）
SMALL_FILE_BYTES = 1 << 20  # 1MB



def split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    # 如果没有特殊字符
    if not special_tokens:
        return [text]
    # 按长度从长到短排序：重叠时优先匹配更长的 token
    tokens = sorted(special_tokens, key=len, reverse=True)
    # 每个 token 先转义，在用 "|"（即正则里的“或”）连起来
    pattern = "|".join(re.escape(token) for token in tokens)
    # pattern = "("+ "|".join(re.escape(token) for token in tokens) + ")"
    return re.split(pattern, text)   # 按特殊字符切分

# 初始化词表，0～255对应字节 0～255，256～xxx 对应特殊字符
# 返回初始化后的词表以及新的下标
def vocab_init(special_tokens: list[str]):
    vocab_dict = {i:bytes([i]) for i in range(256)}
    current_index = 256

    for special_token in special_tokens:
        vocab_dict[current_index] = special_token.encode("utf-8")
        current_index += 1

    return vocab_dict, current_index

# 预分词器：只统计字符串形式的词频
# 把转成 tuple[bytes, ...] 留到最后，只对去重后的词作一次
def pre_tokenizer(piece: str, word_freq: dict[str, int]):
    for m in PAT.finditer(piece):
        word = m.group() 
        word_freq[word] = word_freq.get(word, 0) + 1

# 一个子进程的任务：只读文件的 [start, end) 这一段，按特殊字符切分后预分词，返回这一段的词频
def count_chunk(input_path: str, start: int, end: int, special_tokens: list[str]) -> dict[str, int]:
    with open(input_path, "rb") as f:  # 二进制模式打开
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    word_freq = {} 
    for piece in split_on_special_tokens(chunk, special_tokens):
        pre_tokenizer(piece, word_freq)
    return word_freq

# 并行预分词：把文件切成若干块，每块交给一个进程，最后把各块的词频加起来
def parallel_pre_tokenize(input_path: str, special_tokens: list[str], num_processes: int) -> dict[str, int]:
    with open(input_path, "rb") as f:    # 二进制模式打开
        if special_tokens:
            boundaries = find_chunk_boundaries(f, num_processes, special_tokens[0].encode("utf-8"))
        else:
            # 没有特殊字符就不切，整个文件当一块
            f.seek(0, os.SEEK_END)   # 跳到文件末尾
            boundaries = [0, f.tell()]

    tasks = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]

    if num_processes == 1 or len(tasks) == 1:
        results = [count_chunk(*task) for task in tasks]
    else:
        with Pool(num_processes) as pool:     # Pool启动 num_processes个子进程
            # 将tasks里的每个元组分配给对应的子进程，每个子进程执行 count_chunk(*task)
            results = pool.starmap(count_chunk, tasks)   # 把所有返回值组成一个列表

    # 汇总各块的词频
    word_freq = {}
    for part in results:
        for word, freq in part.items():
            word_freq[word] = word_freq.get(word, 0) + freq
    return word_freq 

# 统计所有字节对的初始频率，同时记录每个字节对出现在哪些词里（存词的下标）
def init_pair_stats(words: list[list[bytes]], counts: list[int]):
    pair_freq = {}
    pair_to_words = {}
    for idx, word in enumerate(words):
        for pair in zip(word, word[1:]):
            pair_freq[pair] = pair_freq.get(pair, 0) + counts[idx]
            pair_to_words.setdefault(pair, set()).add(idx)
    return pair_freq, pair_to_words 

# 在一个词里把所有相邻的 (left, right) 合并成一个 token，返回新的 token 列表
def merge_word(word: list[bytes], left: bytes, right: bytes) -> list[bytes]:
    new_word = [] 
    j = 0 
    while j < len(word):
        # 出现了该bp，合并
        if j < len(word) - 1 and word[j] == left and word[j + 1] == right:
            new_word.append(left + right) 
            j += 2
        else:
            new_word.append(word[j])
            j += 1
    return new_word 

# 执行一次合并：只处理包含 best_pair 的词，并且只更新这些词涉及的字节对频率（增量更新）
def apply_merge(best_pair, words, counts, pair_freq, pair_to_words):
    left, right = best_pair 
    for idx in pair_to_words.pop(best_pair):
        old_word = words[idx] 
        new_word = merge_word(old_word, left, right)
        freq = counts[idx]

        # 先把旧词里所有字节对的贡献减掉
        for pair in zip(old_word, old_word[1:]):
            pair_freq[pair] -= freq
            if pair_freq[pair] == 0:
                del pair_freq[pair]
            if pair in pair_to_words:
                pair_to_words[pair].discard(idx)

        # 再把新词里所有字节对的贡献加回来
        for pair in zip(new_word, new_word[1:]):
            pair_freq[pair] = pair_freq.get(pair, 0) + freq 
            pair_to_words.setdefault(pair, set()).add(idx)

        words[idx] = new_word 



def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], num_processes: int | None = None):
    vocab_dict, current_index = vocab_init(special_tokens)    # 初始词表和最新下标
    merge_list = []   # 初始化合并列表

    # 小文件单进程，大文件用上所有 cpu 核
    if num_processes is None:
        num_processes = 1 if os.path.getsize(input_path) < SMALL_FILE_BYTES else os.cpu_count() 

    word_freq = parallel_pre_tokenize(input_path, special_tokens, num_processes)    # 预分词，得到 {word: count}

    # 只对去重后的词转换成 token 列表：words[i] 是第 i 个词，counts[i] 是他出现的次数
    words = []   # [[bytes], ...]
    counts = []
    for word, freq in word_freq.items():
        word_bytes = word.encode("utf-8")
        words.append([word_bytes[i:i+1] for i in range(len(word_bytes))])
        counts.append(freq)

    pair_freq, pair_to_words = init_pair_stats(words, counts)    # 字节对频率只统计这一次

    # pair_freq 为空说明已经没有字节对可以合并了
    while current_index < vocab_size and pair_freq:
        most_freq_bp = max(pair_freq.items(), key=lambda x: (x[1], x[0]))[0]   # 得到频率最高的 bp

        left_bytes, right_bytes = most_freq_bp 
        vocab_dict[current_index] = left_bytes + right_bytes      # 将该 bp 合并后加入词表
        merge_list.append((left_bytes, right_bytes))     # 将该 bp 加入合并列表

        apply_merge(most_freq_bp, words, counts, pair_freq, pair_to_words)    # 合并并增量更新频率

        current_index += 1

        if len(merge_list) % 1000 == 0:
            print(f"已合并 {len(merge_list)} 次", flush=True)

    return vocab_dict, merge_list




    
