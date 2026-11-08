# OpenWebText baseline（L40，代码版本 c975d2a）

设置：模型结构、训练量、优化器和 TinyStories baseline 完全一样（4 层、d_model 512、16 头、d_ff 1344、context 256，
batch 32 × 40000 步 = 327,680,000 token，AdamW，warmup 200，余弦衰减到 lr/10，bf16 + compile），
只把数据换成 OpenWebText，词表换成在 OWT 上训练的 32,000（参数量从 2270 万变成约 5000 万，多出来的都在 embedding 和 lm_head）。
脚本：`DATASET=owt sh scripts/lr_sweep.sh <lr>`；每个实验约 38 分钟，吞吐约 15.5 万 tok/s（TinyStories 是 29 万，因为输出层大了 3.2 倍）。

## 学习率

| 最大学习率 | 完整验证集 val loss（6640 万 token） | bits/byte | 训练中抽样 val（第 40000 步） |
|---|---|---|---|
| 6e-4 | 4.1823 | 1.382 | 4.1940 |
| 1e-3 | 4.1443 | 1.369 | 4.1554 |
| **2e-3** | **4.1306** | **1.365** | 4.1421 |

- 在 TinyStories 上 1e-3 和 2e-3 几乎一样（1.3939 / 1.3952）；在 OWT 上 2e-3 最好，领先 1e-3 约 0.014。
- 曲线有交叉：训练中段 2e-3 反而最差，学习率衰减下来之后才反超。只看中途的 loss 选学习率会选错。
- 训练中抽样的 val 这次比完整验证集偏高约 0.011，而 TinyStories 上是偏低约 0.009。偏差没有固定方向，就是抽样误差，报告以完整验证集为准。

## 和 TinyStories 对比

| | TinyStories（lr=1e-3） | OpenWebText（lr=2e-3） |
|---|---|---|
| 完整验证集 val loss（每 token） | 1.3939 | 4.1306 |
| 困惑度 exp(loss) | 4.0 | 62.2 |
| 每个 token 平均对应的字节数 | 4.12 | 4.37 |
| **bits/byte** | **0.488** | **1.365** |

怎么理解：
- 两个数据集的 loss 不能直接比“每 token”的数字，因为词表不同，一个 token 对应的文本长度也不同。换算成 bits/byte
  （每个字节的文本平均要用多少比特来编码：loss / ln2 / 每 token 字节数）之后可以比较：OWT 是 TinyStories 的约 2.8 倍。
- 差距来自数据本身：TinyStories 是用简单词汇、固定套路写成的儿童故事，可预测性很高；OWT 是真实网页文本，主题、文体、
  词汇、专有名词都极其多样。同样 5000 万参数、3.28 亿 token 的预算，在 TinyStories 上已经接近学会这种分布，在 OWT 上还差得很远。
- 困惑度 62 大致相当于：模型每一步平均在约 62 个候选词之间犹豫；TinyStories 上只有约 4 个。

## 生成样例（lr=2e-3，`decoding.py --kv_cache`，temperature 0.7，top-p 0.9，prompt "The city council voted on Tuesday to"）

> The city council voted on Tuesday to amend the proposed Bill of Rights to ensure that the city does not enter into a proposal to make it happen.
>
> The city council voted to increase the number of people in the area, but would consider its city to ban the use of a central city of Serena to address the issue.
>
> The council voted to approve the Bill of Rights in Serena, which would ban the use of force in public-sector matters.
>
> The city council voted to ban the use of force in the area, but it voted to ban the use of force in Serena.
>
> The council voted to legalize the use of force in the area.
>
> The council voted to ban the use of force in the area.

- 学会了新闻报道的表面形式：语法基本正确，有段落和引语，“The council voted to …” 这种句式很像新闻。
- 但内容空洞、前后矛盾（一会儿 ban 一会儿 legalize），而且大量重复同一个句式。和 TinyStories 能写出完整小故事相比明显差一截。
- 原因和 loss 高一样：模型容量和训练量相对 OWT 的多样性远远不够，只学到了高频的表面模式；遇到不确定的地方就退回概率最高、最“安全”的句式，于是出现重复。

## 工程上的一个坑：磁盘随机读

刚开始训练时只有约 4 万 tok/s、GPU 利用率 23%：OWT 训练集有 5.4 GB，用 mmap 每个 batch 从 32 个随机位置读取，共享磁盘的随机读跟不上
（训练进程处于 D 状态，IO pressure 约 51%）。先把文件顺序读一遍（`cat data/tokenized/owt_*.npy > /dev/null`，约 39 秒）让它进入页缓存后，
GPU 利用率恢复到 98%、约 15.7 万 tok/s。lr=6e-4 是在缓存预热之后才开始的，所以它的墙钟时间曲线比另外两条起得早。

图：`figures/owt_lr_sweep.png`（OWT 三个学习率）、`figures/ts_vs_owt.png`（同样模型和训练量下 TinyStories 与 OWT 的对比）。
