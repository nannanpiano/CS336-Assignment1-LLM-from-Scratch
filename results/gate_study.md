# Gated Attention 复现（TinyStories，小模型）

复现的是 Qiu et al., *Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free*（NeurIPS 2025，arXiv:2505.06708）
里效果最好的形式：在 attention 输出之后、`output_proj` 之前，乘一个由该层输入算出的逐元素 sigmoid 门控（`--attn_gate elementwise`）。

设置：和 TinyStories baseline 完全一样（4 层、d_model 512、16 头、d_ff 1344、context 256；batch 32 × 40000 步 = 3.28 亿 token；
AdamW，warmup 200，余弦衰减到 lr/10；bf16 + compile），只多了门控。门控每层多一个 512×512 矩阵，总参数 22.70M → 23.75M（+4.6%）。
代码版本 c3d3884 / 0297627 / c2f302b；实验在 L40 上运行（GPU3，headwise seed 2 在 GPU1）。

## 1. 主结果：val loss（完整验证集，546 万 token）

| | seed 0 | seed 1 | seed 2 | 均值 ± 标准差 |
|---|---|---|---|---|
| | seed 0 | seed 1 | seed 2 | 均值 ± 标准差 | 参数量 |
|---|---|---|---|---|---|
| baseline | 1.3939 | 1.3903 | 1.3926 | 1.3923 ± 0.0018 | 22.70M |
| headwise 门控 | 1.3817 | 1.3826 | 1.3857 | 1.3833 ± 0.0021 | 22.73M（+0.1%） |
| **elementwise 门控** | **1.3753** | **1.3770** | **1.3757** | **1.3760 ± 0.0009** | 23.75M（+4.6%） |

- elementwise 门控把 val loss 平均降低 **0.0163**，3 个种子全部更好，两组的范围完全不重叠（baseline 最好 1.3903，门控最差 1.3770）。
  差距约为种子间标准差的 9～18 倍（Welch t ≈ 14），不是随机波动。
- **参数量的质疑基本可以排除**：headwise 门控每层只多 512×16 个参数（总共 +0.1%），也把 val loss 降低了 **0.0090**
  （3 个种子同样全部更好、范围不重叠，Welch t ≈ 5.6），相当于 elementwise 收益的约 55%。所以提升主要来自门控机制本身，
  而不是多出来的参数。elementwise 比 headwise 再好 0.0073，可能来自更细粒度的门控，也可能部分来自多出的 4.6% 参数
  （论文中 elementwise 也略好于 headwise）。
- 代价：elementwise 训练速度慢约 6%（约 27.2 万 vs 28.6 万 tok/s，每次约 21 分钟 vs 20 分钟），headwise 慢约 3%。
- 图：`figures/gate_summary.png`（三组的最终 val 和各层最大激活）、`figures/gate_vs_baseline.png`
  （训练中抽样的 val 曲线噪声较大，曲线会交叉；最终结论以完整验证集为准）。

## 2. 训练稳定性：lr = 3e-3

| | 结果 |
|---|---|
| baseline | 约第 400 步发散（loss 2.77 → 6.76 → 8.7），第 12200 步变成 NaN |
| gated | **同样在约第 400 步发散**（loss 2.83 → 6.28），之后停在 6～7 一段时间，第 8350 步变成 NaN |

- 在这个规模上，**门控没有让模型在 3e-3 下稳定训练**。两者都在学习率爬到 3e-3 之后不久同时发散，门控版刚发散时 loss 稍低一些，但最终也变成了 NaN。
- 论文报告的稳定性收益是“减少 loss 尖峰”，出现在更深的模型（更多层）、更大的 batch 和学习率下。我们这里是学习率本身就超出了稳定范围
  （3e-3 对 batch 32 来说太大），门控解决不了这一类发散。
- 图：`figures/gate_lr3e-3.png`。

## 3. 机制分析：massive activation 和 attention sink

脚本 `cs336_basics/analyze_attention.py`，每个模型都用同一批 64 段验证集文本（每段 256 token）。

| | 各层残差流最大激活（层 1/2/3/4，3 个种子平均） | 平均 | 分给第一个 token 的注意力（各层平均） |
|---|---|---|---|
| baseline | 8.1 / 19.9 / 26.3 / 32.9 | 21.8 | 约 1.1% |
| headwise 门控 | 3.5 / 4.0 / 10.1 / 19.8 | **9.3** | 约 1.0% |
| elementwise 门控 | 3.8 / 5.4 / 6.9 / 20.1 | **9.1** | 约 0.9% |

- **massive activation：复现了。** 门控让残差流里的最大激活值降到 baseline 的约 40%，在前 3 层最明显（降到 1/4～1/3），3 个种子都一致。
  这和论文“门控引入的稀疏性能缓解 massive activation”的结论一致。
- **attention sink：在我们的模型里本来就不存在，所以无法检验。** baseline 分给第一个 token 的注意力只有约 1.1%，
  比“均匀分布”时的期望（约 2.4%）还低，完全没有 sink。论文里的 sink 从 baseline 的第 6 层才开始明显出现（附录 A.3），
  而我们的模型只有 4 层；另外我们的训练窗口是从文本中随机截取的，窗口的第一个 token 通常不是文档开头。
- headwise 门控对最大激活的抑制和 elementwise 几乎一样（平均 9.3 vs 9.1），但 val loss 的收益只有一半多：
  抑制离群激活值和降低 loss 并不是一回事。
- 门控值：elementwise 各层平均 0.21～0.36，有 8%～23% 小于 0.1；headwise 各层平均 0.25～0.50，只有 0%～11% 小于 0.1。
  论文报告的门控更稀疏（大模型里平均约 0.1 左右），我们的小模型门控没那么稀疏，headwise 尤其不稀疏。
- 注：这里 F-Attn 的“均匀分布期望”是对每个 query 位置 t 取 1/t 再平均（t = 2..256），约 2.4%。
- 图：`figures/gate_activation_sink.png`。

## 4. 局限与可选的后续

- **elementwise 的参数没有严格对齐**：多了 4.6% 的参数（论文在稠密模型实验里会相应减小 FFN 宽度来对齐）。headwise 的实验（+0.1% 参数，
  也有约 55% 的收益）已经说明主要收益不来自参数量，但 elementwise 和 headwise 之间那 0.0073 的差距，还没有排除参数量的影响。
- 只在一个规模（4 层）和一个数据集（TinyStories）上做了实验；attention sink 可能需要更深的模型才会出现。
- 稳定性只试了一个学习率（3e-3）；在 2e-3 附近、或更深的模型上比较 loss 尖峰，可能更接近论文的设定。
