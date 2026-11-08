# Attention sink / massive activation 分析

每个模型都用同一批 64 段验证集文本（每段 256 个 token，随机位置截取）。
F-Attn = 位置 ≥1 的 query 分给窗口第一个 token 的平均注意力；M-Act = 每层输出残差流的最大绝对值。

| 模型 | 门控 | F-Attn 逐层 | F-Attn 平均 | M-Act 逐层 | M-Act 平均 | 门控均值 | 门控 <0.1 比例 |
|---|---|---|---|---|---|---|---|
| ts_gpu_lr1e-3 | none | 1.5% / 1.0% / 1.1% / 1.1% | 1.2% | 10 / 24 / 29 / 31 | 24 | — | — |
| ts_gpu_lr1e-3_seed1 | none | 1.4% / 1.1% / 1.1% / 1.1% | 1.2% | 8 / 28 / 31 / 34 | 25 | — | — |
| ts_gpu_lr1e-3_seed2 | none | 1.4% / 1.1% / 1.0% / 1.1% | 1.2% | 6 / 8 / 19 / 34 | 17 | — | — |
| ts_gpu_lr1e-3_gateelementwise_seed0 | elementwise | 1.1% / 0.8% / 1.0% / 0.9% | 1.0% | 2 / 4 / 6 / 20 | 8 | 0.288 | 13% |
| ts_gpu_lr1e-3_gateelementwise_seed1 | elementwise | 1.1% / 0.9% / 1.1% / 0.8% | 1.0% | 3 / 5 / 6 / 21 | 9 | 0.288 | 13% |
| ts_gpu_lr1e-3_gateelementwise_seed2 | elementwise | 1.1% / 0.9% / 1.0% / 0.8% | 1.0% | 7 / 7 / 9 / 20 | 11 | 0.289 | 13% |
| ts_gpu_lr1e-3_gateheadwise_seed0 | headwise | 1.2% / 1.0% / 1.0% / 1.0% | 1.1% | 4 / 5 / 10 / 16 | 9 | 0.365 | 5% |
| ts_gpu_lr1e-3_gateheadwise_seed1 | headwise | 1.3% / 0.9% / 1.0% / 0.9% | 1.0% | 3 / 4 / 6 / 21 | 9 | 0.365 | 4% |
| ts_gpu_lr1e-3_gateheadwise_seed2 | headwise | 1.2% / 0.9% / 0.9% / 0.9% | 1.0% | 3 / 4 / 14 / 22 | 11 | 0.364 | 4% |
