# stard_lab

英文文档是本项目的权威版本。本文档是中文翻译。

本仓库包含 STARD 法条检索实验的训练与评估配置，当前重点是 Qwen3 reranker / cross-encoder 训练。

## 项目目的

STARD 论文中的原始 retrieval pipeline 是不含 reranker 的 dense retrieval pipeline。本项目考查两个问题：在 STARD 法条检索任务中引入 reranker 会带来多大增益，以及在 STARD 训练 pair 上微调 reranker 还能带来多少增益。

## STARD

[STARD](https://arxiv.org/abs/2406.15313) 是一个中文法条检索基准，查询来自非专业人士提出的真实法律咨询问题。已发布数据集包含 1,543 个查询案例和 55,348 条候选法律条文，用于评估模型能否从缺少精确法律术语的日常法律问题中检索相关法条。

该论文后来发表于 Findings of EMNLP 2024，标题为 [STARD: A Chinese Statute Retrieval Dataset Derived from Real-life Queries by Non-professionals](https://aclanthology.org/2024.findings-emnlp.625/)。数据集和原始代码见 [STARD GitHub 仓库](https://github.com/oneal2000/STARD)，该仓库采用 MIT license。

## 内容

- `scripts/train_qwen3_reranker_lora.py`：在 STARD 上训练 Qwen3 Reranker LoRA。
- `scripts/cross_encoder_rerank_eval.py`：基于一阶段排序文件进行 CrossEncoder rerank 评估。
- `outputs/qwen3-embedding-4b-ms-bs16/rank.tsv`：Qwen3-Embedding-4B 一阶段候选，用作 hard negatives 和 dev rerank 候选。
- `outputs/qwen3-embedding-4b-ms-bs16/metrics.json`：一阶段检索指标。
- `results/`：实验配置、评估指标和代表性的 dev rerank 排序结果。
- `docs/data_sources.md`：STARD 数据与训练数据来源说明。
- `docs/train_qwen3_reranker_lora_design.md`：LoRA 训练程序详细设计说明。

## 数据

本仓库不内置原始 STARD 数据集。请将其克隆为同级目录：

```bash
git clone https://github.com/oneal2000/STARD ../STARD
```

当本地 `data/queries.json` 和 `data/corpus.jsonl` 不存在时，训练脚本会自动回退读取 `../STARD/data`。

需要的 STARD 文件：

- `../STARD/data/queries.json`
- `../STARD/data/corpus.jsonl`
- `../STARD/data/example/train.query.txt`
- `../STARD/data/example/dev.query.txt`

## 安装

请先安装与目标机器 CUDA 环境匹配的 PyTorch，然后安装其余依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Base 和 Fine-Tuned Reranker 评估

下表中除论文 baseline 外，其他实验都使用 STARD 原始仓库 split 文件中的同一组 train/dev 划分。带 reranker 的行均对 Qwen3-Embedding-4B 生成的 top-500 候选进行重排。

所有结果均为单 seed 单次运行，未做方差估计。

| Run | Embedder | 是否使用 reranker | Reranker | Recall@10 | MRR@10 |
| --- | --- | --- | --- | ---: | ---: |
| STARD paper p7 Fine-tuned PLM Dense-STARD baseline | Chinese-RoBERTa-WWM fine-tuned dense retriever | No | None | 0.6061 | 0.4724 |
| Qwen3-Embedding-4B first stage | Qwen3-Embedding-4B | No | None | 0.6036 | 0.5010 |
| Qwen3-Reranker-0.6B base CrossEncoder | Qwen3-Embedding-4B | Yes | Qwen3-Reranker-0.6B base | 0.6495 | 0.5524 |
| Qwen3-Reranker-4B base CrossEncoder | Qwen3-Embedding-4B | Yes | Qwen3-Reranker-4B base | 0.7377 | 0.6165 |
| Qwen3-Reranker-0.6B LoRA lr6e-5 hardpool200 rand10 | Qwen3-Embedding-4B | Yes | Qwen3-Reranker-0.6B LoRA, epoch 4 | 0.7779 | 0.6921 |
| Qwen3-Reranker-4B LoRA lr3e-5 hardpool200 rand10 | Qwen3-Embedding-4B | Yes | Qwen3-Reranker-4B LoRA, epoch 2 | **0.8482** | **0.7345** |

论文中的 Fine-tuned PLM Dense-STARD embedder 权重没有公开，因此本仓库采用通用 embedder `Qwen3-Embedding-4B` 作为一阶段替代模型。在不使用 reranker 的情况下，`Qwen3-Embedding-4B` 的 Recall@10 与论文 baseline 基本打平，MRR@10 更高，整体上可以作为后续 reranker 实验的合理一阶段基线。

在这些单次运行中，直接加入未微调的 Qwen3 reranker 已经提升了一阶段排序。0.6B base CrossEncoder 将 Recall@10 从 0.6036 提升到 0.6495，将 MRR@10 从 0.5010 提升到 0.5524。4B base CrossEncoder 的 rerank 增益更大，达到 0.7377 Recall@10 和 0.6165 MRR@10。

微调 reranker 带来了已记录实验中最大的增益。0.6B LoRA 达到 0.7779 Recall@10 和 0.6921 MRR@10；4B LoRA 达到 0.8482 Recall@10 和 0.7345 MRR@10。

## 复现信息

| 项目 | 值 |
| --- | --- |
| 随机种子 | 42 |
| Query split | 1,235 个 train queries，308 个 dev queries，与 STARD 原始仓库 split 文件一致 |
| Corpus size | 55,348 条候选法律条文 |
| 加载的 STARD query 总数 | 1,543 |
| 一阶段候选 | `Qwen3-Embedding-4B` top-500 |
| GPU 硬件 | 2 x NVIDIA GeForce RTX 4090，每张 24,564 MiB |
| 训练设备 | `cuda:0` 训练，`cuda:1` 异步 dev evaluation |
| 软件版本 | `torch==2.5.1`, `transformers==4.57.6`, `peft==0.20.0`, `accelerate==1.14.0`, `sentence-transformers==5.7.0`, `modelscope==1.39.1` |
一阶段 embedder 和 base CrossEncoder 行是本仓库中的 evaluation-only 实验。

## Reranker 训练样本构造

当前训练程序是 LoRA，不是 QLoRA。脚本使用 `AutoModelForCausalLM.from_pretrained(..., torch_dtype=...)` 加载基础模型，然后通过 PEFT `LoraConfig` 挂载 adapter。脚本没有使用 `bitsandbytes`、`load_in_4bit`、`load_in_8bit`，也没有任何量化加载基础模型的路径。

训练正样本来自 `queries.json` 中的 `match_id` 标签。负样本在程序运行时构造：

- `hard_negative_pool_k=200`：对每个 query 取一阶段 top-200 候选，移除 gold 文档后作为 hard negative pool。
- `hard_negatives=30`：从 hard negative pool 中采样 30 条。
- `random_negatives=10`：从全量 corpus 中额外采样 10 条非 gold 文档。

设计动机，不是消融结论：法律条文可能在词面和语义上非常接近，因此 hard negatives 旨在让 reranker 接触看起来合理但非 gold 的法条。random negatives 用于让训练分布中保留明显不相关的法条，也降低模型只适配某一个 dense retriever near-miss 分布的风险。主要风险是假负例：如果 STARD 标注没有覆盖某些实际相关法条，非 gold 候选不一定真负，因此需要依据 dev 指标选择 best checkpoint。

## Training Details of Qwen3-Reranker-0.6B

0.6B LoRA 实验使用 STARD 原始仓库中的 `train.query.txt` 和 `dev.query.txt` 划分。上表中的最佳 0.6B 实验配置如下：

- Base model: `Qwen/Qwen3-Reranker-0.6B`
- Training method: LoRA, not QLoRA
- LoRA config: `r=16`, `alpha=32`, `dropout=0.05`
- LoRA target modules: `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
- Max length: 1024
- Candidate top-k: 500
- Hard-negative pool: 一阶段 top-200 非 gold 候选
- Per-query negatives: 30 个 hard negatives 加 10 个 random negatives
- Train micro-batch size: 14
- Gradient accumulation steps: 2
- Effective batch size: 每个 optimizer step 28 个 pair
- Eval batch size: 12
- Random seed: 42
- 这些单 seed 实验中观察到的最佳 learning rate: `6e-5`

学习率影响：在这些单 seed 单次运行中，`6e-5` 取得了 0.6B MRR@10 的最高观察值。已记录的 `7e-5` hardpool 和 resampling 实验指标更低，但如果没有多 seed 重复实验，不应把它解读为通用学习率结论。

![Qwen3-Reranker-0.6B LoRA learning-rate comparison](docs/assets/qwen3-reranker-0.6b-lr-comparison.svg)

epoch 影响：在 `6e-5` 实验中，Recall@10 在 epoch 2 达到峰值，MRR@10 在 epoch 4 达到峰值。之后多数 epoch 的 dev 指标在这次运行中下降，提示模型可能开始过拟合训练 pair 和候选分布。因此应使用 dev 上的 best checkpoint，而不是最后一个 epoch。

![Qwen3-Reranker-0.6B LoRA epoch curve](docs/assets/qwen3-reranker-0.6b-epoch-curve.svg)

## Training Details of Qwen3-Reranker-4B

4B LoRA 实验同样使用 STARD 原始 train/dev split，并使用同一份 Qwen3-Embedding-4B top-500 一阶段候选。已记录实验配置如下：

- Base model: `Qwen/Qwen3-Reranker-4B`
- Training method: LoRA, not QLoRA
- LoRA config: `r=16`, `alpha=32`, `dropout=0.05`
- LoRA target modules: `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
- Max length: 1024
- Candidate top-k: 500
- Hard-negative pool: 一阶段 top-200 非 gold 候选
- Per-query negatives: 30 个 hard negatives 加 10 个 random negatives
- Train micro-batch size: 2
- Gradient accumulation steps: 14
- Effective batch size: 每个 optimizer step 28 个 pair
- Eval batch size: 4
- Learning rate: `3e-5`
- Epochs: 3
- Random seed: 42

4B 实验从 epoch 1 到 epoch 2 的 MRR@10 有提升，但 epoch 3 回落；同时训练 loss 从 0.6972 继续下降到 0.4657、0.2646。这个模式与早期过拟合一致，但它仍然只是一次 3-epoch 运行；本次记录实验选择 epoch 2 checkpoint。

![Qwen3-Reranker-4B LoRA epoch curve](docs/assets/qwen3-reranker-4b-epoch-curve.svg)

## 训练 Qwen3 Reranker LoRA

下面的命令对应已记录的 0.6B 训练实验：

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/train_qwen3_reranker_lora.py \
  --model-source modelscope \
  --rank-tsv outputs/qwen3-embedding-4b-ms-bs16/rank.tsv \
  --output-dir outputs/qwen3-reranker-0.6b-lora-stard-bs14-lr6e-5-ep10-hardpool200-rand10 \
  --candidate-top-k 500 \
  --hard-negative-pool-k 200 \
  --hard-negatives 30 \
  --random-negatives 10 \
  --epochs 10 \
  --train-batch-size 14 \
  --gradient-accumulation-steps 2 \
  --eval-batch-size 12 \
  --learning-rate 6e-5 \
  --max-length 1024 \
  --resample-hard-negatives-each-epoch \
  --resample-random-negatives-each-epoch
```

如需复现 4B LoRA 训练，请参考已记录的配置：

```text
results/outputs/qwen3-reranker-4b-lora-stard-bs2-ga14-lr3e-5-ep3-hardpool200-rand10/train_config.json
```

## 评估 CrossEncoder

```bash
python scripts/cross_encoder_rerank_eval.py \
  --model Qwen/Qwen3-Reranker-4B \
  --model-source modelscope \
  --input-root outputs \
  --rank-tsv outputs/qwen3-embedding-4b-ms-bs16/rank.tsv \
  --output-root outputs/qwen3-reranker-4b-cross-encoder \
  --top-k 500 \
  --batch-size 16 \
  --torch-dtype bfloat16
```

## License

本仓库采用 MIT License，详见 `LICENSE`。

上游 STARD 数据集和原始代码由 STARD 作者在
[STARD GitHub 仓库](https://github.com/oneal2000/STARD) 中提供，该仓库同样采用
MIT license。
