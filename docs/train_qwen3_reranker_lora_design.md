# `train_qwen3_reranker_lora.py` 详细设计文档

## 1. 背景与目标

`train_qwen3_reranker_lora.py` 用于在 STARD 法条检索任务上微调 `Qwen/Qwen3-Reranker-0.6B`。脚本采用 LoRA 方式训练 decoder/LLM reranker，将“用户咨询问题”和“候选法条”拼接为 Qwen3 Reranker 官方风格的 yes/no 判断提示词，通过 `logit("yes") - logit("no")` 得到相关性分数。

脚本的核心目标是：

- 读取 STARD 的 query、qrels、corpus、train/dev split。
- 读取一阶段检索结果 `rank.tsv`，默认使用 Qwen3-Embedding-4B dense retrieval 的 top-k 候选。
- 为训练集构造正样本、hard negative、random negative。
- 使用 Qwen3-Reranker-0.6B 作为基础模型，加载或新建 LoRA adapter。
- 使用二分类损失训练 query-document 相关性。
- 每个 epoch 保存 adapter，并在 dev split 上 rerank top-k 候选，输出 `rank.tsv` 和 metrics。

典型使用场景：

```bash
CUDA_VISIBLE_DEVICES=0 \
python train_qwen3_reranker_lora.py \
  --model-source modelscope \
  --model-cache /home/ls/stard_embedding_eval/models \
  --rank-tsv outputs/qwen3-embedding-4b-ms-bs16/rank.tsv \
  --output-dir outputs/qwen3-reranker-0.6b-lora-stard-bs14 \
  --candidate-top-k 500 \
  --hard-negatives 30 \
  --random-negatives 5 \
  --epochs 1 \
  --train-batch-size 14 \
  --gradient-accumulation-steps 2 \
  --eval-batch-size 16 \
  --max-length 1024
```

继续训练时：

```bash
CUDA_VISIBLE_DEVICES=0 \
python train_qwen3_reranker_lora.py \
  --adapter outputs/qwen3-reranker-0.6b-lora-stard-bs14/best \
  --epochs 10 \
  --output-dir outputs/qwen3-reranker-0.6b-lora-stard-bs14-continue10
```

## 2. 输入与输出

### 2.1 输入文件

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--queries` | `data/queries.json`，不存在则 `../STARD/data/queries.json` | STARD query 文件，包含 `query_id`、`问题`、`match_id`。 |
| `--corpus` | `data/corpus.jsonl`，不存在则 `../STARD/data/corpus.jsonl` | 法条库，每行一个 JSON，至少包含 `id`、`name`、`content`。 |
| `--train-split` | `data/example/train.query.txt`，不存在则 `../STARD/data/example/train.query.txt` | 训练 query id 列表。 |
| `--dev-split` | `data/example/dev.query.txt`，不存在则 `../STARD/data/example/dev.query.txt` | dev query id 列表。 |
| `--rank-tsv` | `outputs/qwen3-embedding-4b-ms-bs16/rank.tsv` | 一阶段检索候选，列格式为 `query_id, match_id, rank, score` 或至少前三列。 |
| `--adapter` | `None` | 可选 LoRA adapter 路径。指定后在已有 adapter 基础上继续训练。 |

### 2.2 训练与评估数据来源

当前脚本的数据来源需要区分 split、正样本、负样本和评估候选：

```text
train split：由 data/example/train.query.txt 决定
dev split：由 data/example/dev.query.txt 决定
正样本：由 queries.json 的 match_id 决定
hard negatives：由 qwen3-4b dense retrieval rank.tsv 决定；可从 topK 候选池中每个 epoch 随机采样
random negatives：由 corpus.jsonl 随机采样决定，默认每个 epoch 重新采样
dev rerank 候选：由 qwen3-4b dense retrieval rank.tsv 决定
```

因此，LoRA 训练输入并不只取决于 `data/example`。`data/example` 只决定哪些 query 进入 train/dev；训练阶段的 hard negatives 和评估阶段的候选集合都依赖 `--rank-tsv` 指向的一阶段检索结果。random negatives 来自 `corpus.jsonl`，默认会在每个 epoch 重新采样。

### 2.3 输出目录结构

`--output-dir` 下会生成：

```text
output-dir/
  train_config.json
  best/
    adapter_config.json
    adapter_model.safetensors
    tokenizer.json
    ...
  epoch-1/
    adapter_config.json
    adapter_model.safetensors
    tokenizer.json
    dev.rank.tsv
    dev.metrics.json
    epoch_metrics.json
  epoch-2/
    ...
```

主要输出：

- `train_config.json`：本次训练参数快照。
- `epoch-N/adapter_model.safetensors`：第 N 个 epoch 后的 LoRA adapter。
- `epoch-N/dev.rank.tsv`：dev split 每个 query 的 rerank 后候选排序。
- `epoch-N/dev.metrics.json`：dev split 的 Recall/MRR 指标。
- `epoch-N/epoch_metrics.json`：该 epoch 的指标副本；如果关闭评估则为空对象。
- `best/`：按 `MRR@10` 选择的最佳 adapter。

## 3. 核心设计

### 3.1 模型打分方式

Qwen3-Reranker 是 decoder/LLM reranker。脚本没有新增传统分类头，而是复用 causal LM 的最后 token logits：

```text
score(query, doc) = logit("yes") - logit("no")
```

训练时把这个分数送入 `BCEWithLogitsLoss`：

```text
label = 1.0 表示相关法条
label = 0.0 表示不相关法条
```

推理/评估时直接按 score 从高到低排序候选。

### 3.2 Prompt 模板

每个样本先格式化为：

```text
<Instruct>: Given a Chinese legal consultation query, retrieve legal articles that answer the query.
<Query>: 用户咨询问题：...
<Document>: 法条名称：...
法条内容：...
```

随后由 `RerankerCollator` 加上 Qwen chat 风格前后缀：

```text
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".
<|im_end|>
<|im_start|>user
...pair text...
<|im_end|>
<|im_start|>assistant
<think>

</think>

```

模型最后位置应预测 `yes` 或 `no`。脚本取最后位置的 `yes/no` logits 做相关性分数。

### 3.3 训练样本构造

每个训练 query 的样本由三部分组成：

1. 正样本：`queries.json` 中 `match_id` 对应的 gold 法条。
2. hard negatives：来自一阶段 `rank.tsv` 的 top-k 候选。默认过滤掉 gold 后取前 `--hard-negatives` 个；如果设置 `--hard-negative-pool-k` 和 `--resample-hard-negatives-each-epoch`，则先从 dense top pool 中构造非 gold 候选池，再每个 epoch 随机采 `--hard-negatives` 个。
3. random negatives：从整个 corpus 随机采样，过滤掉 gold 和已选 negatives，最多取 `--random-negatives` 个；默认每个 epoch 重新采样一次。

默认：

```text
hard_negatives = 30
hard_negative_pool_k = None
random_negatives = 5
candidate_top_k = 500
```

如果使用下面配置：

```text
hard_negative_pool_k = 200
hard_negatives = 30
resample_hard_negatives_each_epoch = True
random_negatives = 10
```

则语义是：

```text
从 dense top200 的非 gold 候选池里，每个 epoch 随机采 30 个 hard negatives
再从 corpus 全库随机采 10 个 random negatives
```

这意味着 reranker 主要学习在一阶段检索已经认为相似的法条中区分真相关和假相关。

### 3.4 评估方式

评估只在一阶段候选集合内部做 rerank，不做全库检索：

```text
rank.tsv top 500 candidates
  -> Qwen3-Reranker-0.6B LoRA 打分
  -> 按 score 降序排序
  -> 计算 Recall@K 和 MRR@K
```

默认指标：

```text
Recall@5,10,20,30,50,100
MRR@3,5,10
```

因此 `Recall@500` 约等于一阶段候选集对 dev gold 的召回上限；reranker 不能找回 top500 之外漏掉的 gold。

## 4. 总调用链

### 4.1 程序入口调用链

```text
main()
  -> parse_args()
      -> default_queries_path()
      -> default_corpus_path()
      -> default_train_split_path()
      -> default_dev_split_path()
      -> default_rank_path()
      -> normalize_path()
      -> 参数合法性检查
  -> train(args)
      -> read_queries()
      -> read_corpus()
      -> read_split_ids(train)
      -> read_split_ids(dev)
      -> read_rank_tsv()
      -> build_train_examples()
      -> load_model_and_tokenizer()
          -> maybe_download_model()
          -> AutoTokenizer.from_pretrained()
          -> AutoModelForCausalLM.from_pretrained()
          -> PeftModel.from_pretrained() 或 get_peft_model()
      -> RerankerCollator()
      -> PairDataset()
      -> DataLoader()
      -> AdamW / scheduler / BCEWithLogitsLoss
      -> for each epoch:
          -> for each train batch:
              -> PairDataset.__getitem__()
              -> RerankerCollator.__call__()
              -> compute_pair_logits()
              -> loss.backward()
              -> optimizer.step() / scheduler.step()
          -> model.save_pretrained(epoch_dir)
          -> tokenizer.save_pretrained(epoch_dir)
          -> evaluate_split()
              -> score_pairs()
                  -> RerankerCollator.__call__()
                  -> model(**batch)
                  -> yes/no logit difference
              -> evaluate()
              -> 写 dev.rank.tsv / dev.metrics.json
          -> 如 MRR@10 改善，保存 best adapter
```

### 4.2 训练阶段数据流

```text
queries.json + train.query.txt
        -> train_qids, questions, qrels

corpus.jsonl
        -> corpus[doc_id] = {id, name, content}

rank.tsv
        -> rank_by_qid[query_id] = [candidate_doc_id...]

build_train_examples()
        -> [PairExample(qid, doc_id, label)]

PairDataset.__getitem__()
        -> {"text": formatted_pair, "label": 0/1}

RerankerCollator.__call__()
        -> input_ids, attention_mask, labels

model(**batch)
        -> logits[:, -1, :]

compute_pair_logits()
        -> score = yes_logit - no_logit

BCEWithLogitsLoss(score, label)
        -> backward / optimizer step
```

### 4.3 评估阶段数据流

```text
dev query
  -> 取 rank.tsv topK candidate_ids
  -> 对每个 candidate 拼 prompt
  -> batch score_pairs()
  -> scored = sorted(candidate_ids, score, desc)
  -> rankings[qid] = [doc_id...]
  -> evaluate(rankings, qrels)
  -> 写 dev.rank.tsv / dev.metrics.json
```

## 5. 函数与类说明

### 5.1 常量

| 名称 | 作用 |
| --- | --- |
| `RECALL_KS` | Recall 评估的 K 集合。 |
| `MRR_KS` | MRR 评估的 K 集合。 |
| `DEFAULT_INSTRUCTION` | 默认检索任务说明，传入 Qwen reranker 的 `<Instruct>` 字段。 |
| `RERANKER_SYSTEM_PROMPT` | 系统提示，要求模型判断 Document 是否满足 Query，只能回答 yes/no。 |
| `RERANKER_SUFFIX` | Qwen reranker 需要的 assistant suffix，使最后一步 logits 对应 yes/no 判断。 |

### 5.2 路径与默认值函数

#### `normalize_path(path: str) -> str`

把路径中的 Windows 反斜杠替换为当前系统分隔符，并执行 `expanduser`、`normpath`。

主要用途：

- 支持用户传入 `outputs\xxx\rank.tsv` 这种 Windows 风格路径。
- 支持 `~` 展开。
- 统一后续文件读写路径。

#### `default_queries_path() -> str`

优先返回本仓库的 `data/queries.json`；不存在时回退到 `../STARD/data/queries.json`。

#### `default_corpus_path() -> str`

优先返回本仓库的 `data/corpus.jsonl`；不存在时回退到 `../STARD/data/corpus.jsonl`。

#### `default_train_split_path() -> str`

优先返回 `data/example/train.query.txt`；不存在时回退到 `../STARD/data/example/train.query.txt`。

#### `default_dev_split_path() -> str`

优先返回 `data/example/dev.query.txt`；不存在时回退到 `../STARD/data/example/dev.query.txt`。

#### `default_rank_path() -> str`

默认使用项目内的 Qwen3-Embedding-4B 一阶段候选：

```text
outputs/qwen3-embedding-4b-ms-bs16/rank.tsv
```

### 5.3 数据读取函数

#### `read_split_ids(path: str) -> list[str]`

读取 train/dev split 文件。

主要流程：

1. 按行读取文本。
2. 跳过空行。
3. 以 tab 分割，取第一列作为 query id。
4. 如果第一列为空，抛出 `ValueError`。
5. 返回 query id 字符串列表。

适配原因：`train.query.txt` 和 `dev.query.txt` 可能不止一列，但脚本只需要第一列 query id。

#### `read_queries(path: str) -> tuple[dict[str, str], dict[str, set[str]]]`

读取 `queries.json`，返回：

```python
questions: dict[str, str]
qrels: dict[str, set[str]]
```

其中：

- `questions[qid]` 是脱敏后的用户咨询问题，即 JSON 字段 `问题`。
- `qrels[qid]` 是 gold 法条 id 集合，即 JSON 字段 `match_id`。

主要流程：

1. 使用 `json.load` 读取整个 JSON。
2. 校验顶层必须是数组。
3. 遍历每个 query 对象。
4. 读取 `query_id` 和 `问题`。
5. 如果问题为空或不是字符串，则跳过该 query。
6. 遍历 `match_id`，加入 `qrels[qid]`。

#### `read_corpus(path: str) -> dict[str, dict[str, str]]`

读取 `corpus.jsonl` 法条库。

返回结构：

```python
corpus[doc_id] = {
    "id": doc_id,
    "name": article_name,
    "content": article_content,
}
```

主要流程：

1. 按 JSONL 逐行读取。
2. 跳过空行。
3. 解析每行为 JSON。
4. 读取 `id`、`name`、`content`。
5. 校验 `name` 和 `content` 必须是字符串。
6. 将 content 中的字面量 `\n` 替换为真实换行，并去除首尾空白。

#### `read_rank_tsv(path: str, top_k: int | None) -> dict[str, list[str]]`

读取一阶段检索结果。

输入格式至少前三列：

```text
query_id    match_id    rank    score
```

主要流程：

1. 逐行读取 TSV。
2. 如果第一行表头以 `query_id` 开头，则跳过。
3. 校验至少有三列。
4. 读取 `qid`、`doc_id`、`rank`。
5. 按 query 聚合为 `rows_by_qid[qid] = [(rank, doc_id), ...]`。
6. 对每个 query 按 rank 升序排序。
7. 去重 doc_id。
8. 如果指定 `top_k`，只保留前 top_k 个候选。

输出：

```python
rank_by_qid[qid] = [doc_id_1, doc_id_2, ...]
```

### 5.4 文本格式化函数

#### `format_query(question: str) -> str`

将原始问题包装为：

```text
用户咨询问题：...
```

#### `format_passage(article: dict[str, str]) -> str`

将法条对象包装为：

```text
法条名称：...
法条内容：...
```

#### `format_instruction(instruction: str, query: str, doc: str) -> str`

拼接 Qwen reranker 的用户侧任务内容：

```text
<Instruct>: ...
<Query>: ...
<Document>: ...
```

### 5.5 指标函数

#### `evaluate(rankings, qrels, recall_ks, mrr_ks) -> dict[str, float]`

计算 Recall@K 和 MRR@K。

Recall@K：

```text
Recall@K(q) = |topK(q) ∩ gold(q)| / |gold(q)|
```

对所有存在 gold 的 query 求平均。

MRR@K：

```text
MRR@K(q) = 1 / 第一个命中 gold 的 rank
```

如果 topK 内没有命中，则该 query 的 reciprocal rank 为 0。最后对所有存在 gold 的 query 求平均。

注意：

- 该函数只遍历 `rankings` 中出现的 query。
- 没有 gold 的 query 会跳过。
- Recall 使用集合交集，因此同一 doc 重复出现不会重复计数。

### 5.6 数据结构与 Dataset

#### `PairExample`

训练样本的轻量数据结构：

```python
@dataclass
class PairExample:
    qid: str
    doc_id: str
    label: float
```

字段含义：

- `qid`：query id。
- `doc_id`：候选法条 id。
- `label`：`1.0` 为相关，`0.0` 为不相关。

#### `build_train_examples(...) -> list[PairExample]`

构造训练样本，是脚本中最重要的数据准备函数之一。

输入：

- `train_qids`：训练 split 的 query id。
- `qrels`：gold 法条 id 集合。
- `rank_by_qid`：一阶段候选。
- `corpus`：法条库。
- `hard_negatives`：每个 query 从一阶段候选中取多少 hard negative。
- `random_negatives`：每个 query 从全库随机取多少 random negative。
- `seed`：随机种子。

主要流程：

1. 初始化随机数生成器 `random.Random(seed)`。
2. 取全库 doc id 列表，用于 random negative 采样。
3. 遍历每个训练 query。
4. 从 `qrels[qid]` 中取存在于 corpus 的 gold doc id。
5. 如果没有 gold，则记录 `missing_positive` 并跳过。
6. 对每个 gold doc id 添加正样本。
7. 从 `rank_by_qid[qid]` 顺序扫描候选，构造 hard negative pool：
   - 跳过 gold。
   - 跳过 corpus 中不存在的 doc。
   - 默认收集到 `hard_negatives` 数量后停止。
   - 如果设置 `hard_negative_pool_k`，则收集到 `hard_negative_pool_k` 数量后停止。
8. 如果开启 `--resample-hard-negatives-each-epoch` 且 pool 大于 `hard_negatives`，从 pool 中随机采 `hard_negatives` 个；否则取 pool 的前 `hard_negatives` 个。
9. 从全库随机采样补充 random negatives：
   - 跳过 gold。
   - 跳过已选 negatives。
   - 最多尝试 `random_negatives * 50` 次，避免死循环。
10. 如果开启 `--resample-random-negatives-each-epoch` 或 `--resample-hard-negatives-each-epoch`，每个 epoch 使用新的 seed 重建训练样本。
11. 将所有 negatives 添加为 label 0。
12. 对样本整体 shuffle。
13. 打印正负样本数量和缺失正样本 query 数。

设计意图：

- hard negatives 让模型学习区分“一阶段检索很像但其实不相关”的法条。
- random negatives 提供更简单的负样本，避免训练集全部是高相似难负例。
- 对每个 query 保留所有 gold，有利于多标签法条召回。

#### `PairDataset`

PyTorch Dataset 封装。

构造参数：

- `examples`
- `questions`
- `corpus`
- `instruction`

方法：

- `__len__()`：返回样本数。
- `__getitem__(index)`：
  1. 取 `PairExample`。
  2. 根据 `qid` 找到原始问题。
  3. 根据 `doc_id` 找到法条 name/content。
  4. 调用 `format_query()`、`format_passage()`、`format_instruction()`。
  5. 返回：

```python
{
    "text": formatted_pair_text,
    "label": 0.0 or 1.0,
}
```

### 5.7 Batch Collator

#### `RerankerCollator`

负责把 Dataset 输出的文本 batch 转换为 Qwen3-Reranker 可训练输入。

初始化流程：

1. 保存 tokenizer 和 `max_length`。
2. token 化 system/user 前缀：

```text
<|im_start|>system
...
<|im_end|>
<|im_start|>user
```

3. token 化 assistant 后缀：

```text
<|im_end|>
<|im_start|>assistant
<think>

</think>

```

4. 计算可留给 query-document pair 的最大 token 数：

```python
text_max_length = max_length - len(prefix_tokens) - len(suffix_tokens)
```

5. 如果 `text_max_length < 32`，抛出错误，说明 `--max-length` 太小。

`__call__(features)` 主要流程：

1. 取 batch 中所有 `text`。
2. 使用 tokenizer 对文本进行 tokenization：
   - `padding=False`
   - `truncation="longest_first"`
   - `return_attention_mask=False`
   - `max_length=text_max_length`
3. 对每条样本手动拼接：

```python
input_ids = prefix_tokens + pair_input_ids + suffix_tokens
```

4. 调用 `tokenizer.pad()` 做 batch padding，并返回 PyTorch tensor。
5. 把 label 转成 `torch.float32` tensor，写入 `batch["labels"]`。

复杂点：

- 这里手动控制 prefix/suffix，是为了保证最后一个位置对应 Qwen reranker 判断 yes/no 所需的上下文。
- `text_max_length` 先扣除 prefix/suffix 长度，避免最终序列超过 `--max-length`。
- tokenizer 使用 left padding，在 decoder-only 模型上通常更适合 batched inference/training。

### 5.8 模型加载与 LoRA

#### `maybe_download_model(model_name, model_source, model_cache) -> str`

根据参数决定模型路径。

主要流程：

1. 如果 `model_name` 本身是本地路径，直接返回规范化路径。
2. 如果指定 `model_cache`，设置：

```python
HF_HOME = model_cache
MODELSCOPE_CACHE = model_cache
```

3. 如果 `model_source == "modelscope"`，调用 ModelScope 的 `snapshot_download()` 下载或复用缓存。
4. 否则直接返回 `model_name`，由 Hugging Face 相关 API 处理。

设计意图：

- 支持国内环境通过 ModelScope 下载。
- 不需要手工传输模型权重。
- 支持本地已有模型路径。

#### `load_model_and_tokenizer(args)`

加载 tokenizer、基础模型，并加载或创建 LoRA adapter。

主要流程：

1. 调用 `maybe_download_model()` 获取实际模型路径。
2. 加载 tokenizer：

```python
AutoTokenizer.from_pretrained(model_path, padding_side="left")
```

3. 确保有 pad token：

```python
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
```

4. 获取 `yes` 和 `no` 的 token id：

```python
token_true_id = tokenizer.convert_tokens_to_ids("yes")
token_false_id = tokenizer.convert_tokens_to_ids("no")
```

5. 根据 `--torch-dtype` 选择 `float16`、`bfloat16` 或 `float32`。
6. 加载 `AutoModelForCausalLM`。
7. 关闭 cache：

```python
model.config.use_cache = False
```

8. 如果开启 gradient checkpointing：
   - `model.gradient_checkpointing_enable(...)`
   - `model.enable_input_require_grads()`
9. 如果指定 `--adapter`：
   - 用 `PeftModel.from_pretrained()` 加载已有 LoRA。
   - `is_trainable=not args.eval_only`，训练模式下继续更新 adapter。
10. 如果未指定 adapter 且不是 eval-only：
    - 构造 `LoraConfig`。
    - 调用 `get_peft_model()` 给模型注入 LoRA。
11. 打印可训练参数数量。
12. 移动模型到 `args.device`。
13. 返回模型、tokenizer、yes/no token id。

默认 LoRA target modules：

```text
q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

这覆盖注意力投影和 MLP 投影，是 decoder LLM LoRA 微调的常见配置。

### 5.9 打分函数

#### `compute_pair_logits(model, batch, token_true_id, token_false_id)`

训练阶段使用的 batch 打分函数。

主要流程：

1. 从 batch 中取出并移除 `labels`：

```python
labels = batch.pop("labels", None)
```

2. 前向计算：

```python
outputs = model(**batch)
```

3. 取最后一个 token 位置的 logits：

```python
final_logits = outputs.logits[:, -1, :]
```

4. 计算相关性分数：

```python
scores = final_logits[:, token_true_id] - final_logits[:, token_false_id]
```

5. 返回 `scores, labels`。

注意：

- 该函数会修改传入的 batch，因为它 `pop("labels")`。
- 分数没有 sigmoid，后续直接交给 `BCEWithLogitsLoss`。

#### `score_pairs(...) -> list[float]`

评估阶段对一批 query-document pair 打分。

主要流程：

1. `model.eval()`。
2. `torch.no_grad()` 下分批处理 pairs。
3. 把 pair text 包装成 collator 需要的 feature：

```python
{"text": text, "label": 0.0}
```

4. 调用 `collator()` 得到 batch。
5. 删除 labels。
6. batch tensor 移动到 GPU。
7. 前向计算最后位置 logits。
8. 计算 `yes_logit - no_logit`。
9. 收集 CPU float 分数。
10. 结束后 `model.train()`，恢复训练模式。

当前函数签名包含 `tokenizer` 参数，但函数体没有使用。这是一个可清理的小冗余，不影响行为。

### 5.10 Dev 评估函数

#### `evaluate_split(...) -> dict[str, float]`

对指定 query split 执行 rerank 和指标计算。

主要流程：

1. 初始化：

```python
rankings = {}
rank_rows = []
```

2. 遍历 dev qids。
3. 从 `rank_by_qid[qid]` 读取候选 doc ids，并过滤掉 corpus 中不存在的 doc。
4. 格式化 query。
5. 对每个候选法条格式化 passage，再拼成 pair text。
6. 调用 `score_pairs()` 得到每个候选分数。
7. 将 `(doc_id, score)` 按 score 降序排序。
8. 写入：

```python
rankings[qid] = [doc_id_1, doc_id_2, ...]
rank_rows.append((qid, doc_id, rank, score))
```

9. 调用 `evaluate()` 计算 Recall/MRR。
10. 写出：

```text
{output_dir}/{name}.rank.tsv
{output_dir}/{name}.metrics.json
```

11. 在 stdout 打印指标。
12. 返回 metrics。

输出的 `dev.rank.tsv` 每行格式：

```text
query_id    doc_id    rank    score
```

### 5.11 主训练函数

#### `train(args) -> None`

脚本的核心编排函数。

主要流程分为 7 个阶段。

#### 阶段 1：初始化

```python
start = time.perf_counter()
random.seed(args.seed)
torch.manual_seed(args.seed)
```

确保 Python random 和 torch 的随机性可复现。

#### 阶段 2：加载数据

```python
questions, qrels = read_queries(args.queries)
corpus = read_corpus(args.corpus)
train_qids = [qid for qid in read_split_ids(args.train_split) if qid in questions]
dev_qids = [qid for qid in read_split_ids(args.dev_split) if qid in questions]
rank_by_qid = read_rank_tsv(args.rank_tsv, args.candidate_top_k)
```

过滤 split 中不在 `questions` 的 query id，避免后续索引失败。

#### 阶段 3：构造训练样本

调用 `build_train_examples()` 构造正负样本。

如果指定 `--limit-train-examples`，则截断样本列表，主要用于 smoke test。

#### 阶段 4：加载模型与 DataLoader

```python
model, tokenizer, token_true_id, token_false_id = load_model_and_tokenizer(args)
collator = RerankerCollator(tokenizer, args.max_length)
train_dataset = PairDataset(examples, questions, corpus, args.instruction)
train_loader = DataLoader(...)
```

`DataLoader` 使用：

- `shuffle=True`
- 自定义 `collate_fn=collator`
- `num_workers=args.num_workers`

#### 阶段 5：优化器、调度器和损失函数

优化器：

```python
AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
```

调度器：

```python
get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
```

总步数计算：

```python
update_steps_per_epoch = ceil(len(train_loader) / gradient_accumulation_steps)
total_steps = update_steps_per_epoch * epochs
warmup_steps = int(total_steps * warmup_ratio)
```

损失：

```python
pos_weight = negatives / positives
BCEWithLogitsLoss(pos_weight=pos_weight)
```

`pos_weight` 用于缓解正负样本不均衡。默认 hard negatives 很多，负样本远多于正样本，因此启用 `--use-pos-weight` 较合理。

#### 阶段 6：训练循环

每个 epoch：

1. `model.train()`。
2. `optimizer.zero_grad(set_to_none=True)`。
3. 遍历 train_loader。
4. batch 移动到 GPU。
5. 调用 `compute_pair_logits()` 得到 scores 和 labels。
6. 计算 BCE loss。
7. loss 除以 `gradient_accumulation_steps` 后反向传播。
8. 到达累积步数或 epoch 最后一个 batch 时：
   - 可选 gradient clipping。
   - `optimizer.step()`。
   - `scheduler.step()`。
   - 清空梯度。
9. 每 `--log-every` 个 batch 更新 tqdm 的 loss/lr 显示。

复杂点：

- `train_batch_size` 是 micro batch size。
- 实际有效 batch 约为：

```text
train_batch_size * gradient_accumulation_steps
```

- 例如 `14 * 2 = 28`，与原来的 `4 * 8 = 32` 接近。

#### 阶段 7：保存与评估

每个 epoch 训练完后：

1. 保存 adapter 和 tokenizer：

```python
model.save_pretrained(epoch_dir)
tokenizer.save_pretrained(epoch_dir)
```

2. 如果 `--eval-each-epoch` 且 dev qids 非空：
   - 默认同步调用 `evaluate_split()`。
   - 如果设置 `--eval-device` 且不同于 `--device`，并保持 `--async-eval` 开启，则启动独立评估子进程。
   - 异步模式下，训练主进程保存 `epoch-N` adapter 后立即进入 `epoch-N+1` 训练；评估子进程在 `--eval-device` 上加载 `epoch-N` adapter 评估。
   - 如果上一个评估子进程还未结束，启动下一个评估前会等待，避免 GPU1 上叠加多个评估进程。
3. 同步模式由训练主进程写 `epoch_metrics.json` 并更新 `best/`；异步模式由评估子进程写 `epoch_metrics.json` 并按 `MRR@10` 更新 `best/`。
4. 全部训练结束后，如果最后一个异步评估仍在运行，主进程会等待它完成。

### 5.12 参数解析

#### `parse_args() -> argparse.Namespace`

负责定义命令行参数、应用默认值、规范化路径和做基本参数校验。

重要参数分组：

#### 数据路径

- `--queries`
- `--corpus`
- `--train-split`
- `--dev-split`
- `--rank-tsv`
- `--output-dir`

#### 模型与 adapter

- `--model`
- `--model-source`
- `--model-cache`
- `--adapter`
- `--best-dir`
- `--eval-only`

`--eval-only` 用于评估子进程：加载指定 `--adapter`，在 `--device` 上执行 dev rerank，并把 `dev.rank.tsv`、`dev.metrics.json`、`epoch_metrics.json` 写到 `--output-dir`。

#### 样本构造

- `--candidate-top-k`
- `--hard-negative-pool-k`
- `--hard-negatives`
- `--random-negatives`
- `--instruction`

#### 训练超参

- `--epochs`
- `--train-batch-size`
- `--eval-batch-size`
- `--gradient-accumulation-steps`
- `--learning-rate`
- `--weight-decay`
- `--warmup-ratio`
- `--max-grad-norm`
- `--max-length`

#### LoRA 超参

- `--lora-r`
- `--lora-alpha`
- `--lora-dropout`
- `--lora-target-modules`

#### 性能与设备

- `--torch-dtype`
- `--attn-implementation`
- `--device`
- `--eval-device`
- `--gradient-checkpointing`
- `--num-workers`

#### 调试与控制

- `--use-pos-weight`
- `--eval-each-epoch`
- `--async-eval`
- `--resample-hard-negatives-each-epoch`
- `--resample-random-negatives-each-epoch`
- `--limit-train-examples`
- `--log-every`
- `--seed`

参数校验：

- `candidate_top_k >= 1`
- hard/random negatives 非负
- batch size 为正
- gradient accumulation 为正
- epochs 为正

### 5.13 程序入口

#### `main() -> None`

主入口。

主要流程：

1. 调用 `parse_args()`。
2. 如果 `args.eval_only` 为真，抛出 `NotImplementedError`。
3. 调用 `train(args)`。

## 6. 关键行为说明

### 6.1 继续训练行为

指定 `--adapter` 时，脚本会：

1. 先加载基座模型。
2. 再加载 LoRA adapter。
3. 设置 `is_trainable=True`。
4. 继续训练该 adapter。

继续训练不会自动读取上一次 optimizer/scheduler 状态。也就是说，它是“从已有 adapter 权重继续优化”，但优化器和学习率调度器是新建的。

如果希望严格 resume，包括 optimizer state、scheduler state、global step，需要额外保存和恢复 checkpoint，目前脚本没有实现。

### 6.2 评估基准含义

当前评估依赖 `rank.tsv` 候选，例如：

```text
outputs/qwen3-embedding-4b-ms-bs16/rank.tsv
```

配合：

```text
--candidate-top-k 500
```

含义是：

```text
Qwen3-Embedding-4B dense retrieval top500
  -> Qwen3-Reranker-0.6B LoRA rerank
  -> 在 rerank 后结果上计算 Recall/MRR
```

因此指标衡量的是 reranker 对 top500 候选排序的改善，不衡量全库召回能力。

### 6.3 GPU 使用控制

脚本内部默认 `--device cuda`，不负责选择具体 GPU。应通过环境变量控制：

```bash
CUDA_VISIBLE_DEVICES=0 python train_qwen3_reranker_lora.py ...
```

### 6.4 ModelScope 下载策略

默认：

```text
--model-source modelscope
```

配合：

```text
--model-cache /path/to/model-cache
```

可以直接从 ModelScope 下载或复用模型缓存，不需要手工传输权重。

## 7. 复杂函数流程图

### 7.1 `build_train_examples()`

```text
for qid in train_qids:
    gold = qrels[qid] ∩ corpus
    if gold is empty:
        missing_positive += 1
        continue

    add all gold as positive examples

    negatives = []
    for doc_id in rank_by_qid[qid]:
        if doc_id is gold:
            continue
        if doc_id not in corpus:
            continue
        negatives.append(doc_id)
        if len(negatives) >= hard_negatives:
            break

    while len(negatives) < hard_negatives + random_negatives:
        doc_id = random corpus doc
        if doc_id already used or gold:
            continue
        negatives.append(doc_id)

    add negatives as label 0

shuffle all examples
return examples
```

### 7.2 `RerankerCollator.__call__()`

```text
features: [{"text": ..., "label": ...}, ...]
  -> tokenizer(texts, truncation, max_length=text_max_length)
  -> for each input:
         input_ids = prefix_tokens + input_ids + suffix_tokens
  -> tokenizer.pad(...)
  -> labels tensor
  -> batch
```

### 7.3 `train()`

```text
load data
build examples
load model/tokenizer/LoRA
build collator/dataset/dataloader
build optimizer/scheduler/loss

for epoch:
    for batch:
        batch -> GPU
        scores = yes_logit - no_logit
        loss = BCEWithLogitsLoss(scores, labels)
        backward with gradient accumulation
        optimizer/scheduler step

    save epoch adapter
    evaluate dev
    save best if MRR@10 improves
```

### 7.4 `evaluate_split()`

```text
for qid in dev_qids:
    candidate_ids = rank_by_qid[qid]
    pairs = [(query, document), ...]
    scores = score_pairs(pairs)
    sorted candidates by score desc
    rankings[qid] = sorted_doc_ids
    rank_rows += TSV rows

metrics = evaluate(rankings, qrels)
write dev.rank.tsv
write dev.metrics.json
return metrics
```

## 8. 设计取舍

### 8.1 使用 pointwise BCE 而不是 pairwise/listwise loss

当前脚本把每个 query-document pair 当成独立二分类样本。

优点：

- 实现简单。
- 能直接利用 yes/no logits。
- 适合 LoRA 快速验证。
- 正负样本构造直接。

缺点：

- 排序目标是间接优化的，不如 pairwise/listwise 直接。
- 同一个 query 下候选之间的相对顺序没有被显式建模。

后续可扩展：

- Pairwise RankNet loss。
- ListNet/ListMLE。
- InfoNCE，多负样本同 batch 对比。
- LambdaRank/LambdaLoss，直接对 NDCG/MRR 风格目标建模。

### 8.2 使用 `pos_weight`

默认 hard negatives 较多，训练样本严重负样本占优。`pos_weight = negatives / positives` 可以提高正样本损失权重，避免模型倾向输出低相关分。

风险：

- 如果 negatives 中假负例较多，过高的 `pos_weight` 可能导致训练不稳定。
- 可以通过 `--no-use-pos-weight` 关闭。

### 8.3 每个 epoch 都完整 dev rerank

优点：

- 可以观察真实 Recall/MRR。
- 可以自动保存 best adapter。

缺点：

- dev rerank 成本较高。以 `308 queries * top500` 为例，需要对约 15.4 万个 query-doc pair 前向打分。
- 多 epoch 训练时，评估会占用明显时间。

如果只想快速训练，可以用：

```bash
--no-eval-each-epoch
```

但这样不会生成 dev metrics，也不会更新 best adapter。

## 9. 已知限制与改进建议

### 9.1 `--eval-only` 尚未实现

当前 `main()` 中对 `--eval-only` 直接抛错。可以补充一个 `eval_only(args)`：

- 加载 base model。
- 加载 adapter。
- 构建 collator。
- 对 dev split 调用 `evaluate_split()`。

### 9.2 无 optimizer/scheduler checkpoint

继续训练只恢复 LoRA adapter 权重，不恢复 optimizer、scheduler、global step。

如果需要严格断点续训，应保存：

- optimizer state dict
- scheduler state dict
- epoch/global step
- random states

### 9.3 `score_pairs()` 的 `tokenizer` 参数未使用

这是轻微冗余，可以删除参数以简化函数签名。

### 9.4 Hard/random negatives 重采样

当前脚本默认每个 epoch 重新构造训练样本。正样本由 `queries.json` 决定。默认 hard negatives 由 `rank.tsv` 的最高排名非 gold 候选决定，通常保持稳定；random negatives 由 `corpus.jsonl` 按 epoch seed 重新采样。

如果设置：

```bash
--hard-negative-pool-k 200
--resample-hard-negatives-each-epoch
```

则 hard negatives 也会参与重采样：每个 query 先从 dense top200 非 gold 候选构造 pool，再按当前 epoch seed 随机采 `--hard-negatives` 个。

默认 seed 规则：

```text
epoch_seed = seed + epoch - 1
```

因此相同数据、相同参数、相同 seed 下，hard/random negatives 的重采样仍然可复现。

可以用下面参数关闭该行为，回到“启动时采样一次，所有 epoch 复用”的模式：

```bash
--no-resample-random-negatives-each-epoch
```

后续仍可改进：

- 使用上一轮模型挖掘更难的 hard negatives。
- 混合 BM25/QLD/dense 多来源 negatives。

### 9.5 评估只覆盖候选集合

由于 reranker 只重排 `rank.tsv` 的 top-k 候选，无法改善一阶段召回遗漏。若 Recall@500 已经到达上限，继续训练只能提升小 K 排序质量，不能提升 top500 外召回。

## 10. 推荐运行配置

### 10.1 单 GPU 训练

```bash
CUDA_VISIBLE_DEVICES=0 \
python train_qwen3_reranker_lora.py \
  --model-source modelscope \
  --model-cache /path/to/model-cache \
  --rank-tsv outputs/qwen3-embedding-4b-ms-bs16/rank.tsv \
  --output-dir outputs/qwen3-reranker-0.6b-lora-stard-bs14 \
  --candidate-top-k 500 \
  --hard-negatives 30 \
  --random-negatives 5 \
  --epochs 1 \
  --train-batch-size 14 \
  --gradient-accumulation-steps 2 \
  --eval-batch-size 16 \
  --max-length 1024 \
  --learning-rate 2e-4 \
  --log-every 20
```

### 10.2 继续训练 10 个 epoch

```bash
cd ~/stard_embedding_eval/STARD

CUDA_VISIBLE_DEVICES=0 \
HF_HOME=/home/ls/stard_embedding_eval/hf_cache \
MODELSCOPE_CACHE=/home/ls/stard_embedding_eval/models \
../.venv/bin/python train_qwen3_reranker_lora.py \
  --model-source modelscope \
  --model-cache /home/ls/stard_embedding_eval/models \
  --adapter outputs/qwen3-reranker-0.6b-lora-stard-bs14/best \
  --rank-tsv outputs/qwen3-embedding-4b-ms-bs16/rank.tsv \
  --output-dir outputs/qwen3-reranker-0.6b-lora-stard-bs14-continue10 \
  --candidate-top-k 500 \
  --hard-negatives 30 \
  --random-negatives 5 \
  --epochs 10 \
  --train-batch-size 14 \
  --gradient-accumulation-steps 2 \
  --eval-batch-size 16 \
  --max-length 1024 \
  --learning-rate 2e-4 \
  --log-every 20
```

## 11. 和当前实验的关系

当前实验设置可概括为：

```text
一阶段候选：
  Qwen3-Embedding-4B dense retrieval, top500

二阶段重排：
  Qwen3-Reranker-0.6B + LoRA

训练数据：
  STARD train split
  positives = qrels gold 法条
  negatives = dense top500 hard negatives + corpus random negatives

评估数据：
  STARD dev split

主要指标：
  Recall@5/10/20/30/50/100
  MRR@3/5/10
```

上一次 1 epoch 训练完成后，dev 上的结果为：

```text
Recall@5    0.5346
Recall@10   0.6319
Recall@20   0.7803
Recall@30   0.8155
Recall@50   0.8698
Recall@100  0.9276
MRR@3       0.4746
MRR@5       0.4903
MRR@10      0.5024
```

当前脚本仍对 dense top500 候选进行 rerank，但默认只计算到 `Recall@100`。
