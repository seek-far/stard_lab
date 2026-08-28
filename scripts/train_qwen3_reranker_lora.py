"""LoRA training for Qwen3-Reranker on STARD.

训练数据来源：
- train split：由 data/example/train.query.txt 决定。
- dev split：由 data/example/dev.query.txt 决定。
- 正样本：由 queries.json 的 match_id 决定。
- hard negatives：由一阶段 dense retrieval rank.tsv 决定，默认是 Qwen3-Embedding-4B topK 结果。
- random negatives：由 corpus.jsonl 随机采样决定，默认每个 epoch 重新采样。
- dev rerank 候选：由一阶段 dense retrieval rank.tsv 决定。
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from tqdm import tqdm


METRIC_KS = [3, 5, 10, 20, 30, 50]
RECALL_KS = METRIC_KS
MRR_KS = METRIC_KS
DEFAULT_INSTRUCTION = (
    "Given a Chinese legal consultation query, retrieve legal articles that answer the query."
)
RERANKER_SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query and the Instruct "
    'provided. Note that the answer can only be "yes" or "no".'
)
RERANKER_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def normalize_path(path: str) -> str:
    return os.path.normpath(os.path.expanduser(path.replace("\\", os.sep)))


def default_queries_path() -> str:
    if os.path.exists("data/queries.json"):
        return "data/queries.json"
    return "../STARD/data/queries.json"


def default_corpus_path() -> str:
    if os.path.exists("data/corpus.jsonl"):
        return "data/corpus.jsonl"
    return "../STARD/data/corpus.jsonl"


def default_train_split_path() -> str:
    if os.path.exists("data/example/train.query.txt"):
        return "data/example/train.query.txt"
    return "../STARD/data/example/train.query.txt"


def default_dev_split_path() -> str:
    if os.path.exists("data/example/dev.query.txt"):
        return "data/example/dev.query.txt"
    return "../STARD/data/example/dev.query.txt"


def default_rank_path() -> str:
    local_path = "outputs/qwen3-embedding-4b-ms-bs16/rank.tsv"
    if os.path.exists(local_path):
        return local_path
    return local_path


def read_split_ids(path: str) -> list[str]:
    qids = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            qid = line.split("\t", 1)[0]
            if not qid:
                raise ValueError(f"{path}:{line_no} missing query id")
            qids.append(str(qid))
    return qids


def read_queries(path: str) -> tuple[dict[str, str], dict[str, set[str]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")

    questions = {}
    qrels: dict[str, set[str]] = defaultdict(set)
    for obj in data:
        qid = str(obj["query_id"])
        question = obj.get("问题")
        if not isinstance(question, str) or not question.strip():
            continue
        questions[qid] = question.strip()
        for match_id in obj.get("match_id", []):
            qrels[qid].add(str(match_id))
    return questions, qrels


def read_corpus(path: str) -> dict[str, dict[str, str]]:
    corpus = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = str(obj["id"])
            name = obj.get("name")
            content = obj.get("content")
            if not isinstance(name, str) or not isinstance(content, str):
                raise ValueError(f"{path}:{line_no} must contain string name and content")
            corpus[doc_id] = {
                "id": doc_id,
                "name": name,
                "content": content.replace("\\n", "\n").strip(),
            }
    return corpus


def read_rank_tsv(path: str, top_k: int | None) -> dict[str, list[str]]:
    rows_by_qid: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if line_no == 1 and parts[0] == "query_id":
                continue
            if len(parts) < 3:
                raise ValueError(f"{path}:{line_no} must have at least 3 tab-separated columns")
            qid, doc_id, rank = parts[:3]
            rows_by_qid[str(qid)].append((int(rank), str(doc_id)))

    output = {}
    for qid, rows in rows_by_qid.items():
        rows.sort(key=lambda item: item[0])
        seen = set()
        doc_ids = []
        for _rank, doc_id in rows:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc_ids.append(doc_id)
            if top_k is not None and len(doc_ids) >= top_k:
                break
        output[qid] = doc_ids
    return output


def format_query(question: str) -> str:
    return f"用户咨询问题：{question.strip()}"


def format_passage(article: dict[str, str]) -> str:
    return f"法条名称：{article['name']}\n法条内容：{article['content']}"


def format_instruction(instruction: str, query: str, doc: str) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


def evaluate(
    rankings: dict[str, list[str]],
    qrels: dict[str, set[str]],
    recall_ks: list[int],
    mrr_ks: list[int],
) -> dict[str, float]:
    metrics = {}
    for k in recall_ks:
        total = 0.0
        count = 0
        for qid, ranked_docids in rankings.items():
            gold = qrels.get(qid, set())
            if not gold:
                continue
            total += len(set(ranked_docids[:k]) & gold) / len(gold)
            count += 1
        metrics[f"Recall@{k}"] = total / count if count else 0.0

    for k in mrr_ks:
        total = 0.0
        count = 0
        for qid, ranked_docids in rankings.items():
            gold = qrels.get(qid, set())
            if not gold:
                continue
            rr = 0.0
            for rank, doc_id in enumerate(ranked_docids[:k], start=1):
                if doc_id in gold:
                    rr = 1.0 / rank
                    break
            total += rr
            count += 1
        metrics[f"MRR@{k}"] = total / count if count else 0.0
    return metrics


@dataclass
class PairExample:
    qid: str
    doc_id: str
    label: float


def build_train_examples(
    train_qids: list[str],
    qrels: dict[str, set[str]],
    rank_by_qid: dict[str, list[str]],
    corpus: dict[str, dict[str, str]],
    hard_negatives: int,
    hard_negative_pool_k: int | None,
    resample_hard_negatives: bool,
    random_negatives: int,
    seed: int,
) -> list[PairExample]:
    rng = random.Random(seed)
    all_doc_ids = list(corpus.keys())
    examples = []
    missing_positive = 0

    for qid in train_qids:
        gold = [doc_id for doc_id in sorted(qrels.get(qid, set())) if doc_id in corpus]
        if not gold:
            missing_positive += 1
            continue
        gold_set = set(gold)
        for doc_id in gold:
            examples.append(PairExample(qid=qid, doc_id=doc_id, label=1.0))

        hard_pool = []
        pool_limit = hard_negative_pool_k if hard_negative_pool_k is not None else hard_negatives
        for doc_id in rank_by_qid.get(qid, []):
            if doc_id in gold_set or doc_id not in corpus:
                continue
            hard_pool.append(doc_id)
            if len(hard_pool) >= pool_limit:
                break

        if resample_hard_negatives and len(hard_pool) > hard_negatives:
            negatives = rng.sample(hard_pool, hard_negatives)
        else:
            negatives = hard_pool[:hard_negatives]

        random_seen = set(negatives) | gold_set
        attempts = 0
        while len(negatives) < hard_negatives + random_negatives and attempts < random_negatives * 50:
            attempts += 1
            doc_id = rng.choice(all_doc_ids)
            if doc_id in random_seen:
                continue
            random_seen.add(doc_id)
            negatives.append(doc_id)

        for doc_id in negatives:
            examples.append(PairExample(qid=qid, doc_id=doc_id, label=0.0))

    rng.shuffle(examples)
    print(
        f"train_examples={len(examples)} positives={sum(1 for e in examples if e.label == 1.0)} "
        f"negatives={sum(1 for e in examples if e.label == 0.0)} missing_positive_queries={missing_positive} "
        f"hard_negative_pool_k={hard_negative_pool_k or hard_negatives} "
        f"resample_hard_negatives={resample_hard_negatives}"
    )
    return examples


class PairDataset:
    def __init__(
        self,
        examples: list[PairExample],
        questions: dict[str, str],
        corpus: dict[str, dict[str, str]],
        instruction: str,
    ) -> None:
        self.examples = examples
        self.questions = questions
        self.corpus = corpus
        self.instruction = instruction

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        query = format_query(self.questions[example.qid])
        doc = format_passage(self.corpus[example.doc_id])
        return {
            "text": format_instruction(self.instruction, query, doc),
            "label": example.label,
        }


class RerankerCollator:
    def __init__(self, tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prefix_tokens = tokenizer.encode(
            f"<|im_start|>system\n{RERANKER_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n",
            add_special_tokens=False,
        )
        self.suffix_tokens = tokenizer.encode(RERANKER_SUFFIX, add_special_tokens=False)
        self.text_max_length = max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        if self.text_max_length < 32:
            raise ValueError("--max-length is too small for the Qwen reranker prompt")

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        texts = [feature["text"] for feature in features]
        tokenized = self.tokenizer(
            texts,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.text_max_length,
        )
        tokenized["input_ids"] = [
            self.prefix_tokens + input_ids + self.suffix_tokens
            for input_ids in tokenized["input_ids"]
        ]
        batch = self.tokenizer.pad(
            tokenized,
            padding=True,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor([feature["label"] for feature in features], dtype=torch.float32)
        return batch


def maybe_download_model(model_name: str, model_source: str, model_cache: str | None) -> str:
    if os.path.exists(model_name):
        return normalize_path(model_name)
    if model_cache:
        os.environ.setdefault("HF_HOME", normalize_path(model_cache))
        os.environ.setdefault("MODELSCOPE_CACHE", normalize_path(model_cache))
    if model_source == "modelscope":
        from modelscope import snapshot_download

        return snapshot_download(model_name, cache_dir=normalize_path(model_cache) if model_cache else None)
    return model_name


def load_model_and_tokenizer(args: argparse.Namespace):
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = maybe_download_model(args.model, args.model_source, args.model_cache)
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    token_true_id = tokenizer.convert_tokens_to_ids("yes")
    token_false_id = tokenizer.convert_tokens_to_ids("no")
    if token_true_id is None or token_false_id is None:
        raise ValueError("tokenizer must contain yes/no tokens")

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.torch_dtype]
    model_kwargs = {
        "torch_dtype": dtype,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    if args.adapter:
        model = PeftModel.from_pretrained(model, normalize_path(args.adapter), is_trainable=not args.eval_only)
    elif not args.eval_only:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[module.strip() for module in args.lora_target_modules.split(",") if module.strip()],
            bias="none",
        )
        model = get_peft_model(model, lora_config)

    if not args.eval_only:
        model.print_trainable_parameters()
    model.to(args.device)
    return model, tokenizer, token_true_id, token_false_id


def compute_pair_logits(model, batch: dict[str, Any], token_true_id: int, token_false_id: int):
    labels = batch.pop("labels", None)
    outputs = model(**batch)
    final_logits = outputs.logits[:, -1, :]
    scores = final_logits[:, token_true_id] - final_logits[:, token_false_id]
    return scores, labels


def score_pairs(
    model,
    tokenizer,
    collator: RerankerCollator,
    pairs: list[tuple[str, str]],
    token_true_id: int,
    token_false_id: int,
    batch_size: int,
    device: str,
) -> list[float]:
    import torch

    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            features = [{"text": text, "label": 0.0} for text, _doc_id in pairs[start : start + batch_size]]
            batch = collator(features)
            batch.pop("labels", None)
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            final_logits = outputs.logits[:, -1, :]
            batch_scores = final_logits[:, token_true_id] - final_logits[:, token_false_id]
            scores.extend(batch_scores.float().cpu().tolist())
    model.train()
    return scores


def evaluate_split(
    model,
    tokenizer,
    collator: RerankerCollator,
    qids: list[str],
    questions: dict[str, str],
    qrels: dict[str, set[str]],
    corpus: dict[str, dict[str, str]],
    rank_by_qid: dict[str, list[str]],
    instruction: str,
    token_true_id: int,
    token_false_id: int,
    batch_size: int,
    device: str,
    output_dir: str,
    name: str,
) -> dict[str, float]:
    rankings = {}
    rank_rows = []
    for qid in tqdm(qids, desc=f"evaluating {name}"):
        candidate_ids = [doc_id for doc_id in rank_by_qid.get(qid, []) if doc_id in corpus]
        query = format_query(questions[qid])
        pairs = [
            (format_instruction(instruction, query, format_passage(corpus[doc_id])), doc_id)
            for doc_id in candidate_ids
        ]
        scores = score_pairs(
            model,
            tokenizer,
            collator,
            pairs,
            token_true_id,
            token_false_id,
            batch_size=batch_size,
            device=device,
        )
        scored = sorted(zip(candidate_ids, scores), key=lambda item: item[1], reverse=True)
        rankings[qid] = [doc_id for doc_id, _score in scored]
        for rank, (doc_id, score) in enumerate(scored, start=1):
            rank_rows.append((qid, doc_id, rank, score))

    metrics = evaluate(rankings, qrels, recall_ks=RECALL_KS, mrr_ks=MRR_KS)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{name}.rank.tsv"), "w", encoding="utf-8") as f:
        for qid, doc_id, rank, score in rank_rows:
            f.write(f"{qid}\t{doc_id}\t{rank}\t{score:.8f}\n")
    with open(os.path.join(output_dir, f"{name}.metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"{name} metrics")
    for key, value in metrics.items():
        print(f"{key}\t{value:.4f}")
    return metrics


def load_existing_best_mrr(output_dir: str, start_epoch: int) -> float:
    best_mrr = -1.0
    for epoch in range(1, start_epoch):
        metrics_path = os.path.join(output_dir, f"epoch-{epoch}", "epoch_metrics.json")
        if not os.path.exists(metrics_path):
            continue
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        best_mrr = max(best_mrr, float(metrics.get("MRR@10", -1.0)))
    return best_mrr


def evaluate_epoch(
    args: argparse.Namespace,
    model,
    tokenizer,
    collator: RerankerCollator,
    epoch_dir: str,
    dev_qids: list[str],
    questions: dict[str, str],
    qrels: dict[str, set[str]],
    corpus: dict[str, dict[str, str]],
    rank_by_qid: dict[str, list[str]],
    token_true_id: int,
    token_false_id: int,
) -> dict[str, float]:
    eval_device = args.eval_device or args.device
    if eval_device == args.device:
        return evaluate_split(
            model,
            tokenizer,
            collator,
            dev_qids,
            questions,
            qrels,
            corpus,
            rank_by_qid,
            args.instruction,
            token_true_id,
            token_false_id,
            batch_size=args.eval_batch_size,
            device=eval_device,
            output_dir=epoch_dir,
            name="dev",
        )

    import torch

    eval_args = argparse.Namespace(**vars(args))
    eval_args.adapter = epoch_dir
    eval_args.device = eval_device
    eval_args.eval_only = True
    eval_args.gradient_checkpointing = False
    print(f"loading_eval_adapter={epoch_dir} eval_device={eval_device}")
    eval_model, eval_tokenizer, eval_true_id, eval_false_id = load_model_and_tokenizer(eval_args)
    eval_collator = RerankerCollator(eval_tokenizer, args.max_length)
    try:
        return evaluate_split(
            eval_model,
            eval_tokenizer,
            eval_collator,
            dev_qids,
            questions,
            qrels,
            corpus,
            rank_by_qid,
            args.instruction,
            eval_true_id,
            eval_false_id,
            batch_size=args.eval_batch_size,
            device=eval_device,
            output_dir=epoch_dir,
            name="dev",
        )
    finally:
        del eval_model
        torch.cuda.empty_cache()


def wait_for_eval_process(process: subprocess.Popen | None, epoch_dir: str | None) -> None:
    if process is None:
        return
    code = process.wait()
    if code != 0:
        raise RuntimeError(f"async eval failed for {epoch_dir} with exit code {code}")


def build_in_progress_save_steps(
    batches_per_epoch: int,
    gradient_accumulation_steps: int,
    save_fraction: float,
) -> set[int]:
    if save_fraction <= 0 or save_fraction >= 1 or batches_per_epoch <= 1:
        return set()

    save_steps = set()
    progress = save_fraction
    while progress < 1.0:
        step = max(1, math.ceil(batches_per_epoch * progress))
        remainder = step % gradient_accumulation_steps
        if remainder:
            step += gradient_accumulation_steps - remainder
        if step < batches_per_epoch:
            save_steps.add(step)
        progress += save_fraction
    return save_steps


def save_in_progress_adapter(
    model,
    tokenizer,
    output_dir: str,
    epoch: int,
    step: int,
    batches_per_epoch: int,
    optimizer_step: int,
    args: argparse.Namespace,
) -> None:
    checkpoint_dir = os.path.join(output_dir, f"epoch-{epoch}-in-progress")
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    state = {
        "epoch": epoch,
        "step": step,
        "batches_per_epoch": batches_per_epoch,
        "progress": step / batches_per_epoch,
        "optimizer_step": optimizer_step,
        "resume_adapter": checkpoint_dir,
        "resume_start_epoch": epoch,
        "resume_step": step,
        "train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "seed": args.seed,
        "resample_hard_negatives_each_epoch": args.resample_hard_negatives_each_epoch,
        "resample_random_negatives_each_epoch": args.resample_random_negatives_each_epoch,
    }
    with open(os.path.join(checkpoint_dir, "checkpoint_state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    print(f"saved_in_progress_adapter={checkpoint_dir} step={step}/{batches_per_epoch}")


def bool_flag_args(name: str, value: bool) -> list[str]:
    return [f"--{name}" if value else f"--no-{name}"]


def build_eval_command(args: argparse.Namespace, epoch_dir: str) -> list[str]:
    eval_device = args.eval_device or args.device
    command = [
        sys.executable,
        os.path.abspath(__file__),
        "--eval-only",
        "--queries",
        args.queries,
        "--corpus",
        args.corpus,
        "--dev-split",
        args.dev_split,
        "--rank-tsv",
        args.rank_tsv,
        "--output-dir",
        epoch_dir,
        "--model",
        args.model,
        "--model-source",
        args.model_source,
        "--adapter",
        epoch_dir,
        "--instruction",
        args.instruction,
        "--candidate-top-k",
        str(args.candidate_top_k),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--max-length",
        str(args.max_length),
        "--torch-dtype",
        args.torch_dtype,
        "--device",
        eval_device,
        "--best-dir",
        os.path.join(args.output_dir, "best"),
    ]
    if args.model_cache:
        command.extend(["--model-cache", args.model_cache])
    if args.attn_implementation:
        command.extend(["--attn-implementation", args.attn_implementation])
    command.extend(bool_flag_args("gradient-checkpointing", False))
    return command


def launch_epoch_eval(
    args: argparse.Namespace,
    epoch_dir: str,
    active_eval: tuple[subprocess.Popen, str] | None,
) -> tuple[subprocess.Popen, str]:
    if active_eval is not None:
        process, active_epoch_dir = active_eval
        print(f"waiting_previous_eval={active_epoch_dir}")
        wait_for_eval_process(process, active_epoch_dir)

    command = build_eval_command(args, epoch_dir)
    log_path = os.path.join(epoch_dir, "dev.eval.log")
    print(f"launching_async_eval epoch_dir={epoch_dir} eval_device={args.eval_device or args.device} log={log_path}")
    log = open(log_path, "w", encoding="utf-8")
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, close_fds=True)
    return process, epoch_dir


def save_best_if_needed(
    model,
    tokenizer,
    output_dir: str,
    best_dir: str | None,
    metric_name: str = "MRR@10",
) -> None:
    if not best_dir:
        return
    metrics_path = os.path.join(output_dir, "epoch_metrics.json")
    if not os.path.exists(metrics_path):
        return
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    current = float(metrics.get(metric_name, -1.0))
    parent = os.path.dirname(output_dir)
    best_existing = -1.0
    for name in os.listdir(parent):
        if not name.startswith("epoch-"):
            continue
        sibling_metrics = os.path.join(parent, name, "epoch_metrics.json")
        if not os.path.exists(sibling_metrics):
            continue
        with open(sibling_metrics, "r", encoding="utf-8") as f:
            sibling = json.load(f)
        if os.path.abspath(os.path.join(parent, name)) == os.path.abspath(output_dir):
            continue
        best_existing = max(best_existing, float(sibling.get(metric_name, -1.0)))
    if current > best_existing:
        os.makedirs(best_dir, exist_ok=True)
        model.save_pretrained(best_dir)
        tokenizer.save_pretrained(best_dir)
        print(f"saved_best_adapter={best_dir} {metric_name}={current:.4f}")


def eval_only(args: argparse.Namespace) -> None:
    print("loading data")
    questions, qrels = read_queries(args.queries)
    corpus = read_corpus(args.corpus)
    dev_qids = [qid for qid in read_split_ids(args.dev_split) if qid in questions]
    rank_by_qid = read_rank_tsv(args.rank_tsv, args.candidate_top_k)
    print(
        f"queries={len(questions)} corpus={len(corpus)} dev_queries={len(dev_qids)} "
        f"candidate_queries={len(rank_by_qid)}"
    )
    args.gradient_checkpointing = False
    model, tokenizer, token_true_id, token_false_id = load_model_and_tokenizer(args)
    collator = RerankerCollator(tokenizer, args.max_length)
    metrics = evaluate_split(
        model,
        tokenizer,
        collator,
        dev_qids,
        questions,
        qrels,
        corpus,
        rank_by_qid,
        args.instruction,
        token_true_id,
        token_false_id,
        batch_size=args.eval_batch_size,
        device=args.device,
        output_dir=args.output_dir,
        name="dev",
    )
    with open(os.path.join(args.output_dir, "epoch_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    save_best_if_needed(model, tokenizer, args.output_dir, args.best_dir)


def train(args: argparse.Namespace) -> None:
    import torch
    from torch.nn import BCEWithLogitsLoss
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    start = time.perf_counter()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("loading data")
    questions, qrels = read_queries(args.queries)
    corpus = read_corpus(args.corpus)
    train_qids = [qid for qid in read_split_ids(args.train_split) if qid in questions]
    dev_qids = [qid for qid in read_split_ids(args.dev_split) if qid in questions]
    rank_by_qid = read_rank_tsv(args.rank_tsv, args.candidate_top_k)
    print(
        f"queries={len(questions)} corpus={len(corpus)} train_queries={len(train_qids)} "
        f"dev_queries={len(dev_qids)} candidate_queries={len(rank_by_qid)}"
    )

    def epoch_data_seed(epoch: int) -> int:
        if args.resample_random_negatives_each_epoch or args.resample_hard_negatives_each_epoch:
            return args.seed + epoch - 1
        return args.seed

    def make_epoch_examples(epoch: int) -> list[PairExample]:
        epoch_examples = build_train_examples(
            train_qids,
            qrels,
            rank_by_qid,
            corpus,
            hard_negatives=args.hard_negatives,
            hard_negative_pool_k=args.hard_negative_pool_k,
            resample_hard_negatives=args.resample_hard_negatives_each_epoch,
            random_negatives=args.random_negatives,
            seed=epoch_data_seed(epoch),
        )
        if args.limit_train_examples:
            epoch_examples = epoch_examples[: args.limit_train_examples]
            print(f"limited_train_examples={len(epoch_examples)}")
        return epoch_examples

    examples = make_epoch_examples(epoch=1)
    if args.resample_random_negatives_each_epoch and args.random_negatives > 0:
        print("random_negatives_resampled_each_epoch=True")
    if args.resample_hard_negatives_each_epoch and args.hard_negative_pool_k:
        print("hard_negatives_resampled_each_epoch=True")

    model, tokenizer, token_true_id, token_false_id = load_model_and_tokenizer(args)
    collator = RerankerCollator(tokenizer, args.max_length)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_batches_per_epoch = math.ceil(len(examples) / args.train_batch_size)
    update_steps_per_epoch = math.ceil(train_batches_per_epoch / args.gradient_accumulation_steps)
    total_steps = max(1, update_steps_per_epoch * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    positives = sum(1 for example in examples if example.label == 1.0)
    negatives = max(1, len(examples) - positives)
    pos_weight = torch.tensor([negatives / max(1, positives)], device=args.device)
    criterion = BCEWithLogitsLoss(pos_weight=pos_weight if args.use_pos_weight else None)
    print(
        f"train_batches_per_epoch={train_batches_per_epoch} total_steps={total_steps} warmup_steps={warmup_steps} "
        f"pos_weight={pos_weight.item():.4f} use_pos_weight={args.use_pos_weight}"
    )

    best_mrr = load_existing_best_mrr(args.output_dir, args.start_epoch)
    if best_mrr >= 0:
        print(f"loaded_existing_best_mrr10={best_mrr:.4f}")
    best_dir = os.path.join(args.output_dir, "best")
    os.makedirs(args.output_dir, exist_ok=True)
    config_name = "train_config.json" if args.start_epoch == 1 else f"train_config.resume_epoch_{args.start_epoch}.json"
    with open(os.path.join(args.output_dir, config_name), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    global_step = 0
    active_eval: tuple[subprocess.Popen, str] | None = None
    async_eval = bool(args.async_eval and args.eval_each_epoch and args.eval_device and args.eval_device != args.device)
    if async_eval:
        print(f"async_eval=True train_device={args.device} eval_device={args.eval_device}")
        if args.adapter and args.start_epoch > 1:
            prior_metrics_path = os.path.join(args.adapter, "epoch_metrics.json")
            if not os.path.exists(prior_metrics_path):
                print(f"launching_resume_adapter_eval={args.adapter}")
                active_eval = launch_epoch_eval(args, args.adapter, active_eval)
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        epoch_examples = examples if epoch == 1 else make_epoch_examples(epoch)
        train_dataset = PairDataset(epoch_examples, questions, corpus, args.instruction)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=args.num_workers,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        active_steps = 0
        resume_step = args.resume_step if epoch == args.start_epoch else 0
        in_progress_save_steps = build_in_progress_save_steps(
            len(train_loader),
            args.gradient_accumulation_steps,
            args.in_progress_save_fraction,
        )
        if resume_step:
            print(f"resuming_epoch={epoch} skipping_batches={resume_step}")
        progress = tqdm(train_loader, desc=f"training epoch {epoch}")
        for step, batch in enumerate(progress, start=1):
            if step <= resume_step:
                continue

            batch = {key: value.to(args.device) for key, value in batch.items()}
            scores, labels = compute_pair_logits(model, batch, token_true_id, token_false_id)
            loss = criterion(scores.float(), labels.float())
            (loss / args.gradient_accumulation_steps).backward()
            running_loss += loss.item()
            active_steps += 1

            if step % args.gradient_accumulation_steps == 0 or step == len(train_loader):
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if args.log_every > 0 and step % args.log_every == 0:
                avg_loss = running_loss / max(1, active_steps)
                progress.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            if step in in_progress_save_steps:
                save_in_progress_adapter(
                    model,
                    tokenizer,
                    args.output_dir,
                    epoch,
                    step,
                    len(train_loader),
                    global_step,
                    args,
                )

        epoch_dir = os.path.join(args.output_dir, f"epoch-{epoch}")
        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)
        print(f"saved_adapter={epoch_dir}")

        metrics = {}
        if args.eval_each_epoch and dev_qids:
            if async_eval:
                active_eval = launch_epoch_eval(args, epoch_dir, active_eval)
            else:
                metrics = evaluate_epoch(
                    args,
                    model,
                    tokenizer,
                    collator,
                    epoch_dir,
                    dev_qids,
                    questions,
                    qrels,
                    corpus,
                    rank_by_qid,
                    token_true_id,
                    token_false_id,
                )
                mrr = metrics.get("MRR@10", 0.0)
                if mrr > best_mrr:
                    best_mrr = mrr
                    model.save_pretrained(best_dir)
                    tokenizer.save_pretrained(best_dir)
                    print(f"saved_best_adapter={best_dir} best_mrr10={best_mrr:.4f}")
                with open(os.path.join(epoch_dir, "epoch_metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(metrics, f, ensure_ascii=False, indent=2)
        else:
            with open(os.path.join(epoch_dir, "epoch_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)

    if active_eval is not None:
        process, epoch_dir = active_eval
        print(f"waiting_final_eval={epoch_dir}")
        wait_for_eval_process(process, epoch_dir)

    print(f"total_seconds={time.perf_counter() - start:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default=default_queries_path())
    parser.add_argument("--corpus", default=default_corpus_path())
    parser.add_argument("--train-split", default=default_train_split_path())
    parser.add_argument("--dev-split", default=default_dev_split_path())
    parser.add_argument("--rank-tsv", default=default_rank_path())
    parser.add_argument("--output-dir", default="outputs/qwen3-reranker-0.6b-lora-stard")
    parser.add_argument("--model", default="Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--model-source", choices=["hf", "modelscope"], default="modelscope")
    parser.add_argument("--model-cache", default=None)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--best-dir", default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--candidate-top-k", type=int, default=500)
    parser.add_argument("--hard-negative-pool-k", type=int, default=None)
    parser.add_argument("--hard-negatives", type=int, default=30)
    parser.add_argument("--random-negatives", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=14)
    parser.add_argument("--in-progress-save-fraction", type=float, default=0.25)
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-device", default=None)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-pos-weight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-each-epoch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--async-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resample-hard-negatives-each-epoch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resample-random-negatives-each-epoch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit-train-examples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.queries = normalize_path(args.queries)
    args.corpus = normalize_path(args.corpus)
    args.train_split = normalize_path(args.train_split)
    args.dev_split = normalize_path(args.dev_split)
    args.rank_tsv = normalize_path(args.rank_tsv)
    args.output_dir = normalize_path(args.output_dir)
    args.model_cache = normalize_path(args.model_cache) if args.model_cache else None
    args.adapter = normalize_path(args.adapter) if args.adapter else None
    args.best_dir = normalize_path(args.best_dir) if args.best_dir else None
    if args.candidate_top_k < 1:
        raise ValueError("--candidate-top-k must be >= 1")
    if args.hard_negative_pool_k is not None and args.hard_negative_pool_k < 1:
        raise ValueError("--hard-negative-pool-k must be >= 1")
    if args.hard_negative_pool_k is not None and args.hard_negative_pool_k > args.candidate_top_k:
        raise ValueError("--hard-negative-pool-k must be <= --candidate-top-k")
    if args.hard_negatives < 0 or args.random_negatives < 0:
        raise ValueError("negative counts must be >= 0")
    if args.train_batch_size < 1 or args.eval_batch_size < 1:
        raise ValueError("batch sizes must be >= 1")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be >= 1")
    if args.in_progress_save_fraction < 0 or args.in_progress_save_fraction >= 1:
        raise ValueError("--in-progress-save-fraction must be >= 0 and < 1")
    if args.resume_step < 0:
        raise ValueError("--resume-step must be >= 0")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.start_epoch < 1:
        raise ValueError("--start-epoch must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    if args.eval_only:
        eval_only(args)
        return
    train(args)


if __name__ == "__main__":
    main()
