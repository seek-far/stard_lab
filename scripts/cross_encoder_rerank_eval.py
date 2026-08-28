import argparse
import inspect
import json
import os
import time
from collections import defaultdict
from collections.abc import Iterable
from glob import glob
from typing import Any

from tqdm import tqdm


RECALL_KS = [5, 10, 20, 30, 50, 100, 200, 300, 400, 500]
MRR_KS = [3, 5, 10]


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


def read_queries(path: str) -> tuple[list[tuple[str, str]], dict[str, set[str]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")

    queries = []
    qrels: dict[str, set[str]] = defaultdict(set)
    for obj in data:
        qid = str(obj["query_id"])
        question = obj.get("问题")
        if not isinstance(question, str) or not question.strip():
            continue
        queries.append((qid, question))
        for match_id in obj.get("match_id", []):
            qrels[qid].add(str(match_id))
    return queries, qrels


def read_corpus(path: str) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    by_id = {}
    name_to_id = {}
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
            by_id[doc_id] = {
                "id": doc_id,
                "name": name,
                "content": content.replace("\\n", "\n").strip(),
            }
            name_to_id.setdefault(name, doc_id)
    return by_id, name_to_id


def read_rank_tsv(
    path: str,
    corpus_by_id: dict[str, dict[str, str]],
    corpus_name_to_id: dict[str, str],
    top_k: int | None,
) -> tuple[dict[str, list[str]], int]:
    rows_by_qid: dict[str, list[tuple[int, str]]] = defaultdict(list)
    missing = 0
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
            qid, raw_match, raw_rank = parts[:3]
            match_id = raw_match if raw_match in corpus_by_id else corpus_name_to_id.get(raw_match)
            if match_id is None:
                missing += 1
                continue
            rows_by_qid[str(qid)].append((int(raw_rank), match_id))

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
    return output, missing


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


def resolve_rank_paths(input_root: str, rank_paths: list[str]) -> list[str]:
    if rank_paths:
        return [normalize_path(path) for path in rank_paths]
    pattern = os.path.join(normalize_path(input_root), "*", "rank.tsv")
    return sorted(glob(pattern))


def output_name_for_rank(input_root: str, rank_path: str) -> str:
    parent = os.path.basename(os.path.dirname(rank_path))
    root = os.path.basename(normalize_path(input_root))
    if parent and parent != root:
        return parent
    return os.path.splitext(os.path.basename(rank_path))[0]


def load_existing_predictions(path: str) -> dict[str, list[dict[str, Any]]]:
    if not os.path.exists(path):
        return {}
    output = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            output[str(obj["query_id"])] = obj["candidates"]
    return output


def write_jsonl_record(output, qid: str, candidates: list[dict[str, Any]]) -> None:
    output.write(json.dumps({"query_id": qid, "candidates": candidates}, ensure_ascii=False) + "\n")
    output.flush()


def format_query(question: str) -> str:
    return f"用户咨询问题：{question.strip()}"


def format_passage(article: dict[str, str]) -> str:
    return f"法条名称：{article['name']}\n法条内容：{article['content']}"


def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen = set()
    output = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def model_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    kwargs = {}
    if args.torch_dtype:
        import torch

        dtype_by_name = {
            "auto": "auto",
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        kwargs["torch_dtype"] = dtype_by_name[args.torch_dtype]
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    if args.device_map:
        kwargs["device_map"] = args.device_map
    return kwargs


def maybe_download_model(args: argparse.Namespace) -> str:
    model_name = args.model
    if os.path.exists(model_name):
        return normalize_path(model_name)

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    if args.model_cache:
        os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", normalize_path(args.model_cache))
        os.environ.setdefault("HF_HOME", normalize_path(args.model_cache))
        os.environ.setdefault("MODELSCOPE_CACHE", normalize_path(args.model_cache))

    if args.model_source == "modelscope":
        try:
            from modelscope import snapshot_download
        except ImportError as exc:
            raise RuntimeError("modelscope is required when --model-source modelscope") from exc
        return snapshot_download(model_name, cache_dir=normalize_path(args.model_cache) if args.model_cache else None)

    return model_name


def load_cross_encoder(args: argparse.Namespace):
    from sentence_transformers import CrossEncoder

    model_path = maybe_download_model(args)
    kwargs: dict[str, Any] = {
        "max_length": args.max_length,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.device:
        kwargs["device"] = args.device
    if args.model_cache:
        kwargs["cache_folder"] = normalize_path(args.model_cache)

    init_params = inspect.signature(CrossEncoder.__init__).parameters
    model_kwargs = model_kwargs_from_args(args)
    if "model_kwargs" in init_params:
        kwargs["model_kwargs"] = model_kwargs
    elif model_kwargs:
        print("warning: installed sentence-transformers does not support model_kwargs; dtype/device_map ignored")

    print(f"loading cross_encoder={model_path}")
    return CrossEncoder(model_path, **kwargs)


def score_query(
    model,
    question: str,
    candidate_ids: list[str],
    corpus_by_id: dict[str, dict[str, str]],
    batch_size: int,
) -> list[dict[str, Any]]:
    query = format_query(question)
    pairs = [(query, format_passage(corpus_by_id[doc_id])) for doc_id in candidate_ids]
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    scored = []
    for input_rank, (doc_id, score) in enumerate(zip(candidate_ids, scores), start=1):
        article = corpus_by_id[doc_id]
        scored.append(
            {
                "match_id": doc_id,
                "match_name": article["name"],
                "input_rank": input_rank,
                "score": float(score),
            }
        )
    scored.sort(key=lambda item: (-item["score"], item["input_rank"]))
    return scored


def write_rank_files(
    output_dir: str,
    queries: list[tuple[str, str]],
    predictions: dict[str, list[dict[str, Any]]],
    qrels: dict[str, set[str]],
) -> dict[str, float]:
    rankings = {}
    rank_path = os.path.join(output_dir, "rank.tsv")
    names_path = os.path.join(output_dir, "rank_names.tsv")
    with open(rank_path, "w", encoding="utf-8") as rank_out, open(
        names_path, "w", encoding="utf-8"
    ) as names_out:
        for qid, _question in queries:
            candidates = predictions.get(qid, [])
            rankings[qid] = [candidate["match_id"] for candidate in candidates]
            for rank, candidate in enumerate(candidates, start=1):
                score = candidate["score"]
                rank_out.write(f"{qid}\t{candidate['match_id']}\t{rank}\t{score:.8f}\n")
                names_out.write(f"{qid}\t{candidate['match_name']}\t{rank}\t{score:.8f}\n")

    metrics = evaluate(rankings, qrels, recall_ks=RECALL_KS, mrr_ks=MRR_KS)
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics


def rerank_one_source(
    args: argparse.Namespace,
    model,
    rank_path: str,
    queries: list[tuple[str, str]],
    qrels: dict[str, set[str]],
    corpus_by_id: dict[str, dict[str, str]],
    corpus_name_to_id: dict[str, str],
) -> None:
    source_name = output_name_for_rank(args.input_root, rank_path)
    output_dir = os.path.join(args.output_root, source_name)
    os.makedirs(output_dir, exist_ok=True)
    predictions_path = os.path.join(output_dir, "rerank_predictions.jsonl")

    candidate_by_qid, missing = read_rank_tsv(rank_path, corpus_by_id, corpus_name_to_id, args.top_k)
    predictions = load_existing_predictions(predictions_path) if args.resume else {}
    print(
        f"source={source_name} candidates_queries={len(candidate_by_qid)} "
        f"missing_rank_rows={missing} resume_done={len(predictions)}"
    )

    mode = "a" if args.resume else "w"
    query_by_qid = dict(queries)
    pending_qids = [
        qid
        for qid, _question in queries
        if qid in candidate_by_qid and qid not in predictions
    ]
    if args.limit_queries is not None:
        pending_qids = pending_qids[: args.limit_queries]

    with open(predictions_path, mode, encoding="utf-8") as out:
        for qid in tqdm(pending_qids, desc=f"reranking {source_name}"):
            candidates = score_query(
                model,
                query_by_qid[qid],
                candidate_by_qid[qid],
                corpus_by_id,
                batch_size=args.batch_size,
            )
            predictions[qid] = candidates
            write_jsonl_record(out, qid, candidates)

    metrics = write_rank_files(output_dir, queries, predictions, qrels)
    with open(os.path.join(output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_rank_tsv": rank_path,
                "model": args.model,
                "model_source": args.model_source,
                "top_k": args.top_k,
                "max_length": args.max_length,
                "batch_size": args.batch_size,
                "torch_dtype": args.torch_dtype,
                "device": args.device,
                "device_map": args.device_map,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"metrics source={source_name}")
    for key, value in metrics.items():
        print(f"{key}\t{value:.4f}")
    print(f"rank_path={os.path.join(output_dir, 'rank.tsv')}")
    print(f"metrics_path={os.path.join(output_dir, 'metrics.json')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="outputs")
    parser.add_argument("--rank-tsv", action="append", default=[])
    parser.add_argument("--queries", default=default_queries_path())
    parser.add_argument("--corpus", default=default_corpus_path())
    parser.add_argument("--output-root", default="outputs/qwen3-reranker-4b-cross-encoder")
    parser.add_argument("--model", default="Qwen/Qwen3-Reranker-4B")
    parser.add_argument("--model-source", choices=["hf", "modelscope"], default="hf")
    parser.add_argument("--model-cache", default=None)
    parser.add_argument("--hf-endpoint", default=None)
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--device-map", default=None)
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-queries", type=int, default=None)
    args = parser.parse_args()

    args.input_root = normalize_path(args.input_root)
    args.queries = normalize_path(args.queries)
    args.corpus = normalize_path(args.corpus)
    args.output_root = normalize_path(args.output_root)
    args.model_cache = normalize_path(args.model_cache) if args.model_cache else None
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.max_length < 1:
        raise ValueError("--max-length must be >= 1")
    if args.limit_queries is not None and args.limit_queries < 1:
        raise ValueError("--limit-queries must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    start = time.perf_counter()

    rank_paths = resolve_rank_paths(args.input_root, args.rank_tsv)
    if not rank_paths:
        raise FileNotFoundError(f"no rank.tsv found under {args.input_root}")

    print("loading data")
    queries, qrels = read_queries(args.queries)
    corpus_by_id, corpus_name_to_id = read_corpus(args.corpus)
    print(f"queries={len(queries)} corpus={len(corpus_by_id)} rank_files={len(rank_paths)}")
    for rank_path in rank_paths:
        print(f"rank_tsv={rank_path}")

    model = load_cross_encoder(args)
    for rank_path in rank_paths:
        rerank_one_source(
            args,
            model,
            rank_path,
            queries,
            qrels,
            corpus_by_id,
            corpus_name_to_id,
        )

    print(f"total_seconds={time.perf_counter() - start:.2f}")


if __name__ == "__main__":
    main()
