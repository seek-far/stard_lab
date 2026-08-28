# Training Data Sources

Original dataset:

- Repository: https://github.com/oneal2000/STARD
- License: MIT, as stated by the upstream STARD repository.

This project does not copy the original STARD dataset into the repo. Keep STARD as a sibling
checkout:

```bash
git clone https://github.com/oneal2000/STARD ../STARD
```

The trainer resolves data paths in this order:

- `data/queries.json`, otherwise `../STARD/data/queries.json`
- `data/corpus.jsonl`, otherwise `../STARD/data/corpus.jsonl`
- `data/example/train.query.txt`, otherwise `../STARD/data/example/train.query.txt`
- `data/example/dev.query.txt`, otherwise `../STARD/data/example/dev.query.txt`

Training data construction:

- Train split: query IDs from `data/example/train.query.txt`.
- Dev split: query IDs from `data/example/dev.query.txt`.
- Positive labels: `match_id` values in `queries.json`.
- Hard negatives: non-gold documents from `outputs/qwen3-embedding-4b-ms-bs16/rank.tsv`.
- Random negatives: sampled from `corpus.jsonl`.
- Dev rerank candidates: top candidates from `outputs/qwen3-embedding-4b-ms-bs16/rank.tsv`.

The included first-stage candidate file was produced by Qwen3-Embedding-4B and is used to make
reranker training reproducible without recomputing dense retrieval:

```text
outputs/qwen3-embedding-4b-ms-bs16/rank.tsv
outputs/qwen3-embedding-4b-ms-bs16/metrics.json
```

First-stage Qwen3-Embedding-4B metrics:

| Metric | Value |
| --- | ---: |
| MRR@10 | 0.501038 |
| Recall@10 | 0.603621 |
| Recall@50 | 0.773001 |
| Recall@100 | 0.836503 |
| Recall@500 | 0.926399 |
