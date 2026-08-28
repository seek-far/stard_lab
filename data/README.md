# Data Sources

The original STARD dataset is maintained at:

https://github.com/oneal2000/STARD

The upstream repository is MIT licensed. This project does not vendor the original STARD dataset;
clone it next to this repository:

```bash
git clone https://github.com/oneal2000/STARD ../STARD
```

The training scripts use these files from `../STARD/data` when local `data/` files are absent:

- `queries.json`
- `corpus.jsonl`
- `example/train.query.txt`
- `example/dev.query.txt`
