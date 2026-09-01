# Pip Guardiola

Look up a footballer-season and rank who *plays* like them — not who scored the same number of goals.

**Live: [pipguardiola.com](https://pipguardiola.com)**

Canonical example: **Kevin De Bruyne, 2019-20**. Filter by era, league, position, and minutes. Named playing styles (wide creator, destroyer, poacher, …) are a label overlay, not how retrieval works.

## How it works

Outfield seasons are z-scored rates, style shares, and possession-adjusted defending (FBref Big 5, 2017-18–2024-25, 900+ minutes). A small JAX encoder (InfoNCE on Poisson resamples of the same season) maps that vector to a 32-d metric. Neighbours are cosine in that space. Keepers stay on PCA-whitened cosine — the learned metric lost there.

Clustering does not retrieve. A 16-style prototype match names the page (deep-lying playmaker, overlapping full-back, sweeper-keeper, …). Feature-group cosine explains why two seasons lined up.

## Numbers

Adjacent-season retrieval on the outfield index (same player, next season, held out of training):

| Metric | MRR | Recall@10 | Median rank | Position purity@10 |
|---|---|---|---|---|
| Z-scored cosine | 0.159 | 0.284 | 57 | 0.869 |
| PCA-whitened cosine | 0.135 | 0.247 | 83 | 0.805 |
| **Learned (InfoNCE)** | **0.197** | **0.358** | **30** | 0.864 |

Keepers: PCA-whitened cosine MRR 0.104 vs 0.082 learned. De Bruyne 2019-20’s nearest Premier League neighbours under the learned metric are Bruno Fernandes (several seasons) and Christian Eriksen 2017-18, driven by creation.

## Run it locally

Python 3.12, [uv](https://docs.astral.sh/uv/), Node 22. Artifacts are committed, so you do not need to re-download FBref.

```bash
uv sync --all-extras
uv run kindred-api             # http://127.0.0.1:8317
```

```bash
cd web
npm install
npm run dev                    # http://127.0.0.1:43917
```

Rebuild features/model only if you are changing the pipeline: `kindred-ingest` → `kindred-features` → `kindred-train` → `kindred-eval` → `kindred-cluster`. Do not scrape FBref from a datacenter IP.
