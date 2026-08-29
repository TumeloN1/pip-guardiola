# Kindred

Soccer playstyle similarity. Look up a player-season — the canonical example is **Kevin De Bruyne, 2019-20** — and rank the footballers whose *style* is closest, not the ones who scored the same number of goals.

The dashboard filters by era, competition, position, and minutes. Weight sliders re-encode and re-rank live. A JAX contrastive encoder learns the metric; clustering is used only to label the space, not to retrieve.

## Run it

Python 3.12, [uv](https://docs.astral.sh/uv/), Node 22. No GPU — CPU JAX is enough.

```bash
uv sync --all-extras
uv run kindred-ingest          # downloads FBref release RDS into data/raw/
uv run kindred-features
uv run kindred-train
uv run kindred-eval
uv run kindred-cluster
uv run kindred-api             # http://127.0.0.1:8317
```

```bash
cd web
npm install
npm run dev                    # http://127.0.0.1:43917
```

Processed artifacts (`player_seasons.parquet`, features, encoder, eval) are committed, so the API and UI work without re-downloading FBref. Re-run ingest if you want a fresher mirror.

## What the numbers say

Adjacent-season retrieval on the outfield index (hold the same-player next-season pair out of training):

| Metric | MRR | Recall@10 | Median rank | Position purity@10 |
|---|---|---|---|---|
| Z-scored cosine | 0.159 | 0.284 | 57 | 0.869 |
| PCA-whitened cosine | 0.135 | 0.247 | 83 | 0.805 |
| **Learned (InfoNCE)** | **0.197** | **0.358** | **30** | 0.864 |

The encoder beats both baselines on the headline metric. Position purity is a sanity check — high, not 1.0, which is what you want.

Keepers are a 13-d space where PCA-whitened cosine still wins (MRR 0.104 vs 0.082 learned). The API serves PCA-whitened cosine for keepers and the learned metric for everyone else.

De Bruyne 2019-20's nearest neighbours under the learned metric, Premier League, 900+ minutes: Bruno Fernandes (several seasons) and Christian Eriksen 2017-18, explained by the creation group. That is the face-validity check.

## Data provenance

**Do not scrape FBref from a datacenter.** `fbref.com` sits behind Cloudflare and returns 403 to cloud IPs. The soccerdata / worldfootballR live scrapers work from a residential IP (your laptop), not from this environment.

**Use the GitHub release assets, not the in-repo `data/` folder** of [JaseZiv/worldfootballR_data](https://github.com/JaseZiv/worldfootballR_data). The git-tree copy of `fb_big5_advanced_season_stats` is stale (last commit 2022-11-02, ends mid 2022-23). Current files:

```
https://github.com/JaseZiv/worldfootballR_data/releases/download/fb_big5_advanced_season_stats/big5_player_{TABLE}.rds
```

Tables used: `standard`, `shooting`, `passing`, `passing_types`, `possession`, `defense`, `misc`, `keepers`, `keepers_adv`, plus `big5_team_possession` for possession-adjusted defending. `big5_player_gca.rds` is not in the releases (9-byte 404) — there is no SCA/GCA.

Gzip-compressed RDS v3, read with `pyreadr`. No R runtime required.

Usable window: **2017-18 through 2024-25** (Season_End_Year 2018–2025). Advanced passing/possession/defense tables gate it. 2025-26 is partially present in standard/shooting/passing and is excluded until possession and defense catch up. Big 5 leagues, not just England. The dashboard defaults to the Premier League with a competition toggle.

### Schema drift (the correctness trap)

Confirmed by per-season null rates, not guessed:

- Progressive passes in the *passing* table: `Prog` for 2018–2022, `PrgP` for 2023–2025. Coalesced. Canonical `prg_p` still comes from the standard table (`PrgP_Progression`), which is complete and matches the De Bruyne fixture (280).
- Expected assists: `xA` for 2018–2023, `xA_Expected` for 2024–2025. Coalesced.
- `xAG` in passing is null before 2023. Standard `xAG_Expected` is complete — we take it from there.
- Age is an integer until 2023-24, then `"32-337"` (years-days). Both forms are parsed.
- Ratio columns (`SoT_percent_Standard`, `Succ_percent_Take`, …) are 70–90% populated because they are null when the denominator is zero. All rates and shares are recomputed from counts.
- Pressures are gone. FBref removed them retroactively; do not look for `Press_Pressures`.
- Keepers advanced: `Att_Passes` through 2023, `Att (GK)_Passes` from 2024. Coalesced.

Join key: `(Url, Season_End_Year, Squad)` is unique. `Url` is the stable FBref player id — never join on name. Mid-season transfers produce one row per club; we aggregate to one player-season by summing counts and recomputing rates. Alexis Sánchez 2017-18 (Arsenal + Manchester Utd) is the fixture for that.

### Verification row

Kevin De Bruyne, 2019-20 (`Season_End_Year == 2020`), FBref standard table:

`Squad=Manchester City  Pos=MF  Age=28  Min=2791  Gls=13  Ast=20  xG=7.3  xAG=20.0  PrgP=280  PrgC=139  PrgR=257`

If ingest produces anything else for `e46012d4-2020`, something is wrong. `tests/test_ingest.py` asserts this.

## How similarity is computed

Clustering is the wrong primitive for retrieval. A cluster label is a coarse bucket. Kindred learns a metric.

**Features** (`src/kindred/features.py`), 900-minute floor, z-scored within `(season, competition)`:

1. Per-90 volumes (npxG, xAG, shots, key passes, progressive actions, tackles, …).
2. Style shares that do not care how much ball a player sees: touch distribution, tackle thirds, pass-length mix, progressive-pass share, distance per carry, take-ons per touch.
3. Possession-adjusted defending: `padj = raw_per90 * 50 / (100 - team_possession)`, using `big5_team_possession`. City defenders stop looking passive just because they hold the ball.

**Model** (`src/kindred/model.py`): Flax NNX MLP `43 → 128 → 64 → 32`, L2-normalized. InfoNCE positives are two Poisson resamples of the same player-season (counting stats are approximately Poisson). Consecutive-season pairs are held out of training so adjacent-season MRR is honest.

**Archetypes** (`src/kindred/cluster.py`): JIT diagonal GMM over the 32-d space, K chosen by BIC, named from the original z-features of each component's members. Soft memberships show up as bars on the player page. Gradient-based group scores explain *why* two players matched.

## Layout

```
src/kindred/ingest.py      release RDS → player_seasons.parquet
src/kindred/features.py    rates, shares, PAdj, z-scores
src/kindred/similarity.py  cosine / PCA baselines, filters
src/kindred/evaluate.py    adjacent-season MRR, face validity
src/kindred/model.py       Flax NNX encoder + InfoNCE
src/kindred/train.py       CLI
src/kindred/cluster.py     GMM + explanations
src/kindred/api.py         FastAPI on 8317
web/                       Next.js on 43917
artifacts/                 parquet, encoder, eval.json, GMM
```

## Limits

- No SCA/GCA. Understat `xGChain` / `xGBuildup` would partly substitute, but joining them needs fuzzy name matching and only covers 2019-20 onward, so they stay out of the core pipeline.
- 2024-25 possession/defense coverage is incomplete relative to the standard table; ~122 player-seasons with 900+ minutes drop out of the outfield index.
- FBref from a cloud VM will 403. Don't try.
- Keepers use PCA-whitened cosine, not the contrastive encoder.

## GitHub

This project lives at [TumeloN1/pip-guardiola](https://github.com/TumeloN1/pip-guardiola).
