# Sparcent

Code for *Sparcent: faster multi-vector dense retrieval with sparse
centroid indices* by Erman Yafay and Ismail Sengor Altingovde
(Knowledge and Information Systems, 2026,
[doi:10.1007/s10115-026-02907-x](https://doi.org/10.1007/s10115-026-02907-x)).

Sparcent replaces PLAID's exhaustive query–centroid scoring with a
learned sparse retrieval over *sparse centroids*: each ColBERT
centroid is represented by the averaged, pruned SPLADE vectors of the
passages it contains, indexed with BMP. At search time the top `ks`
centroids are retrieved from this index and the first `kr` are
re-ranked with dense scores. On a single CPU core this is up to 4.6×
faster than PLAID and 1.3× faster than WARP with nearly unchanged
quality, for 0.2% extra memory (46 MB on MS MARCO).

## What is implemented where

Sparcent is implemented in Erman Yafay's research forks of ColBERT,
SPLADE and BMP, not in the upstream repositories. `environment.yml`
installs the colbert fork as a package with its `sparcent` extra,
which brings in the splade and BMP forks, pinned to the commits that
reproduce the paper's results. The PISA fork and ciff are command line
tools, included as submodules and built by `setup.sh`; only ciff is
used unmodified. Each fork's README details its changes and credits
the original authors.

- `colbert` ([fork of ColBERT](https://github.com/yfy-/colbert)):
  Sparcent index building in the `Indexer` (`colbert/sparcent/`:
  centroid passage ordering by count, max or mean similarity, and the
  PISA → CIFF → BMP index build) and Sparcent candidate generation in
  PLAID search (BMP retrieves `ks` centroids, dense scores re-rank the
  first `kr`), plus per-query latency measurement.
- `splade` ([fork of SPLADE](https://github.com/yfy-/splade)): the
  sparse centroid vectors (`splade/sparcent/`: averaged SPLADE vectors
  of each centroid's top passages, pruned to an L1 submass and
  quantized to 8 bits) and sparse query encoding.
- `bmp` ([fork of BMP](https://github.com/yfy-/BMP)): low-overhead
  per-query Python bindings that return NumPy arrays, threshold
  initialization for any `k`, and search options for latency
  measurement.
- `pisa/` ([fork of PISA](https://github.com/yfy-/pisa-score-cache))
  from the author's score caching work; Sparcent uses its standard
  tools to parse, invert and reorder (recursive graph bisection) the
  centroid documents.
- `ciff/` ([ciff](https://github.com/pisa-engine/ciff) 0.3.1,
  unmodified): exports the PISA index to CIFF for BMP.
- `scripts/`: `index.py` and `search.py`, the end-to-end indexing and
  search pipeline.

## Setup

Linux x86_64 with an NVIDIA GPU (driver supporting CUDA 11.7) and
`mamba`, `micromamba` or `conda`:

```bash
git clone --recursive git@github.com:yfy-/sparcent.git
cd sparcent
./setup.sh                 # creates the "sparcent" environment (SPARCENT_ENV=name to change)
mamba activate sparcent
```

`setup.sh` fetches the submodules, creates the environment from
`environment.yml` (Python 3.8, PyTorch 1.13.1 + CUDA 11.7, as used for
the paper's measurements, and the colbert fork with its `sparcent`
extra: the splade fork and the BMP Python bindings), and builds
`pisa2ciff` and the PISA tools.

## Usage

Build a ColBERT index together with its Sparcent index (stored in
`<index>/sparcent/`):

```bash
python scripts/index.py --collection collection.tsv --index-name msmarco.nbits=2 \
    --root experiments --experiment msmarco --checkpoint colbert-ir/colbertv2.0 \
    --nbits 2 --sparcent --sample-method count --nearest-percent 0.01 \
    --pruning-submass 0.05 --bp-depth 13
```

`--overwrite reuse` adds a Sparcent index to an existing ColBERT
index. `python scripts/index.py -h` lists all parameters (passage
ordering, SPLADE model and batching, PISA and BMP options). Expensive
intermediate results (passage order, SPLADE encodings, composed
centroid vectors) are kept in `<index>/sparcent/` and reused when
building again with other pruning parameters.

Search with Sparcent on one CPU thread; queries are encoded with
ColBERT and SPLADE before the timed loop, each query runs `--num-runs`
times and its minimum latency is kept:

```bash
python scripts/search.py --index-name msmarco.nbits=2 --root experiments --experiment msmarco \
    --queries queries.dev.small.tsv -k 10 100 1000 --threads 1 --gpus 0 --num-runs 3 \
    --sparcent --ks 5000 --kr 3000 --alpha 0.85 --beta 0.9
```

Without `--sparcent` the same index is searched with plain PLAID.
Latencies (mean, p50, p95, p99) are printed and saved in the `.meta`
file next to each ranking. Evaluate with:

```bash
python -m utility.evaluate.msmarco_passages --qrels qrels.dev.small.tsv \
    --ranking experiments/msmarco/scripts.search/search/msmarco.nbits=2.k=10.ranking.tsv
```

## Results

The paper's Table 3 compares Sparcent<sub>P</sub> (Sparcent with
PLAID) to PLAID on the 6,980 MS MARCO dev queries. Latencies are
per-query means in ms on an Intel Xeon w3-2435.

| k    | Method               | 1 core | 8 cores | MRR@10 | Recall@k |
|------|----------------------|--------|---------|--------|----------|
| 10   | PLAID                | 103.0  | 19.8    | 0.391  | 0.678    |
|      | Sparcent<sub>P</sub> | 29.8   | 9.1     | 0.388  | 0.670    |
| 100  | PLAID                | 137.6  | 28.3    | 0.394  | 0.906    |
|      | Sparcent<sub>P</sub> | 53.2   | 14.9    | 0.393  | 0.897    |
| 1000 | PLAID                | 206.6  | 47.3    | 0.394  | 0.976    |
|      | Sparcent<sub>P</sub> | 114.1  | 31.4    | 0.394  | 0.970    |

On LoTTE the single-core speed-up reaches 4.6× (Table 5). Running the
indexing and search commands above on the full MS MARCO collection
reproduces the single-core Sparcent<sub>P</sub> rows: MRR@10 and
Recall@k come out at most 0.005 higher and latencies at most 3% higher.

## Citation

```bibtex
@article{yafay2026sparcent,
  title   = {Sparcent: faster multi-vector dense retrieval with sparse centroid indices},
  author  = {Yafay, Erman and Altingovde, Ismail Sengor},
  journal = {Knowledge and Information Systems},
  volume  = {68},
  pages   = {280},
  year    = {2026},
  doi     = {10.1007/s10115-026-02907-x}
}
```

## Acknowledgements

Sparcent builds on
[ColBERT/PLAID](https://github.com/stanford-futuredata/ColBERT),
[SPLADE](https://github.com/naver/splade),
[BMP](https://github.com/pisa-engine/BMP),
[PISA](https://github.com/pisa-engine/pisa) and
[ciff](https://github.com/pisa-engine/ciff); all credit for these
projects goes to their authors.
