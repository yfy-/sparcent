#!/usr/bin/env python3

"""Search an index with PLAID or Sparcent and report query latencies.

Each query is run --num-runs times and its minimum latency is kept. The latency
summary is printed and stored in the .meta file next to each ranking.
"""

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--index-name", required=True)
    parser.add_argument(
        "--queries", required=True, help="TSV queries: qid<TAB>query"
    )
    parser.add_argument("--root", default="experiments")
    parser.add_argument("--experiment", default="default")
    parser.add_argument("--run-name", default="search")
    parser.add_argument("-k", type=int, nargs="+", default=[10])
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument(
        "--threads", type=int, help="CPU threads (default: all cores)"
    )
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument(
        "--ncells",
        type=int,
        help="PLAID nprobe (default: 1, 2 and 4 for k <= 10, 100 and above)",
    )
    parser.add_argument(
        "--centroid-score-threshold",
        type=float,
        help="PLAID threshold (default: 0.5, 0.45 and 0.4 per k as above)",
    )
    parser.add_argument(
        "--ndocs",
        type=int,
        help="PLAID ndocs (default: 256, 1024 and max(4k, 4096) per k)",
    )

    sparcent = parser.add_argument_group("Sparcent")
    sparcent.add_argument(
        "--sparcent",
        action="store_true",
        help="Retrieve centroids from the index's Sparcent index",
    )
    sparcent.add_argument(
        "--ks", type=int, help="Centroids retrieved from the sparse index"
    )
    sparcent.add_argument(
        "--kr", type=int, help="Centroids re-ranked by dense scores"
    )
    sparcent.add_argument(
        "--alpha", type=float, default=1.0, help="BMP block pruning factor"
    )
    sparcent.add_argument(
        "--beta", type=float, default=1.0, help="BMP query terms ratio kept"
    )
    sparcent.add_argument(
        "--sparcent-index", help="Use this BMP index instead of the index's"
    )
    sparcent.add_argument(
        "--sparse-queries",
        help="Precomputed SPLADE queries ('qid: terms') instead of encoding",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # Thread pools and devices are fixed when torch is first imported.
    if args.threads is not None:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[var] = str(args.threads)

    if args.gpus == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from pathlib import Path

    import torch

    from colbert import Searcher
    from colbert.data import Queries
    from colbert.infra import ColBERTConfig, Run, RunConfig

    if args.threads is not None:
        torch.set_num_threads(args.threads)

    search_config = dict(
        ncells=args.ncells,
        centroid_score_threshold=args.centroid_score_threshold,
        ndocs=args.ndocs,
        sparcent=args.sparcent,
        sparcent_index=args.sparcent_index,
        sparcent_k_score=args.ks,
        sparcent_k_rerank=args.kr,
        sparcent_alpha=args.alpha,
        sparcent_beta=args.beta,
    )
    search_config = {k: v for k, v in search_config.items() if v is not None}
    sparse_path = Path(args.sparse_queries) if args.sparse_queries else None
    run_config = RunConfig(
        nranks=1,
        root=os.path.abspath(args.root),
        experiment=args.experiment,
        name=args.run_name,
        overwrite=True,
        gpus=args.gpus,
    )
    # A fresh searcher per k: PLAID's per-k defaults are set on first search.
    for k in args.k:
        with Run().context(run_config):
            queries = Queries(path=args.queries, sparse_path=sparse_path)
            searcher = Searcher(
                index=args.index_name, config=ColBERTConfig(**search_config)
            )
            ranking = searcher.search_all_timed(
                queries, k=k, num_run=args.num_runs
            )
            ranking.save(f"{args.index_name}.k={k}.ranking.tsv")


if __name__ == "__main__":
    main()
