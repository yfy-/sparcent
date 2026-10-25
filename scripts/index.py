#!/usr/bin/env python3

"""Build a ColBERT (PLAID) index and, with --sparcent, its Sparcent index."""

import argparse
import os

from colbert import Indexer
from colbert.infra import ColBERTConfig, Run, RunConfig
from colbert.sparcent.sample_centroids import ORDER_METHODS


def parse_args():
    defaults = ColBERTConfig()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--collection", required=True, help="TSV collection: pid<TAB>passage"
    )
    parser.add_argument("--index-name", required=True)
    parser.add_argument("--checkpoint", default="colbert-ir/colbertv2.0")
    parser.add_argument("--root", default="experiments")
    parser.add_argument("--experiment", default="default")
    parser.add_argument("--nbits", type=int, default=2)
    parser.add_argument("--nranks", type=int, default=1)
    parser.add_argument(
        "--gpus", type=int, help="GPUs to use (default: all visible)"
    )
    parser.add_argument(
        "--overwrite",
        choices=["false", "true", "reuse", "resume"],
        default="false",
        help="'reuse' keeps an existing ColBERT index, e.g. to add Sparcent",
    )

    sparcent = parser.add_argument_group("Sparcent")
    sparcent.add_argument(
        "--sparcent",
        action="store_true",
        help="Also build the sparse centroid index",
    )
    sparcent.add_argument(
        "--sample-method",
        choices=ORDER_METHODS,
        default=defaults.sparcent_sample_method,
        help="Order of the passages of each centroid",
    )
    sparcent.add_argument(
        "--sample-device",
        choices=["cuda", "cpu"],
        default=defaults.sparcent_sample_device,
    )
    sparcent.add_argument(
        "--sample-jobs", type=int, default=defaults.sparcent_sample_jobs
    )
    sparcent.add_argument(
        "--splade-model", default=defaults.sparcent_splade_model
    )
    sparcent.add_argument("--splade-lora-dir")
    sparcent.add_argument(
        "--splade-batch-size",
        type=int,
        default=defaults.sparcent_splade_batch_size,
        help="Passages per saved SPLADE encoding batch",
    )
    sparcent.add_argument(
        "--splade-encode-batch-size",
        type=int,
        default=defaults.sparcent_splade_encode_batch_size,
    )
    sparcent.add_argument(
        "--nearest-percent",
        type=float,
        default=defaults.sparcent_nearest_percent,
        help="Ratio n of passages composing each centroid",
    )
    sparcent.add_argument(
        "--pruning-submass",
        type=float,
        default=defaults.sparcent_pruning_submass,
        help="L1 mass ratio alpha kept in each centroid vector",
    )
    sparcent.add_argument(
        "--docs-jobs",
        type=int,
        default=defaults.sparcent_docs_jobs,
        help="Processes writing the centroid documents",
    )
    sparcent.add_argument(
        "--bp-depth",
        type=int,
        default=defaults.sparcent_bp_depth,
        help="Recursive graph bisection depth",
    )
    sparcent.add_argument(
        "--pisa-bin-dir", help="Directory of the PISA tools (default: PATH)"
    )
    sparcent.add_argument(
        "--pisa-jobs", type=int, default=defaults.sparcent_pisa_jobs
    )
    sparcent.add_argument(
        "--pisa-batch-size",
        type=int,
        default=defaults.sparcent_pisa_batch_size,
    )
    sparcent.add_argument(
        "--bmp-block-size",
        type=int,
        default=defaults.sparcent_bmp_block_size,
    )
    sparcent.add_argument(
        "--no-bmp-compress-range",
        action="store_true",
        help="Do not compress the BMP block upper bounds",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_config = dict(
        nranks=args.nranks,
        root=os.path.abspath(args.root),
        experiment=args.experiment,
    )
    if args.gpus is not None:
        run_config["gpus"] = args.gpus

    overwrite = {"false": False, "true": True}.get(
        args.overwrite, args.overwrite
    )
    with Run().context(RunConfig(**run_config)):
        config = ColBERTConfig(
            nbits=args.nbits,
            sparcent=args.sparcent,
            sparcent_sample_method=args.sample_method,
            sparcent_sample_device=args.sample_device,
            sparcent_sample_jobs=args.sample_jobs,
            sparcent_splade_model=args.splade_model,
            sparcent_splade_lora_dir=args.splade_lora_dir,
            sparcent_splade_batch_size=args.splade_batch_size,
            sparcent_splade_encode_batch_size=args.splade_encode_batch_size,
            sparcent_nearest_percent=args.nearest_percent,
            sparcent_pruning_submass=args.pruning_submass,
            sparcent_docs_jobs=args.docs_jobs,
            sparcent_bp_depth=args.bp_depth,
            sparcent_pisa_bin_dir=args.pisa_bin_dir,
            sparcent_pisa_jobs=args.pisa_jobs,
            sparcent_pisa_batch_size=args.pisa_batch_size,
            sparcent_bmp_block_size=args.bmp_block_size,
            sparcent_bmp_compress_range=not args.no_bmp_compress_range,
        )
        indexer = Indexer(checkpoint=args.checkpoint, config=config)
        indexer.index(
            name=args.index_name,
            collection=args.collection,
            overwrite=overwrite,
        )


if __name__ == "__main__":
    main()
