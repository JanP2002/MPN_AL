#!/usr/bin/env python
"""Build the two reports of the 50-tiles-from-clusters experiment from a finished run (see run_quota_selection.py).

    python scripts/build_quota_reports.py --results out/analysis --images prep/slides/<id>/tiles \
        --embeddings lunit-dino-s16=.../embeddings.parquet --embeddings <winner model>=... --out out

Writes ``<out>/raport_wyniki/`` (the experiment explained for an outside reader) and ``<out>/raport_wariant/`` (the best
variant in pictures and a blinded pack). Hand-written conclusions are read from ``--notes-results`` and
``--notes-variant`` (sections ``## key``). The PDF files need Edge or Chrome.
"""

import argparse
import logging
from pathlib import Path

import _bootstrap  # noqa: F401  (makes `mpn_al` importable from a plain checkout)
import numpy as np
from PIL import Image

from mpn_al.evaluation import quota_plates, quota_report
from mpn_al.evaluation.cluster_cli import parse_named
from mpn_al.evaluation.quota_analysis import RECOMMENDED_MODEL
from mpn_al.evaluation.quota_results import Run
from mpn_al.evaluation.selection_cli import aligned_vectors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True, help="the --out folder of run_quota_selection.py")
    parser.add_argument("--images", type=Path, required=True, help="directory with <tile_id>.png")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--embeddings", action="append", default=[], type=parse_named, metavar="NAME=FILE",
                        help="embeddings of the winner's model and of the reference model (needed for the second report)")
    parser.add_argument("--notes-results", type=Path)
    parser.add_argument("--notes-variant", type=Path)
    parser.add_argument("--only", choices=["results", "variant"])
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    def image_of(tile_id: str) -> np.ndarray:
        with Image.open(args.images / f"{tile_id}.png") as handle:
            return np.asarray(handle.convert("RGB"))

    if args.only != "variant":
        target = quota_report.build(args.results, image_of, args.out / "raport_wyniki", notes=args.notes_results, pdf=not args.no_pdf)
        logging.info("report 1: %s", target)
    if args.only != "results":
        run = Run(args.results)
        needed = list(dict.fromkeys([run.winner[0], RECOMMENDED_MODEL]))
        files = dict(args.embeddings)
        missing = [m for m in needed if m not in files]
        if missing:
            parser.error(f"give --embeddings for {', '.join(missing)}")
        ids = run.manifest["tile_id"].tolist()
        vectors = {m: aligned_vectors(Path(files[m]), ids) for m in needed}
        target = quota_plates.build(args.results, image_of, vectors, args.out / "raport_wariant", notes=args.notes_variant, pdf=not args.no_pdf)
        logging.info("report 2: %s", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
