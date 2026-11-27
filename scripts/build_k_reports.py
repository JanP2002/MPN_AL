#!/usr/bin/env python
"""Build the two reports of the K selection experiment from a finished grid search (see run_k_selection.py).

    python scripts/build_k_reports.py --results out/analysis --images prep/slides/<id>/tiles \
        --embeddings lunit-dino-s16=out/embeddings/lunit-dino-s16/embeddings.parquet ... --out out

Writes ``<out>/raport_k/`` (choosing K: tables and charts) and ``<out>/raport_wizualny/`` (plates of the two best
models and a blinded pack). Hand-written commentary is read from ``--notes-k`` and ``--notes-visual`` (sections ``## key``).
The PDF files need Edge or Chrome.
"""

import argparse
import logging
from pathlib import Path

import _bootstrap  # noqa: F401  (makes `mpn_al` importable from a plain checkout)
import numpy as np
import pandas as pd
from PIL import Image

from mpn_al.evaluation import k_report, plates_report
from mpn_al.evaluation.cluster_cli import parse_named
from mpn_al.evaluation.selection_cli import aligned_vectors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True, help="the --out folder of run_k_selection.py")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--images", type=Path, help="directory with <tile_id>.png (needed for the plates)")
    parser.add_argument("--embeddings", action="append", default=[], type=parse_named, metavar="NAME=FILE",
                        help="embeddings of at least the two best models (needed for the plates)")
    parser.add_argument("--notes-k", type=Path)
    parser.add_argument("--notes-visual", type=Path)
    parser.add_argument("--only", choices=["k", "visual"])
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    if args.only != "visual":
        target = k_report.build(args.results, args.out / "raport_k", notes=args.notes_k, pdf=not args.no_pdf)
        logging.info("report 1: %s", target)
    if args.only != "k":
        if args.images is None or not args.embeddings:
            parser.error("the visual report needs --images and --embeddings")
        top = k_report.Results(args.results).top_models(2)
        files = dict(args.embeddings)
        missing = [m for m in top if m not in files]
        if missing:
            parser.error(f"give --embeddings for {', '.join(missing)}")
        ids = pd.read_parquet(args.results / "manifest.parquet")["tile_id"].tolist()
        vectors = {m: aligned_vectors(Path(files[m]), ids) for m in top}

        def image_of(tile_id: str) -> np.ndarray:
            with Image.open(args.images / f"{tile_id}.png") as handle:
                return np.asarray(handle.convert("RGB"))

        target = plates_report.build(args.results, image_of, vectors, args.out / "raport_wizualny", notes=args.notes_visual, pdf=not args.no_pdf)
        logging.info("report 2: %s", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
