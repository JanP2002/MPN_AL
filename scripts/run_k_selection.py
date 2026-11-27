#!/usr/bin/env python
"""Grid search over the number of tiles per annotation round; see mpn_al.evaluation.k_cli.

    python scripts/run_k_selection.py --tiles-parquet prep/tiles.parquet --descriptors old/descriptors.parquet \
        --embeddings lunit-dino-s16=runs/lunit/embeddings.parquet --out outputs/selection/<id>_smallK

Protocol: docs/architecture/k-selection-experiment.md. Run with --help for all options.
"""

import _bootstrap  # noqa: F401  (makes `mpn_al` importable from a plain checkout)

from mpn_al.evaluation.k_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
