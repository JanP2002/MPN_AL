#!/usr/bin/env python
"""Select 50 tiles from 5-10 clusters and compare the ways of doing it; see mpn_al.evaluation.quota_cli.

    python scripts/run_quota_selection.py --tiles-parquet prep/tiles.parquet --descriptors old/descriptors.parquet \
        --embeddings lunit-dino-s16=runs/lunit/embeddings.parquet --out outputs/selection/<id>_quota

Protocol: docs/architecture/quota-selection-experiment.md. Run with --help for all options.
"""

import _bootstrap  # noqa: F401  (makes `mpn_al` importable from a plain checkout)

from mpn_al.evaluation.quota_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
