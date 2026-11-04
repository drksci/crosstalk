#!/usr/bin/env python
"""DLNN pipeline CLI.

  python run.py --task modrec --tiers V,E,C,R --cascade --llm none
  python run.py --task modrec --model models/tdlnet_modrec_ternary.npz     # the width-8 secondary model
  python run.py --task narma10 --llm none
"""
import argparse
import sys

from dlnn.db.load import DbUnavailable
from dlnn.llm.backend import BackendUnavailable
from dlnn.pipeline import ModelUnavailable, run_pipeline


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["narma10", "modrec"], default="narma10")
    ap.add_argument("--tiers", default="V,E,C,R")
    ap.add_argument("--cascade", action="store_true",
                    help="C writes params/circuit_extracted.json, R reads it through dlnn.cascade (needs tiers C and R)")
    ap.add_argument("--llm", default="none", help="none (template) | qwen | bitnet")
    ap.add_argument("--db", choices=["sqlite", "postgres"], default="sqlite")
    ap.add_argument("--pg-dsn", default=None, help="libpq DSN for --db postgres (default: $DLNN_PG_DSN)")
    ap.add_argument("--out", default="out")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--model", default=None,
                    help="modrec: ternary dlnn-model/1 file (default models/tdlnet_modrec_w32_ternary.npz)")
    ap.add_argument("--float-model", default=None, help="modrec: float baseline for the reference block "
                    "(default: the *_float.npz next to --model)")
    ap.add_argument("--cr-subset", type=int, default=None,
                    help="run tiers C and R on a fixed stratified subset of this many test examples "
                         "(0 = the full test set; default: pipeline.DEFAULT_CR_SUBSET)")
    ap.add_argument("--n-test", type=int, default=None, help="modrec: generate a smaller test set (multiple of 40; for tests)")
    a = ap.parse_args()
    kw = {}
    if a.cr_subset is not None:
        kw["cr_subset"] = a.cr_subset or None
    try:
        run_pipeline(a.task, a.tiers, a.cascade, a.llm, a.db, a.out, a.run_id, model=a.model,
                     float_model=a.float_model, n_test=a.n_test, pg_dsn=a.pg_dsn, **kw)
    except ModelUnavailable as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except (NotImplementedError, BackendUnavailable) as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    except DbUnavailable as e:
        print(f"error (run finished, database not loaded): {e}", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
