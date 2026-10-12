#!/usr/bin/env python
"""The R2 evalset commands: stratified hand-check (A3) and oracle snapshot (E2).

A thin wrapper over `medical_rag.eval.evalset` -- `python -m medical_rag.eval.evalset`
is the identical command. Subcommands:

    .venv/bin/python scripts/evalset.py build   # pass 1: blind hand_check.md + CSV
    .venv/bin/python scripts/evalset.py score   # pass 2: per-stratum agreement table
    .venv/bin/python scripts/evalset.py export  # verdicts_<sha>.jsonl + snapshot_metadata.json
    .venv/bin/python scripts/evalset.py check   # snapshot == live current-sha cache slice?
"""

from medical_rag.eval.evalset import main

if __name__ == "__main__":
    raise SystemExit(main())