#!/usr/bin/env python
"""Retrieve -> judge the union of candidates -> metrics -> FINDINGS.md.

A thin wrapper over `medical_rag.eval.harness`, which is a library module with
its own CLI -- `python -m medical_rag.eval.harness` is the identical command.
This file exists because the repo's convention is that runnable commands live in
`scripts/`: someone reading `ls scripts/` should see the study's commands
without first learning the package layout.

    # what a run would cost, before paying for it
    .venv/bin/python scripts/run_retrieval_grid.py --dry-run

    # rebuild FINDINGS.md from a run already on disk -- zero endpoint calls,
    # and the gate a refactor of this code has to pass: the table must come out
    # cell-for-cell identical to the committed one
    .venv/bin/python scripts/run_retrieval_grid.py \
        --rerender outputs/exploration/retrieval_tuning/20260929T120215Z

    # cache state (how much of it is usable under the current rubric)
    .venv/bin/python -m medical_rag.eval.judge --stats

Only the union of strategies this runs gets judged, so a new lever has to be
added to the grid to become measurable -- see `PLAN.md`'s constraints. For
scoring a strategy against verdicts that already exist, use
`scripts/eval_retrieval.py`, which loads no endpoint at all.
"""

from medical_rag.eval.harness import main

if __name__ == "__main__":
    main()
