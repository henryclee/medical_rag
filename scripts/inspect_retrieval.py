#!/usr/bin/env python
"""Read a retrieval run back, or inspect live retrieval.

A thin wrapper over `medical_rag.eval.inspector` -- `python -m
medical_rag.eval.inspector` is the identical command.

A number in FINDINGS.md is not inspectable evidence unless the chunks behind it
are, which is what this is for:

    # offline: replay a stored run's grid with its cached verdicts, zero calls,
    # no index load (the chunk sidecar has the text)
    .venv/bin/python scripts/inspect_retrieval.py --from-run \
        outputs/exploration/retrieval_tuning/20260929T120215Z

    # one question at the terminal, one cell only
    .venv/bin/python scripts/inspect_retrieval.py --question-id 1312 \
        --print-chunk dense__orig:3

    # the interactive knob panel: pays ~4 s of loads once per session instead
    # of once per guess (each knob change is then one grid, ~10 s). Localhost only -- it renders full
    # corpus text and its Judge button spends endpoint time on the shared
    # endpoint; never repoint it at 0.0.0.0.
    .venv/bin/python scripts/inspect_retrieval.py --serve

`--serve --dry-run` prints the route table without loading anything, and
`--port 0` binds a free port so two hypotheses can sit in two browsers, each
with its own loaded index.
"""

from medical_rag.eval.inspector import main

if __name__ == "__main__":
    main()
