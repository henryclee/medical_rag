"""The repo root, derived once.

Modules that moved into the package used to compute this themselves as
``Path(__file__).parents[n]``, where ``n`` was correct for the directory they
were written in. A move keeps the code and changes the depth, so
``DEFAULT_JUDGE_CACHE`` -- derived from that root -- would have silently started
pointing inside ``src/medical_rag/`` and every run would have "found" an empty
cache, paying for its verdicts twice while reporting a reuse rate of zero.

Nothing else in the package should count parent directories.
"""

from pathlib import Path

# paths.py -> medical_rag -> src -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
