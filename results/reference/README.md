# Compact reference results

See [results documentation](../../docs/results.md) for computation definitions,
commands and provenance. `tables/` contains readable snapshots of the reproducible
paper tables. Source data and statistics are retained in the named analysis
subdirectories. `background/` contains explicitly unresolved source candidates;
it is excluded from paper result reproduction.

Run `python scripts/validate_reference.py` from the repository root to check
stored hashes and arithmetic agreement. No training or image analysis is rerun.
