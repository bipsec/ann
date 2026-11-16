# Superseded Stage D rows (full-corpus ground truth)

These rows scored a partially-filled index against ground truth for the FULL 500,000-vector
corpus. Vectors not yet inserted therefore counted as misses, so recall climbed mechanically
with every batch (0.911 -> 0.9997 for HNSW) and the curve measured completeness rather than
drift, which is what Stage D is about.

Retained rather than deleted (EXPERIMENT_PLAN rule 0.6). The insert-versus-rebuild comparison
in these rows is still valid, because both sides were scored against the same ground truth at
the same corpus size.

Replaced by rows in `../update_runs/`, which score each batch against ground truth restricted
to the vectors the index actually contains, and additionally report the old quantity as
`recall_vs_full_corpus_gt`.
