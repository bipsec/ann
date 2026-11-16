# Superseded: off-campaign re-measurements (2026-11-04)

95 test-split sweep rows produced when `--phase all` on `configs/stage_sq.yaml` re-measured the
winners of every earlier stage, not just its own two cells.

They are excluded because they were measured under a different machine occupancy than the
campaign that produced the reported numbers: 2 concurrent runners on a quiet box, against the
16-wide pinned pool used for Stages A and B. They disagree with the originals systematically
upward, by up to 1.43x (hnsw N=100K D=768: 308 -> 439 us). Section 4.2 of the paper documents
why such measurements are not comparable across campaigns.

Retained rather than deleted (EXPERIMENT_PLAN rule 0.6). The `hnsw_sq` rows from the same run
were KEPT in `../sweeps/`: they are that stage's own first measurement, not a re-measurement.

`results/tuned/tuned.parquet` was never refreshed with these rows, so no reported table was
affected.
