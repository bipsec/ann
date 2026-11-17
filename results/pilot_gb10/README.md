# GB10 pilot of the Stage E and H2 campaigns (2026-11-16/17)

Run on host `5a5f956b01cd41d6` (aarch64 GB10), which was simultaneously serving a 70B vLLM model;
the drift canary spread 828-1,138 us (1.37x). Stage E completed (15 arms x 3 R); the H2 rerun
stopped at 114/356 jobs (no error logged). Kept as a pilot, NOT the reported campaign: the
reported runs are on the VM `1d3d9b66e7d3e1ac`. Moved out of `results/raw/` so the VM's
resumable runners and the analysis never mix these rows with VM rows.

Headline pilot observations (to be confirmed on the VM): at R=0.95 the 384-d prefix reaches
only 0.74 recall vs the 1536-d top-10 at the same 1,808 B/vec where SQ8 @1536 reaches 0.996;
RaBitQ @1536 + SQ8 re-rank is the cheapest arm >= 2 KB/vec. H2: HNSW uncorrelated R=0.95,
S* ratio 0.37 [0.32, 0.42] from D=128 to 1536, log-log slope -0.41 [-0.47, -0.36].
