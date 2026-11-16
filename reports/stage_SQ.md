# Stage SQ report: scalar quantization for the sub-raw memory regime

## Provenance

- Date: 2026-11-05T00:30:30+00:00
- Git commit: `31a41c051e320d4badda0af45b2b6f50415fe8bb-dirty`
- Hardware id: `1d3d9b66e7d3e1ac` (AMD EPYC-Milan Processor, 16 cores, 58.8 GB)
- Source: `results/tuned/tuned.parquet`

8-bit scalar-quantized HNSW (`hnsw_sq`) against plain HNSW at N=500,000, D in {384, 1536}. Both families were measured in the same run: absolute latency on this host drifts ~1.12x (median) between sessions, so a cross-campaign comparison would confound compression with drift.

## Memory footprint

| index_type   |    D |   index_MB |   index_x_raw |   rss_x_raw |
|:-------------|-----:|-----------:|--------------:|------------:|
| hnsw         |  384 |        840 |         1.094 |       1.096 |
| hnsw_sq      |  384 |        264 |         0.344 |       0.346 |
| hnsw         | 1536 |       3144 |         1.023 |       1.024 |
| hnsw_sq      | 1536 |        840 |         0.273 |       0.274 |

SQ8 stores the vectors at one byte per dimension instead of four, so resident memory falls from 1.02-1.10x raw to 0.27-0.35x. The graph links are unchanged and are D-independent, which is why compression improves with D: at D=1536 the vectors dominate and the ratio approaches the 0.25x floor, while at D=384 the fixed link overhead is proportionally larger.

## Memory-budget feasibility

| index | D | resident / raw | 1.5x | 0.5x | 0.1x |
|---|---|---|---|---|---|
| hnsw | 384 | 1.096x | yes | no | no |
| hnsw_sq | 384 | 0.346x | yes | yes | no |
| hnsw | 1536 | 1.024x | yes | no | no |
| hnsw_sq | 1536 | 0.274x | yes | yes | no |

**This is the result that changes the main finding.** In the main grid no index satisfied a sub-raw budget at R >= 0.90: IVF-PQ was small enough but capped at 0.888 recall, and everything else exceeded 1x raw. SQ8 fits the 0.5x budget AND reaches the recall targets, so the correct statement becomes **"a scalar-quantized graph index qualifies at 0.5x raw; nothing qualifies at 0.1x"** rather than "no index qualifies". The 0.1x budget needs ~10x compression and SQ8 gives ~4x.

## Cost at each recall target (test queries)

| index_type   |    D |    R | reachable   |   ef_search |   recall_test | meets_R_on_test   |   lat_p95_us |   build_seconds_mean |
|:-------------|-----:|-----:|:------------|------------:|--------------:|:------------------|-------------:|---------------------:|
| hnsw         |  384 | 0.9  | True        |          24 |      0.934562 | True              |      250.762 |              65.8151 |
| hnsw_sq      |  384 | 0.9  | True        |          24 |      0.932388 | True              |      229.52  |              55.2598 |
| hnsw         |  384 | 0.95 | True        |          32 |      0.952462 | True              |      294.824 |              65.8151 |
| hnsw_sq      |  384 | 0.95 | True        |          32 |      0.949225 | False             |      255.833 |              55.2598 |
| hnsw         |  384 | 0.99 | True        |         128 |      0.990788 | True              |      856.167 |              65.8151 |
| hnsw_sq      |  384 | 0.99 | True        |         256 |      0.989538 | False             |     1349.09  |              55.2598 |
| hnsw         | 1536 | 0.9  | True        |          16 |      0.911288 | True              |      392.077 |             196.796  |
| hnsw_sq      | 1536 | 0.9  | True        |          16 |      0.909413 | True              |      349.137 |             217.772  |
| hnsw         | 1536 | 0.95 | True        |          32 |      0.960363 | True              |      576.734 |             196.796  |
| hnsw_sq      | 1536 | 0.95 | True        |          32 |      0.956625 | True              |      518.403 |             217.772  |
| hnsw         | 1536 | 0.99 | True        |          64 |      0.990787 | True              |     1369.74  |             316.493  |
| hnsw_sq      | 1536 | 0.99 | True        |          96 |      0.990775 | True              |     1580.95  |             311.879  |

### Paired latency ratio

|    D |    R |   hnsw |   hnsw_sq |   sq_over_hnsw |
|-----:|-----:|-------:|----------:|---------------:|
|  384 | 0.9  |  250.8 |     229.5 |            0.9 |
|  384 | 0.95 |  294.8 |     255.8 |            0.9 |
|  384 | 0.99 |  856.2 |    1349.1 |            1.6 |
| 1536 | 0.9  |  392.1 |     349.1 |            0.9 |
| 1536 | 0.95 |  576.7 |     518.4 |            0.9 |
| 1536 | 0.99 | 1369.7 |    1580.9 |            1.2 |

SQ is faster than uncompressed HNSW in 4 of 6 (D, R) pairs, by up to 13%. Fewer bytes per vector means less memory traffic per distance evaluation, which outweighs the decode cost at the lower recall targets. At R=0.99 it reverses: SQ needs a higher efSearch to compensate for quantization error, and the extra traversal costs more than the bytes saved.

## Tuning-to-test gap

2 of 12 winners were selected on tuning queries but fall just short on the test queries (worst -0.0008). These rows must not be cited as having achieved R:

| index_type   |   D |    R |   recall_test |   recall_gap_vs_R |
|:-------------|----:|-----:|--------------:|------------------:|
| hnsw_sq      | 384 | 0.95 |      0.949225 |        -0.000775  |
| hnsw_sq      | 384 | 0.99 |      0.989538 |        -0.0004625 |

## Caveats

- Two cells only (N=500K, D in {384, 1536}); this is a targeted probe of the memory regime, not a full stage.
- Earlier `hnsw_sq` rows measured on 2026-10-30 and 2026-11-03 accumulated while a stage-wide re-measurement was running and are quarantined under `results/raw/builds_superseded_offcampaign/`.
- SQ8 is a 4x compression; the 0.1x budget would need a different technique (finer PQ with re-ranking, or 4-bit SQ) and is untested here.
