# Stage D report: updates

## Provenance

- Date: 2026-11-17T00:12:04+00:00
- Git commit: `1980cef6a1d63ed03322b38ca8ba5fa388306854-dirty`
- Data hardware id: `1d3d9b66e7d3e1ac` (AMD EPYC-Milan Processor, 16 cores, 58.8 GB)
- Generated on: `5a5f956b01cd41d6` (aarch64; not the data host)
- Source: `results/raw/update_runs`

33 rows across 3 index types (hnsw, ivf_flat, ivf_pq) at N=500,000.

## How recall is measured here

Recall is scored against ground truth restricted to the vectors the index actually contains. Scoring a partially-filled index against full-corpus ground truth counts the not-yet-inserted vectors as misses, so recall rises mechanically with every batch and measures completeness rather than drift. The old quantity is retained per row as `recall_vs_full_corpus_gt`, and the superseded rows are kept under `results/raw/update_runs_superseded_fullcorpus_gt/`.

## Recall drift and the rebuild reference

| index_type   |   batches |   recall_first |   recall_last |    drift |   rebuild_reference |   insert_vs_rebuild |   inserts_per_s_median |   p95_us_median |
|:-------------|----------:|---------------:|--------------:|---------:|--------------------:|--------------------:|-----------------------:|----------------:|
| hnsw         |        10 |        0.9998  |       0.99971 | -9e-05   |             0.99972 |            -1e-05   |                   1571 |           10890 |
| ivf_flat     |        10 |        0.99246 |       0.99256 |  0.0001  |             0.99228 |             0.00029 |                  56113 |           26699 |
| ivf_pq       |        10 |        0.71321 |       0.71356 |  0.00035 |             0.71245 |             0.00111 |                  58576 |           11465 |

Drift is negligible for every family (largest absolute change 0.00035), and each index ends within 0.00111 of a from-scratch rebuild on the same vectors. The asymmetry is ingest throughput: hnsw 1,571 vec/s, ivf_flat 56,113 vec/s, ivf_pq 58,576 vec/s.

## Per-batch detail

| index_type   |   batch_id |   inserted_total |   recall_at_fixed_params |   recall_vs_full_corpus_gt |   lat_p95_us |   inserts_per_second | retrained   |   rebuild_reference_recall |
|:-------------|-----------:|-----------------:|-------------------------:|---------------------------:|-------------:|---------------------:|:------------|---------------------------:|
| hnsw         |          0 |           455000 |                 0.9998   |                   0.910925 |     10938.8  |              867.028 | False       |                 nan        |
| hnsw         |          1 |           460000 |                 0.9998   |                   0.920938 |     10827    |             1567.55  | False       |                 nan        |
| hnsw         |          2 |           465000 |                 0.9998   |                   0.930425 |     10842.2  |             1503.17  | False       |                 nan        |
| hnsw         |          3 |           470000 |                 0.9998   |                   0.940013 |     10732.8  |             1479.72  | False       |                 nan        |
| hnsw         |          4 |           475000 |                 0.9998   |                   0.950263 |     10783.4  |             1576.32  | False       |                 nan        |
| hnsw         |          5 |           480000 |                 0.999787 |                   0.959588 |      9993.59 |             1463.06  | False       |                 nan        |
| hnsw         |          6 |           485000 |                 0.999762 |                   0.970175 |     11015.6  |             1637.41  | False       |                 nan        |
| hnsw         |          7 |           490000 |                 0.999762 |                   0.980663 |     11181.1  |             1645.84  | False       |                 nan        |
| hnsw         |          8 |           495000 |                 0.999737 |                   0.99     |     11319.3  |             1658.2   | False       |                 nan        |
| hnsw         |          9 |           500000 |                 0.999713 |                   0.999713 |     11642.2  |             1855.36  | False       |                 nan        |
| hnsw         |         10 |           500000 |                 0.999725 |                   0.999725 |         0    |                0     | True        |                   0.999725 |
| ivf_flat     |          0 |           455000 |                 0.992463 |                   0.904463 |     25318.8  |            90333.1   | False       |                 nan        |
| ivf_flat     |          1 |           460000 |                 0.99245  |                   0.914388 |     25355.7  |            55851.8   | False       |                 nan        |
| ivf_flat     |          2 |           465000 |                 0.992463 |                   0.923838 |     25788.5  |            56376.1   | False       |                 nan        |
| ivf_flat     |          3 |           470000 |                 0.992487 |                   0.933363 |     25618.6  |            46572.3   | False       |                 nan        |
| ivf_flat     |          4 |           475000 |                 0.992487 |                   0.9435   |     26712.4  |            59101.3   | False       |                 nan        |
| ivf_flat     |          5 |           480000 |                 0.992463 |                   0.95275  |     26979.2  |            51860.9   | False       |                 nan        |
| ivf_flat     |          6 |           485000 |                 0.992413 |                   0.963213 |     26686.3  |            47961.4   | False       |                 nan        |
| ivf_flat     |          7 |           490000 |                 0.99245  |                   0.973625 |     27139    |            59755.4   | False       |                 nan        |
| ivf_flat     |          8 |           495000 |                 0.992463 |                   0.982875 |     27374.6  |            64346.5   | False       |                 nan        |
| ivf_flat     |          9 |           500000 |                 0.992563 |                   0.992563 |     27349.7  |            49009.2   | False       |                 nan        |
| ivf_flat     |         10 |           500000 |                 0.992275 |                   0.992275 |         0    |                0     | True        |                   0.992275 |
| ivf_pq       |          0 |           455000 |                 0.713213 |                   0.676212 |     11352.9  |            78086     | False       |                 nan        |
| ivf_pq       |          1 |           460000 |                 0.713138 |                   0.680762 |     11736    |            70981     | False       |                 nan        |
| ivf_pq       |          2 |           465000 |                 0.713325 |                   0.685013 |     11896.2  |            58110.5   | False       |                 nan        |
| ivf_pq       |          3 |           470000 |                 0.71345  |                   0.689163 |     11439.8  |            55009.8   | False       |                 nan        |
| ivf_pq       |          4 |           475000 |                 0.71375  |                   0.693425 |     11362.3  |            64622     | False       |                 nan        |
| ivf_pq       |          5 |           480000 |                 0.7136   |                   0.697175 |     11120.3  |            56167.1   | False       |                 nan        |
| ivf_pq       |          6 |           485000 |                 0.71335  |                   0.701425 |     11279.1  |            58223.1   | False       |                 nan        |
| ivf_pq       |          7 |           490000 |                 0.712725 |                   0.70545  |     11490.3  |            61508.3   | False       |                 nan        |
| ivf_pq       |          8 |           495000 |                 0.713375 |                   0.709738 |     12165.8  |            58930.6   | False       |                 nan        |
| ivf_pq       |          9 |           500000 |                 0.713562 |                   0.713562 |     12169.6  |            52480.9   | False       |                 nan        |
| ivf_pq       |         10 |           500000 |                 0.71245  |                   0.71245  |         0    |                0     | True        |                   0.71245  |

## Caveats

- Deletions were not tested (excluded by the plan).
- The insert ratio is modest: 50,000 added to 450,000, about 11%. A system growing by multiples of its original corpus should not read this as licence never to rebuild.
- IVF-Flat's pre-tune fell back to maximum effort (`nprobe=512`) because it missed R>=0.95 on the tuning queries at the reduced corpus size; its measured recall against prefix ground truth is well above target.
