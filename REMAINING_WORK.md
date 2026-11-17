# Remaining work — what is left to do

> **Status update 2026-11-16 (post-study audit; see `logs/decisions.md` for every item).**
> Done locally and committed on branch `study-complete`: repo moved to root and committed;
> Stage C measurement bugs fixed (cache keys, empty masks, timed region, thread leak, shared
> clustering, interleaving, canary, ndis counters); HNSW repeat builds now seeded (they were
> byte-identical); crossover re-analysed (H2: HNSW shift 4/4, IVF inconclusive 3/3) with a cost
> model predicting the direction in 6/7; scaling exponents with CIs (C1 done); Stage D pre-tune
> bug fixed; conservative lower-CI selection added; new Stage E (truncate vs quantize) built and
> smoke-tested; paper H2 text corrected. **Needs the VM (and your OK for > 2 h):**
> 1. `git pull` the branch on the VM, `uv sync --extra dev --extra hnswlib`, `make test`.
> 2. `make stage-e` (~1-1.5 h) and `make stage-c-h2` (~5-7 h; `CORES` = 8 cores).
> 3. `make analysis && make paper && python -m annstudy.paper --check-only`.
> 4. Re-run `make analysis` on the VM once anyway: `filter_proxy` (seed fix) only runs where
>    the Stage B base indexes live.
> Items below marked B1/B2/E2-E4/D1-D3 are unchanged; A1/A2/E1/F1-F3 are superseded by the
> git-based workflow above.


_Standalone checklist to conduct and consolidate later. Companion: [`FEEDBACK.md`](FEEDBACK.md)
(results + what went wrong). Status legend: ☐ not started · ◐ partial · ✅ done this pass._
_All measured work runs on the Linux VM (`hardware_id 1d3d9b66e7d3e1ac`); the generator must NOT be
run on the Windows machine — it scopes to the wrong host and rebuilds from stale local rows._

---

## A. Finalize & regenerate the paper (do first — on the VM)
Code edits in this pass (figure boxes in `paper.py`, Stage A/B report fix in `coordinate.py`) only
take effect after regenerating on the VM, then syncing back.

- ☐ **A1 · P0 · 5 min** — On the VM, pull the edited `src/` (see §E push step), then run:
  `make analysis && make stage-reports && make paper`, then `python -m annstudy.paper --check-only`
  (must exit 0 — no untraceable number). Then re-sync to the local machine.
- ☐ **A2 · P0 · 2 min** — Confirm `reports/stage_A.md` (N=500K only) and `reports/stage_B.md`
  (N∈{100K,500K,1M}) now differ (`diff` them).
- ☐ **A3 · P1 · later** — Generate the actual figures from the boxes now embedded in `paper/main.md`
  (`make analysis` already writes PNGs under `figures/`; the boxes name the source CSV + PNG for each).

## B. Measurement side-checks the plan lists but that were never run (deferred, optional)
Gate 5: "none changes the headline; all strengthen it." Run on the VM before submission if desired.

- ☐ **B1 · P2 · ~30 min** — **hnswlib HNSW cross-check** at one cell (N=500K, D=768). hnswlib 0.8.0 is
  installed. Confirms FAISS HNSW results aren't a library artifact. _NOTE: the paper must not claim
  this was done until it is — the old orphaned `paper/03-method.md` wrongly implied it; `paper/main.md`
  correctly lists it as future work._
- ☐ **B2 · P2 · ~1 h** — **k ∈ {1, 100} side check** at N=500K, D=768 (headline uses k=10). Needs a
  small run; the coordinator groups by cell so k≠10 needs its own sweep (see mismatch M1 note in
  `logs/decisions.md`).
- ☐ **B3 · P2 · a campaign** — **Drift canary:** a fixed canary config interleaved through any future
  campaign to measure between-session latency drift directly (the ~1.1× effect is currently
  uncharacterized). Gate 5's single recommended protocol addition before any follow-up run.

## C. Analysis gaps
- ☐ **C1 · P1 · ~1–2 h** — Add bootstrap **confidence-interval columns** to `scaling_fits.csv`
  (intervals on the b/c/d exponents) in `src/annstudy/analysis/scaling.py`; the plan (§5) asked for
  them and the paper currently reports point estimates only. Regenerate via `make analysis`.
- ☐ **C2 · P2 · 15 min** — Reconcile the **H1a/H1b/H1c hypothesis labels** between `reports/findings.md`
  (dimension/scale/budget) and the methodology doc's RQ1 wording, so the paper's §6 is internally
  consistent.

## D. Open Gate-5 human decisions (yours to make)
- ☐ **D1 · P1** — Keep **H1c** as a reported result? The winner changes with memory budget in only
  2 of 30 cells (localized). Decide: report as a minor finding, or fold into discussion.
- ☐ **D2 · P1** — Report the **correlated S=0.001** rows at all, given measured selectivity is 54–61%
  over target there? Options: keep with the measured-vs-nominal caveat, or drop to S ≥ 0.01.
- ☐ **D3 · P1** — Run the §B side-checks before submission, or cite them as future work? (Headline
  holds either way.)

## E. Release & write-up
- ☐ **E1 · P1 · ~1 h** — **Push the local code/doc edits to the VM.** The sync is pull-only; add/commit
  a push step (git, or an rsync the other direction). Needed so the `paper.py`/`coordinate.py` edits and
  the deleted orphaned files actually reach the authoritative VM before A1.
- ☐ **E2 · P1 · ~2 h** — Assemble the **final submission** from `paper/main.md` + the generated figures;
  confirm the target venue + page limit (methodology lists VLDB/SIGMOD/SIGIR/CIKM workshops or arXiv).
- ☐ **E3 · P2 · ~1 h** — Optional **LaTeX/PDF** export (`pandoc paper/main.md`) + a BibTeX references
  file (references are currently a markdown bullet list). No LaTeX scaffold exists yet.
- ☐ **E4 · P2** — **Archive a release** with a persistent identifier (DOI); confirm MS MARCO license
  terms allow releasing derived vectors (methodology release plan item 5).

## F. Repo hygiene (mostly closed this pass — must be mirrored on the VM via E1)
- ✅/☐ **F1** — Deleted orphaned `paper/01…09.md` + `paper/reproducibility.md`; `paper/README.md` now a
  pointer to `main.md`. (Local; re-do on VM or push, else the next pull restores them.)
- ✅/☐ **F2** — Fixed stale status tables in root `README.md`. (Local; push to VM.)
- ☐ **F3 · P2** — Decide the sync direction of truth: the VM is authoritative and the local sync is
  pull-only, so any local edit (code, deletions, READMEs) is reverted on the next pull unless pushed
  first. Establish git-on-both-ends or a push script to stop fighting the sync.

---

## Recommended order
1. **E1** (push local edits to VM) → **A1, A2** (regenerate paper + stage reports on VM, re-sync).
2. **D1–D3** (decide the three open claims) — they gate what the paper says.
3. **C1** (CI columns) and **C2** (label reconcile) — cheap, strengthen rigor.
4. **B1, B2** if you want the side-checks in the paper; otherwise cite as future work.
5. **E2 → E3 → E4** (assemble, export, release).
6. **B3** drift canary only if a follow-up campaign is planned.
