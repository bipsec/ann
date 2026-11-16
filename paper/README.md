# paper/

**The paper is [`main.md`](main.md)** — a single machine-generated document, fully populated with
measured results from all four stages (A–D). Do **not** hand-edit it: it is overwritten on every
regeneration.

## How it is produced
- `make paper` → `python -m annstudy.paper` writes `main.md` **and** the 17 tables under
  [`tables/`](tables/) (both `.md` and `.csv`), interpolating every number from files under
  `results/`. Then it runs a traceability check.
- `python -m annstudy.paper --check-only` verifies that every number in `main.md` traces to a
  result file (exits non-zero if not).
- Regenerate the summary CSVs the paper consumes with `make analysis` first if raw results changed;
  the Stage C/D narrative reports come from `make stage-reports`.

## Editing
Change wording or structure in **`src/annstudy/paper.py`** (function `write_paper`), never in
`main.md`. Figure placeholders are emitted by `fig_box()` as fenced boxes (see §7b of `main.md`);
render the actual PNGs with `make analysis`.

## Must run on the measurement host
`annstudy.paper` scopes result rows to the current machine's `hardware_id`. Regenerate on the Linux
measurement host (`hardware_id 1d3d9b66e7d3e1ac`) — running it on another machine rebuilds the paper
from that machine's rows and drops the authoritative data.

The earlier hand-written `01-…09.md` section scaffold has been removed; it was superseded by `main.md`.
