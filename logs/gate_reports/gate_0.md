# GATE 0 Report: Positioning Decision

## What ran
Phase 0 literature search across 13 papers using 4 targeted search queries on arxiv.

## What passed
- Related work table filled with 13 papers (exceeds the 10-paper minimum), each with a link and one-line verdict.
- All 4 search terms explored; no hits for controlled D × S studies.

## Key finding
**No existing paper measures how the filter-selectivity crossover changes with dimension on a single corpus.** The D × S headline is novel.

The closest prior work is "Filtered ANN as a Phase Transition" (Mandarapu & Kunkunuru, 2026) which models selectivity crossovers but uses fixed-D data. Dimension does not appear in their crossover formulas.

## Decision required

Keep the **D × S headline** (how the filter-selectivity crossover moves with dimensionality), or shift the headline to the **recall-constrained framing + decision rule**?

**Recommendation:** Keep the D × S headline. The literature search confirms it is a genuine gap with no prior work. The recall-constrained framing and decision rule remain strong secondary contributions. The Phase Transition paper's formulas at fixed D provide a direct comparison point — our study tests whether their D-independent prediction holds.

## Options

1. **Keep D × S headline** — "Which ANN Index Is Cheapest at a Given Recall? A Cost-Surface Study over Dimensionality, Scale, and Filter Selectivity." Novel gap confirmed; Phase Transition paper provides falsifiable baseline.

2. **Shift to recall-constrained + decision rule** — Lead with cost surfaces and the tested decision tree. Safer if the D × S interaction turns out to be weak (null result), but less distinctive.
