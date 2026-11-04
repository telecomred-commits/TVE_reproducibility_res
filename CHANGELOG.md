# Changelog

## v1.1.0 — revision 1 of manuscript robotics-4632217 (November 2026)

Code
* **Correction of TVE** (`src/tve/tve.py::TVEPlanner.prepare`). When the skeleton with one generator per
  obstacle reported defects, v1.0.0 kept it if the skeleton with contour generators reported the *same* number
  of defects; v1.1.0 always keeps the contour skeleton in that case, as the paper prescribes. The correction
  changes only the maps in which both skeletons have defects: 12 of the 455 maps of E1–E7
  (`results/raw/r1_skeleton_correction_maps.csv`).
* New baselines `mvdrm` and `g2vd` (`src/baselines/voronoi_roadmaps.py`), reimplementations of MVDRM
  (Ayawli et al., 2021) and G2VD (Wen et al., 2025) from their published descriptions.
* New `TVEConfig` fields, all with the values used in v1.0.0 as defaults: `penalty` (β = 1.7, previously
  hard-coded), `prune_bound` (True) and `n_validate` (4), the last two only for the diagnostic of the pruning.
* `src/baselines/visibility.py`: the offset of the vertices is a parameter (`desplazamiento_px`, default 2.0 px,
  the value of v1.0.0).

Experiments and analyses
* E8 (sensitivity of the parameters of TVE), E9 (narrow passages), E10 (MVDRM, G2VD, WDT and TVE on the queries
  of E1) and three revision checks: reference optimum (`r1_reference_check`), pruning of candidates
  (`r1_pruning_check`) and maps affected by the correction (`r1_skeleton_correction`).
* `statistics/analyze.py`: ε-optimality table, tables of E8–E10, Table A2 (the 12 re-executed maps) and the new
  numbers of the text. `statistics/check_raw.py`: integrity check of the raw files and SHA-256 checksums.
* Figures: final printed widths, 7-pt text, 2-m scale bars, new gallery layout, narrow-passage figure,
  Supplementary Figure S1 with the 84 maps of E1 (`figures/make_supplementary.py`).

Results
* `results/raw/e1.csv` … `e7.csv`: corrected results (the rows of the 12 affected map jobs were replaced in place;
  E4 did not change and keeps its original rows). `results/raw/original_submission/`: results of v1.0.0.
  `results/raw/revision1_reruns.json`: the re-executed jobs.
* `results/raw/e8.csv`, `e9.csv`, `e10.csv` and `r1_*.csv`: new.
* `results/verification_revision1.txt`: complete re-execution of E1–E10 from this package (geometric results
  identical except for floating-point rounding, relative differences < 10⁻⁵).

## v1.0.0 — first submission (September 2026)

Experiments E1–E7, fourteen baselines, raw results, tables and figures of the submitted manuscript.
