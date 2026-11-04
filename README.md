# Reproducibility package — TVE: Voronoi tessellation skeleton and funnel path tightening

This package reconstructs **all the results** of the article (revised version, revision 1)

> O. Penagos, I. Ladino, C. R. Suárez. *Fast and Curvature-Aware Robot Navigation on Camera-Derived Maps
> Using a Voronoi Tessellation Skeleton and Funnel Path Tightening.* Manuscript robotics-4632217.
> Fundación Universitaria Los Libertadores, Bogotá, Colombia.

It contains everything needed by a reviewer or a third party to **generate the scenes, run experiments
E1–E10, and reproduce every table, figure and number of the paper**: the implementation of TVE, the fourteen
baselines of E1, the reimplementations of MVDRM and G2VD (E9, E10), the seven scene generators and the
narrow-passage maps, the camera image-formation model, the image-processing pipeline, the exact parameters
and seeds, the statistical analysis, the raw results that produced the paper and the scripts that rebuild the
tables and figures.

## 0. Version v1.1.0: what changed in revision 1

The previous release (v1.0.0, first submission) contained experiments E1–E7. This release contains everything
that the revised manuscript describes (details in `CHANGELOG.md`):

* **New experiments.** E8: one-at-a-time sensitivity of the parameters of TVE (K, α, β, δ, ℓ_g) on 63 maps
  not used elsewhere; E9: 144 synthetic maps with a narrow passage of 1–16 px of free width in three
  geometries; E10: MVDRM and G2VD (reimplemented in `src/baselines/voronoi_roadmaps.py`), WDT and TVE on the
  168 queries of E1. New table of ε-optimality (share of paths with L ≤ (1+ε)L*) computed from E1.
* **Correction of TVE.** When the skeleton with one generator per obstacle reported defects, the code kept it
  if the skeleton with contour generators reported the *same* number of defects, contrary to the procedure of
  the paper; now the contour skeleton is always used in that case (`src/tve/tve.py::TVEPlanner.prepare`). The
  correction changes only the maps in which both skeletons have defects: **12 of the 455 maps of E1–E7**,
  listed with the results of TVE before and after in `results/raw/r1_skeleton_correction_maps.csv` and
  `r1_skeleton_correction_queries.csv` (Table A2 of the paper). Those map jobs were executed again
  (`tools/rerun_affected.py`) and their rows replaced, in place, in `results/raw/e*.csv`; E4 was re-executed
  but did not change and keeps its original rows. `results/raw/revision1_reruns.json` lists the jobs, and
  `results/raw/original_submission/` keeps the raw results of the first submission (**before**), while
  `results/raw/` holds the corrected ones (**after**).
* **Revision checks.** `experiments/r1_reference_check.py` (bias of the reference optimum L*),
  `experiments/r1_pruning_check.py` (effect of the pruning of candidates) and
  `experiments/r1_skeleton_correction.py` (the 12 maps); their results are `results/raw/r1_*.csv`.
* **New parameters,** all with the values of v1.0.0 as defaults: `TVEConfig.penalty` (β = 1.7, hard-coded
  before), `TVEConfig.prune_bound` and `TVEConfig.n_validate` (diagnostic of the pruning), and the offset of
  the vertices of the visibility graph (`desplazamiento_px`, 2 px).
* **Figures.** All figures are drawn at their final printed width (13.86 cm or 18.47 cm) with a 7-pt font and
  carry 2-m scale bars on the maps; output names are descriptive (`fig_pipeline.pdf`, ...). New: the
  narrow-passage figure and **Supplementary Figure S1** (`figures/output/figS1_e1_maps.pdf`, the 84 maps of E1,
  produced by `figures/make_supplementary.py`, which also checks that the ratios printed in the figure equal
  those of `results/raw/e1.csv`).
* **Integrity.** `statistics/check_raw.py` checks that every raw file contains exactly the runs of the design
  and writes `results/raw_files_check.txt` and `results/checksums_sha256.txt`.

## 1. Contents

```
README.md                  this file
CHANGELOG.md               changes between v1.0.0 (first submission) and v1.1.0 (revision 1)
CITATION.cff               citation metadata (read by GitHub and Zenodo)
LICENSE                    MIT license
requirements.txt           exact versions of the dependencies (pip)
environment.yml            the same environment for conda
paper_experiments.py       master script of the RUN step (E1-E10)
paper_figures.py           master script of the ANALYZE and FIGURES steps
verify.py                  compares a new execution with the raw results of the paper
src/
  tve/                     TVE (tve.py) and its adapter to the common planner interface (adapter.py)
  baselines/               the fourteen baselines of E1 and voronoi_roadmaps.py (MVDRM, G2VD) (Section 6)
    wdt/                   WDT, the original wavefront-dilation tessellation method (Ladino et al., 2020)
    models/drl_dqn.pt      trained D3QN policy used in the paper
  common/                  shared infrastructure: environment, configuration space, exact collision
                           test, tessellation data structures, skeleton graph, planning problem, metrics
  perception/pipeline.py   image processing: rectification, floor-model segmentation, morphology
  registry.py              common interface of all planners (names, defaults, execution)
scenes/
  generators.py            the seven scene types (office, warehouse, apartment, workshop, store, plaza,
                           hospital ward)
  narrow_passages.py       synthetic maps with one narrow passage of controlled width (E9)
  generate.py              GENERATE step: builds every scene and exports S and T of every query (E1-E7)
image_formation/model.py   camera image-formation and degradation model (floor textures, shading,
                           cast shadows, illumination gradient, vignetting, noise, blur, JPEG, perspective)
configs/                   every parameter exactly as used (see Section 5)
seeds/experiment_seeds.csv seeds of E1-E10 and the rule that generates S and T
seeds/queries_E*.csv       S and T of every query of E1-E7 (produced by scenes/generate.py)
experiments/               one script per experiment: e1_comparison.py ... e10_voronoi_planners.py,
                           the revision checks r1_reference_check.py, r1_pruning_check.py and
                           r1_skeleton_correction.py,
                           _common.py (shared pipeline) and train_d3qn.py (optional retraining)
statistics/                nonparametric.py (Friedman, Wilcoxon, Holm, bootstrap, effect size),
                           analyze.py (ANALYZE step) and check_raw.py (integrity of the raw files)
tools/rerun_affected.py    re-executes the 12 map jobs affected by the correction of the skeleton
figures/make_figures.py    FIGURES step; output in figures/output/
figures/make_supplementary.py  Supplementary Figure S1 (84 maps of E1 with the paths of TVE, WDT, optimum)
results/raw/               raw results that produced the paper (e1.csv ... e10.csv), the results of the
                           first submission (original_submission/) and the log of the revision re-runs
results/tables/            tables of the paper as CSV, LaTeX bodies, numbers quoted in the text,
                           statistics_summary.json
results/verification*.txt  reports of the complete re-executions from this package
tests/                     tests of the statistics and of the determinism of the pipeline
```

## 2. Installation

Python **3.14.6** was used (Windows 11 x64, Intel Core i7-12700KF, 64 GB RAM).

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

or, with conda: `conda env create -f environment.yml && conda activate tve-reproducibility`.

The dependencies are NumPy 2.3.5, OpenCV 5.0.0 (`opencv-python`), pandas 3.0.3, Matplotlib 3.11.1,
PyTorch 2.14.0 (CPU, only for the D3QN baseline) and pytest 9.1.1. **SciPy is not required**: the
statistical tests are implemented in `statistics/nonparametric.py`, and the quadratic program of G2VD is
solved with an ADMM written with NumPy.

Check the installation (about 1 minute):

```bash
python -m pytest tests -q
```

## 3. Reproducing the paper: generate → run → analyze → figures

All commands are run from the root of the package.

### 3.1 Tables, figures and numbers from the raw results (no planner is executed, ~3 min)

```bash
python paper_figures.py
```

This regenerates, from `results/raw/`, every table of Section 4 of the paper and the measured part of
Table 1 (`results/tables/*.csv`, with the LaTeX bodies in `results/tables/latex/`), every number quoted in the
text (`results/tables/latex/numbers.tex`, `results/tables/statistics_summary.json`) and the figures
(`figures/output/`). Supplementary Figure S1 re-runs TVE, WDT and the visibility graph on the 168 queries of
E1: `python figures/make_supplementary.py --workers 8` (~5 min).

### 3.2 Full reconstruction from the seeds

| Step | Command | Output | Time* |
|---|---|---|---|
| 1. generate | `python scenes/generate.py` (add `--images` to save the images) | `seeds/queries_E*.csv`, `results/scenes/` | ~5 min |
| 2. run | `python paper_experiments.py all --workers 8` | `results/rerun/e1.csv` … `e10.csv` | 60–120 min |
| 3. analyze | `python statistics/analyze.py --raw results/rerun` | `results/tables/` | <1 min |
| 4. figures | `python figures/make_figures.py --raw results/rerun` | `figures/output/` | ~2 min |
| check | `python verify.py` | report of identical columns | <1 min |

\* with 8 worker processes on the machine described above. Steps 3 and 4 can be run together with
`python paper_figures.py --raw results/rerun`.

Every experiment can also be run on its own, e.g. `python experiments/e9_narrow_passages.py --workers 8`. The
option `--quick` runs one layout per scene and level; a quick execution is a subset of the full one and can be
checked with `python verify.py --subset`. New executions are written to `results/rerun/` and never overwrite
`results/raw/`.

### 3.3 Revision checks and integrity of the files

| Command | Output | Time* |
|---|---|---|
| `python experiments/r1_skeleton_correction.py --out results/rerun/r1_skeleton_correction_maps.csv` | the 12 maps affected by the correction and TVE before/after | ~1 min |
| `python experiments/r1_pruning_check.py --out results/rerun/r1_pruning_check.csv` | TVE with and without the pruning of candidates on the queries of E1 | <1 min |
| `python experiments/r1_reference_check.py --out results/rerun/r1_reference_check.csv` | L* with the default and with closer vertices on the queries of E1 | ~15 min |
| `python tools/rerun_affected.py` | the 12 map jobs re-executed and spliced into the results of v1.0.0 (`results/rerun/spliced/`) | ~5 min |
| `python statistics/check_raw.py` | `results/raw_files_check.txt`, `results/checksums_sha256.txt` | <1 min |

### 3.4 What is exactly reproducible

* **Geometric and topological results** — maps, images, detected obstacles, S and T, path lengths, length
  ratios, validity, clearances, turning radii, sharp turns, travel times of the robot models, safety against
  the ground truth, homotopy classes and image-processing metrics — are deterministic: all random generators
  are seeded (Section 4) and no planner uses a time budget. `verify.py` compares every non-timing column of a
  new execution with `results/raw/`: boolean and categorical columns must match exactly and numeric columns
  within a relative tolerance of 10⁻⁵, because vectorized NumPy reductions can differ in the last bits
  depending on memory alignment. The reports of the complete re-executions from this package are in
  `results/verification.txt` (first submission, E1–E7) and `results/verification_revision1.txt` (E1–E10).
* **Computation times** depend on the processor, the operating system and the load; they are not expected to
  coincide in milliseconds, but the comparative trends and the statistical conclusions should be recovered.
  Every process is limited to one thread (`OMP_NUM_THREADS=1`, `cv2.setNumThreads(1)`,
  `torch.set_num_threads(1)`).

## 4. Seeds and generation of S and T

`seeds/experiment_seeds.csv` lists, for every experiment, the scenes, levels, layout seeds, number of queries
per map and the strings that seed the random generators:

* scene and image: `numpy.random.default_rng(zlib.crc32(f"paper|{scene}|{level}|{seed}"))` — it draws the
  image conditions within the ranges of the level, renders the image and perturbs the four floor corners
  (±2 px, as marked by an operator);
* queries: `default_rng(zlib.crc32(f"q{n}|{scene}|{level}|{seed}"))` (E4: `q4|{scene}|{seed}`; E10 uses the
  queries of E1, `q1`);
* E9 (no camera): the narrow-passage map is seeded with `default_rng(10000·kind + 100·gap + seed)` and its
  query with `default_rng(1000·gap + seed + offset(kind))` (`experiments/e9_narrow_passages.py`).

S and T are drawn uniformly in the free space of **both** the detected and the ground-truth configuration
spaces, in the same connected region, at least 45% of the map side apart (30% in E3); the first two
conditions guarantee a valid query for every planner and a meaningful safety check
(`experiments/_common.py::sample_queries`). `scenes/generate.py` writes the resulting coordinates of E1–E7 to
`seeds/queries_E*.csv`; the coordinates of every query are also stored in every row of the raw results
(`sx, sy, tx, ty`). E5 discards the queries in which the new obstacle disconnects S from T or the first TVE
route fails (84 drawn, 76 used) and E7 those in which the reference optimum is not found (252 drawn, 250 used).

## 5. Configurations

| File | Content |
|---|---|
| `configs/experiment.json` | robot diameter (0.45 m), R_min (0.8 m), map side (400 px), levels, query separation, safety threshold, planner list and names |
| `configs/tve.json` | `planner`: TVEConfig used in E2, E3, E5–E9; `registry`: parameters of the common interface used in E1, E4 and E10; `ablation_E6`; `candidates_E7`; `sensitivity_E8` |
| `configs/baselines.json` | the parameters passed to each baseline (including MVDRM and G2VD), the WDT configuration used when it is called directly, and a glossary of the (Spanish) parameter identifiers |
| `configs/perception.json` | image-processing parameters |
| `configs/image_formation.json` | ranges of the image-formation model for the low, medium and high levels |
| `configs/scenes.json` | scene generators and floor sizes |
| `configs/d3qn_training.json` | training configuration of the D3QN baseline |

The configuration files were generated by introspection of the objects used in the experiments, and the
experiment scripts read them; they are therefore the parameters that produced the paper. Notes:

* The parameters of TVE and of the baselines were fixed before the experiments, are identical for every map
  and query, were not tuned on the evaluation maps and are not claimed to be optimal; E8 measures the
  sensitivity of those of TVE. Hybrid A* uses the turning radius R_min of the vehicle.
* In E1, E4 and E10, TVE was executed through the common interface, whose default fillet radius is 1.0 m (a
  target radius larger than R_min = 0.8 m); in the other experiments with R = R_min. The Ackermann feasibility
  is always evaluated with R_min = 0.8 m.
* MVDRM and G2VD are reimplementations from their published descriptions (Ayawli et al., 2021 and 2025; Wen
  et al., 2025, arXiv 2201.12981v4); the details that the descriptions leave open are documented at the top
  of `src/baselines/voronoi_roadmaps.py` and in Table 4 of the paper. They are not the official
  implementations of their authors.
* The illustrative pipeline, gallery and paths figures use the defaults of the common interface (paths
  figure: Hybrid A* with a 1.0 m turning radius).

## 6. Planners

| Paper name | Key | Implementation |
|---|---|---|
| TVE (proposed) | `tve` | `src/tve/tve.py` |
| WDT (Ladino et al., 2020) | `teselado` | `src/baselines/wdt/` |
| Dijkstra, A*, Theta* | `dijkstra`, `astar`, `theta` | `src/baselines/grid_search.py` |
| Visibility graph (reference optimum) | `visibility` | `src/baselines/visibility.py` |
| RRT, RRT*, PRM | `rrt`, `rrt_star`, `prm` | `src/baselines/sampling.py` |
| APF | `potential` | `src/baselines/potential.py` |
| Hybrid A* | `hybrid_astar` | `src/baselines/hybrid_astar.py` |
| ACO | `aco` | `src/baselines/aco.py` |
| Homotopy planner | `topological` | `src/baselines/topological.py` |
| SLAM-based | `slam` | `src/baselines/slam.py` |
| DRL (D3QN) | `drl` | `src/baselines/drl.py` + `src/baselines/models/drl_dqn.pt` |
| MVDRM (reimplementation) | `mvdrm` | `src/baselines/voronoi_roadmaps.py` |
| G2VD (reimplementation) | `g2vd` | `src/baselines/voronoi_roadmaps.py` |

All planners receive the same `Problem` (`src/common/problem.py`): the detected configuration space, S, T and
the scale; all paths are validated with the same exact segment–pixel collision test
(`src/common/geometry.py`) and measured with the same metrics (`src/common/metrics.py`, formulas in
Section 3.12 of the paper). The registry also contains `teselado_opt`, a variant of WDT that is not used in
the paper. The D3QN policy can be retrained with `python experiments/train_d3qn.py` (optional; the
experiments use the shipped model).

## 7. Experiments, tables and figures

| Exp. | Script | Paper | Raw results |
|---|---|---|---|
| E1 | `experiments/e1_comparison.py` | Tables 5, 6 (ε-optimality), 7, 8, 10; Figures 6, 7, 9, 11 | `results/raw/e1.csv` |
| E2 | `experiments/e2_tve_vs_wdt.py` | Table 9; Figure 8; paired tests of Section 4.3 | `results/raw/e2.csv` |
| E3 | `experiments/e3_multiquery.py` | Figure 12 | `results/raw/e3.csv` |
| E4 | `experiments/e4_scalability.py` | Figure 13 | `results/raw/e4.csv` |
| E5 | `experiments/e5_replanning.py` | Table 11 | `results/raw/e5.csv` |
| E6 | `experiments/e6_ablation.py` | Table 12 | `results/raw/e6.csv` |
| E7 | `experiments/e7_homotopy.py` | Table 14 | `results/raw/e7.csv` |
| E8 | `experiments/e8_sensitivity.py` | Table 13 | `results/raw/e8.csv` |
| E9 | `experiments/e9_narrow_passages.py` | Figure 10; Section 4.7 | `results/raw/e9.csv` |
| E10 | `experiments/e10_voronoi_planners.py` | Table 1b; Section 4.4 | `results/raw/e10.csv` |
| — | `figures/make_figures.py` | Figures 3, 4, 5 (qualitative) | — |
| — | `figures/make_supplementary.py` | Supplementary Figure S1 | — |
| R1 | `experiments/r1_reference_check.py` | Section 3.12 (reference optimum) | `results/raw/r1_reference_check.csv` |
| R1 | `experiments/r1_pruning_check.py` | Section 3.7 (pruning of candidates) | `results/raw/r1_pruning_check.csv` |
| R1 | `experiments/r1_skeleton_correction.py` | Table A2, Section 4.7 | `results/raw/r1_skeleton_correction_*.csv` |

Tables 2–4 of the paper are descriptive (scenes, levels, planners), Table A1 is the notation, Table A2 lists the 12 maps re-executed after the correction, and Figures 1
and 2 are the TikZ block and flow diagrams of the manuscript. The CSV files in `results/tables/` keep the
names of the first submission for the tables that already existed (e.g. `table05_e1_results.csv`); the new
ones are `table01b_voronoi_planners.csv`, `table06_epsilon_optimality.csv`, `table13_sensitivity.csv` and
`table14_narrow_passages.csv`.

**Statistics** (`statistics/nonparametric.py`, applied in `statistics/analyze.py`): Friedman test with average
ranks over the planners with at least 95% of valid paths and the queries solved by all of them; Wilcoxon
signed-rank tests (normal approximation with tie and continuity corrections) of TVE against every planner,
with Holm correction and effect size r = z/√n; percentile bootstrap (5000 resamples, seed 0) for the 95% CI of
the median speed-up. All values are written to `results/tables/statistics_summary.json`.

**Columns of the raw files** (identifiers of the source code): `exito` success, `sin_colision` collision-free
in the detected configuration space, `longitud_m` length, `relacion_optima` length / visibility-graph length,
`tiempo_ms` computation time, `holgura_min_m`/`holgura_media_m` minimum/mean clearance, `giro_total_deg`
total turning, `giros_bruscos` turns above 45°, `radio_giro_min_m` minimum turning radius,
`tiempo_holonomico_s`/`tiempo_diferencial_s`/`tiempo_ackermann_s` travel times, `factible_ackermann`
feasible for R_min, `gt_safe`/`gt_penetration_px`/`gt_clearance_m` safety against the ground truth,
`t_*_ms` stage times (`teselado` tessellation, `esqueleto` skeleton, `conexion_ST` S/T connection, `rutas`
route search, `reduccion` midpoint reduction, `embudo` funnel and validation, `suavizado` smoothing). E8 adds
`parameter`, `value`, `is_default`; E9 adds `kind`, `gap_px`, `gap_m`, `passage_clearance_px` and
`through_passage` (the valid path crosses the narrow passage).

## 8. Notes

* Levels are named `facil`, `medio`, `dificil` in the code and low, medium, high in the paper.
* Comments inside the core modules are partly in Spanish, the language of the authors; interfaces,
  configuration files and this documentation are in English.
* The camera images are synthesized from the ground-truth layouts so that collisions with the real obstacles
  can be measured pixel by pixel; a validation with captured photographs is planned as future work (see the
  Discussion of the paper).

## 9. Publishing this version (GitHub and Zenodo)

1. Replace the content of the GitHub repository with this folder (the `.gitignore` excludes new executions).
2. The code is released under the MIT license (`LICENSE`); select the same license in Zenodo.
3. Commit, create the tag `v1.1.0` and publish a GitHub release with that tag; with the Zenodo–GitHub
   integration enabled, Zenodo archives the release and assigns it a DOI (the metadata come from
   `CITATION.cff`). Alternatively, upload the zip of this folder to Zenodo as a new version of the existing
   record.
4. Insert the URL of the repository and the DOI of the release in the Data Availability Statement of the
   manuscript, which describes exactly the content of this version (E1–E10, the revision checks, the raw
   results before and after the correction of the skeleton and Supplementary Figure S1).
