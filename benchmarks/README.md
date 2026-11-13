# MARBLE benchmark adapters

Each adapter turns an external white-box benchmark into MARBLE `benchmark.yaml` directories
(schema: `DESIGN.md` §3). Check one with:

```bash
marble bench validate  <dir>                 # schema + repo + globs + protected manifest
marble bench calibrate <dir> --repeats 3     # baseline duration + score noise → calibration.json
marble bench eval      <dir> [--candidate D] [--mode quick|full] [--repeats N]
```

| Adapter | Generator | Editable | Score | Needs |
|---|---|---|---|---|
| `toy/pairwise` | (static) | `solution.py` | speedup ↑ | numpy-free Python |
| `toy/fit` | (static) | `model.py` | val MSE ↓ | numpy |
| KernelBench | `kernelbench/make_problems.py` | `model_new.py`, `kernels/**` | speedup vs eager PyTorch ↑ | `$KERNELBENCH_ROOT` (+ `.venv`), NVIDIA GPU, CUDA toolkit |
| AlgoTune | `algotune/make_tasks.py` | `solver.py`, `solver_lib/**` | geo-mean speedup vs reference ↑ | `$ALGOTUNE_ROOT` (+ `.venv`) |
| autoresearch | `autoresearch/make_bench.py` | `train.py` | val_bpb after fixed budget ↓ | `$AUTORESEARCH_ROOT` (+ `.venv`, `prepare.py` run once), NVIDIA GPU |

Generated directories (`kernelbench/problems/`, `algotune/tasks/`, `autoresearch/generated/`) are not
committed; regenerate them from the subset files.

## KernelBench

```bash
export KERNELBENCH_ROOT=/path/to/KernelBench CUDA_HOME=/usr/local/cuda
python kernelbench/make_problems.py --subset kernelbench/subset.txt   # or --problem 1:88
marble bench calibrate kernelbench/problems/kb-l1-088-mingptnewgelu   # uses the 'reference' mode
```
- **Baseline** = the reference with `Model` renamed `ModelNew` (speedup ≈ 1). It fails KernelBench's
  static check (no custom kernel), so it can only be timed in the `reference` mode, which is
  `calibration_only` — MARBLE refuses to score candidates in it.
- **Correctness** = KernelBench's own check + its static anti-hacking checker (errors fail; warnings recorded).
- **Timing**: `speedup` uses MARBLE timing (reference and candidate interleaved in rounds, equal warm-up,
  median). KernelBench's own mean-based, candidate-first numbers are kept as `kb_speedup`; they are
  biased by first-call outliers (identical max-pool 2D copy: `kb_speedup` 0.68×, MARBLE 1.00× on A10G).
- **Memory**: many level-1 shapes target 80 GB GPUs (e.g. ReLU/Softmax/GELU inputs are 6.4 GB) and OOM
  in the correctness check on 24 GB GPUs. The starter subset only uses problems that fit an A10G.
- Upstream fix needed: `_process_input_tensor` cast integer inputs (class targets) to fp32 →
  `../marble-benchmarks/patches/kernelbench-keep-integer-input-dtype.patch`.
- Analyze tools: `nsys` (timeline) and `ncu` (counters; needs `NVreg_RestrictProfilingToAdminUsers=0`).

## AlgoTune

```bash
export ALGOTUNE_ROOT=/path/to/AlgoTune
python algotune/make_tasks.py --subset algotune/subset.txt            # fetches each dataset once
```
- **Baseline** `solver.py` delegates to the reference `Task.solve` (speedup ≈ 1); the reference source is
  copied to `eval/reference_impl.py` for the agent to read.
- **quick** = first 10 test problems × 3 fresh-process repeats (~15 s for pagerank);
  **full** = all test problems × 5 repeats. Score = geometric mean of per-problem min-time ratios.
- **Integrity**: candidate code only runs in worker processes; each problem is timed once per process
  (memoisation does not help: tested 0.96×); solutions return as JSON and are verified with
  `is_solution` in the driver (`python -I`, candidate dir not importable). A candidate that patched
  `is_solution` in its own process and returned wrong answers was rejected (10/10 invalid).
- Upstream fix needed for the official harness: `../marble-benchmarks/patches/algotune-sysmodules-snapshot.patch`.

## autoresearch

```bash
export AUTORESEARCH_ROOT=/path/to/autoresearch      # after `uv sync && uv run prepare.py` there
python autoresearch/make_bench.py --device-batch-size 16   # 16 fits 24 GB; upstream 128 needs 80 GB
```
- One mode (`train`): fixed 5-minute training budget + compile/eval (~7 min on A10G).
- `prepare.py` (data, budget, `evaluate_bpb`) is protected; wall clock is measured outside the process.
- **Known gap**: the model computes its own loss, so `val_bpb` is *self-reported*
  (`trust: self_reported` in score.json). Planned: protected re-evaluation from saved weights + logits.
