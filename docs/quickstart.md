# MARBLE quick start

## 1. Install

```bash
git clone <this repo> marble && cd marble
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q                       # ~1 min, no API calls
claude -p "say ok"              # Claude Code must be able to reach a model
```

## 2. A first run on a toy benchmark (CPU, ~5 min, < $1)

```bash
marble bench validate benchmarks/toy/pairwise
marble run plan benchmarks/toy/pairwise --out runs/pw-plan --hours 0.5 --budget-usd 2 --gpus none
# read runs/pw-plan/resource_plan.md; edit runs/pw-plan/resources.yaml if needed
marble run start benchmarks/toy/pairwise --resources runs/pw-plan/resources.yaml --out runs/pw --supervise
```
In another terminal: `marble run status runs/pw`. When it stops: `runs/pw/report.md` and `runs/pw/best.patch`.

Steer a running run:
```bash
marble run control runs/pw --note "prefer numpy-only solutions"   # pinned into every planner prompt
marble run control runs/pw --stop                                  # stop after the current iteration
marble run resume runs/pw --max-iters 10 --budget-usd 4            # continue later
```

## 3. From Claude Code

```bash
claude --plugin-dir plugins/marble
```
then: `/marble plan benchmarks/toy/pairwise`, `/marble start …`, `/marble status`, `/marble inspect <run>`,
`/marble report <run>`. Claude asks how the resources may be used before anything starts.

## 4. Real benchmarks

See `benchmarks/README.md` for KernelBench (GPU), AlgoTune (CPU) and karpathy/autoresearch (GPU):
each has a generator that turns the upstream checkout into `benchmark.yaml` directories, plus notes on
hardware fit and integrity.

## 5. Your own benchmark

A directory with `benchmark.yaml` and `repo/`:
```yaml
schema_version: 1
name: my-bench
repo: repo
editable: ["solution.py"]
protected: ["eval/**"]
evaluate:
  quick: {cmd: "${MARBLE_PYTHON} eval/eval.py --quick --out {out}", timeout: 300}
  full:  {cmd: "${MARBLE_PYTHON} eval/eval.py --out {out}", timeout: 1800}
score: {file: "{out}/score.json", key: speedup, direction: maximize}
correctness: {key: correct, equals: true}
noise: {repeats: 3, min_rel_improvement: 0.03}
```
The evaluator writes `score.json` (`{"speedup": …, "correct": true}`). Check it with
`marble bench validate`, measure noise with `marble bench calibrate`, try a candidate with
`marble bench eval --candidate <dir>`.
