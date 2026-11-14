# MARBLE — Monitored Analyze-Refine Budgeted Loop Engine

MARBLE runs **unattended, budgeted automated research on white-box benchmarks**: you give it a benchmark
(baseline code + an evaluator you can read + a metric) and a budget, and it keeps improving the code with
Claude Code worker agents — analyse, plan one change, implement, evaluate in a clean room, reflect — until
the budget, the clock or progress runs out. Then it tells you what it found, how, and why.

It builds on *MARS: Modular Agent with Reflective Search* (arXiv 2602.02660) and adds what unattended runs
need: an Inspector against stuck jobs and dead loops, a supervisor, integrity guards against gaming the
evaluator, and a lesson pool where lessons have to earn trust.

| Example run (4× A10G box) | Baseline → best | Cost | Time |
|---|---|---|---|
| AlgoTune `ode_brusselator` | 1.0× → **800×** (confirmed on the full test set) | $1.04 | 20 min |
| AlgoTune `pagerank` | 1.0× → **41.6×** quick / 35.3× full | $0.53 | 7 min |
| KernelBench MinGPT GELU | 1.0× → **8.3×** (fused CUDA kernel, ~75% of DRAM bandwidth) | $0.41 | 12 min |
| KernelBench LayerNorm | 1.0× → **4.9×** | $1.07 | 22 min |
| karpathy/autoresearch | val_bpb 1.326 → **1.179** | $0.45 | 70 min |

Speedups are measured by MARBLE's harness for each benchmark (see `benchmarks/README.md`); they are not
leaderboard numbers until checked with the benchmark's official evaluator.

## How it works

- **Benchmark contract** (`benchmark.yaml`): editable files, protected files (SHA-256 checked before and
  after every eval), eval modes (`quick` for the loop, `full` to confirm new bests), score + direction,
  correctness gate, noise. Adapters: toy CPU benchmarks, KernelBench, AlgoTune, karpathy/autoresearch.
- **Search**: budget-aware MCTS over candidates (draft a new approach / improve a node), reward
  `G · (t/L)^w` so faster candidates win ties; Thompson sampling over approach families.
- **Roles**: Analyst (measured facts, from profilers like `nsys`/`ncu`), Planner (one targeted change,
  cites lessons), Implementer, Debugger, Reflector — each an isolated headless `claude -p` worker with an
  explicit tool list, a $ cap, a timeout and an edit guard that only allows editable files.
- **Integrity**: clean-room evaluation of every candidate, protected-file checksums, correctness first,
  noise-aware comparisons, confirmation in `full` mode, integrity gate on lessons.
- **Lessons**: typed (positive / scoped negative / implementation note / debug / inspection), admission
  gates, Beta trust from citations and isolated lesson tests, negative lessons expire.
- **Inspector**: expected durations from calibration; slow-but-active jobs get one extension, stalled or
  hung jobs are investigated by a cheap model that sees the evidence and the candidate's code; repeated
  failures, duplicate code and repeated ideas are detected; a supervisor restarts a crashed controller.
- **Resources**: `marble run plan` probes the machine, calibrates the baseline and proposes
  `resources.yaml`; you decide how the resources may be used; nothing starts without a $ budget.

Full design: [`DESIGN.md`](DESIGN.md).

## Install

```bash
pip install -e ".[dev]"          # Python >= 3.10; provides the `marble` CLI
claude --version                 # Claude Code CLI with model access (Anthropic API, Bedrock or Vertex)
```
Claude Code plugin (skill `/marble`, MCP server, hooks): `plugins/marble/` — `claude --plugin-dir plugins/marble`.

## Quick start

```bash
marble bench validate benchmarks/toy/pairwise
marble run plan  benchmarks/toy/pairwise --out runs/pw-plan --hours 0.5 --budget-usd 2
marble run start benchmarks/toy/pairwise --resources runs/pw-plan/resources.yaml --out runs/pw --supervise
marble run status runs/pw
marble run report runs/pw            # runs/pw/report.md, runs/pw/best.patch
```
More: [`docs/quickstart.md`](docs/quickstart.md) · benchmark adapters: [`benchmarks/README.md`](benchmarks/README.md).

## Repository layout

```
src/marble/     spec, cleanroom, launcher, evaluate, calibrate        (benchmark SDK)
                workspace, runstate, agents, hooks, roles, search      (the loop)
                policy (MCTS), lessons, inspector, supervisor          (search, memory, safety)
                resources (planner), report, mcp_server, replay, cli
benchmarks/     toy/, kernelbench/, algotune/, autoresearch/          (adapters + generators)
plugins/marble/ Claude Code plugin
tests/          unit, fault-injection and replay tests (no API key needed)
```

## Testing

`pytest` runs everything without API calls: unit tests, a fault-injection suite (hangs, infinite loops,
repeated failures, duplicate code, controller crashes) and a replay of a recorded live run.

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
