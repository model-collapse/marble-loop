# MARBLE plugin for Claude Code

**MARBLE — Monitored Analyze-Refine Budgeted Loop Engine.** Unattended, budgeted automated research on
white-box benchmarks: give it a benchmark (code + evaluator + metric) and a budget; it searches for better
solutions with Claude Code worker agents and reports what it found and why.

## What the plugin adds

| Component | What it does |
|---|---|
| `/marble` skill | `plan` (probe hardware, calibrate, propose `resources.yaml`, confirm with you) · `start` (supervised, detached) · `status` · `inspect` · `steer` · `stop` · `report` |
| `marble` MCP server | read run state from any session: runs, tree, nodes, diffs, lessons, incidents, reports, logs; `marble_control` to stop/pause/resume or pin a note for the planner |
| SessionStart hook | a short summary of your MARBLE runs at the start of each session |
| PreToolUse hook | blocks edits to run state, and to a benchmark's baseline/evaluator while a run on it is live |

## Install

```
/plugin install marble --marketplace <owner>/<repo>
```
(`<owner>/<repo>` is filled in when the repository is published.) For local development:
`claude --plugin-dir plugins/marble`.

## Requirements

- The `marble` CLI (Python ≥ 3.10): `pip install marble-loop`, or `pip install -e .` in a checkout of the
  MARBLE repository. Set `MARBLE_BIN` if it is not on `PATH`. Without it the hooks are no-ops.
- Claude Code with model access (Anthropic API, Bedrock or Vertex). MARBLE's worker agents are headless
  `claude -p` processes using the same credentials.
- Benchmarks: see the MARBLE repository's `benchmarks/` (toy CPU benchmarks, KernelBench, AlgoTune,
  karpathy/autoresearch adapters) or write your own `benchmark.yaml`.

## Quick start

```
/marble plan benchmarks/toy/pairwise
/marble start benchmarks/toy/pairwise
/marble status
/marble report <run>
```

## Safety

Worker agents run with an explicit tool allowlist, no user settings or MCP servers, a per-call $ cap and
timeout, and an edit guard that only allows the benchmark's editable files. Every candidate is evaluated in
a clean room; protected files are checksum-verified before and after each evaluation. Runs require a $
budget. See `DESIGN.md` in the repository for the full threat model.

## License

Apache-2.0
