---
title: "APEE: Evaluating Multi-Agent LLM Systems with LLM-as-a-Judge"
description: >-
  An open framework for scoring teams of small LLM agents with larger LLM judges: 12
  collaboration scenarios, 6 coordination patterns and 4 judging protocols. The clearest
  result is about the judges: the protocol moved scores more than anything the agents did.
date: 2025-12-09
updated: 2026-09-27
tags: [LLM Evaluation, Multi-Agent Systems, LLM-as-a-Judge, Ollama]
metrics:
  - value: "3.6 → 9.0"
    label: "mean score for the same agents under different judging protocols"
  - value: "0.9 pts"
    label: "average gap between a scenario's best and worst score over four reruns (max 1.4)"
  - value: "43 / 48"
    label: "evaluations where mistral-small3.2 outscored gpt-oss"
key_findings:
  - "**The judging protocol dominated the score.** On the same agent outputs, the mean per-agent score was 9.0/10 with progressive deepening, 6.6–6.9 with the basic two-judge ensemble, 5.1 with a four-persona jury and 3.6 with a calibrated jury. An absolute LLM-judge score means little without its protocol."
  - "**Rerunning the pipeline moved scenario scores by up to 1.4 points.** Four runs of the same scenarios gave run averages from 7.02 to 7.50, and the best-scoring collaboration pattern was different in each run. One run can't rank patterns."
  - "**The two judges disagreed systematically.** mistral-small3.2:24b scored higher than gpt-oss:20b in 43 of 48 evaluations, by 0.71 points on average. With two judges, 'median' aggregation is simply their mean, so the choice of judges shifts every score."
  - "**Collaboration is the weak spot.** The LLM-judged collaboration score (L2) was the lowest and least stable level. The emergent-behavior scenario, where agents work in parallel, had the lowest L2 in every run (4.25–5.5)."
  - "**The framework itself is reusable.** Six coordination patterns, a three-level metric model and four judging protocols run entirely on local models through Ollama, with 113 unit tests."
resources:
  - title: Package documentation
    url: README.md
    note: Installation, API usage, evaluation modes and the package layout.
  - title: Raw results (JSON)
    url: data/README.md
    note: One file per run with every score, judge breakdown and judge feedback, plus a guide to each file.
  - title: Coordination patterns
    url: apee/coordination/PATTERNS.md
    note: How each of the six patterns passes messages between agents.
  - title: Evaluation patterns
    url: apee/evaluation/EVALUATION_PATTERNS.md
    note: Design of progressive deepening, the persona jury and the calibration loop.
---

Benchmarks like MMLU or HumanEval score one model on one answer. More and more systems now chain several models together: one plans, one writes code, one reviews. Their quality depends on how well the models hand work to each other, and single-model benchmarks can't see that.

**APEE** (Adaptive Poly-Agentic Evaluation Ecosystem) is my attempt at a reusable harness for that problem. It runs teams of small local models through collaborative scenarios and asks larger models to judge the result. This post describes the framework, what the December 2025 evaluation runs show, and, most usefully, how much the scores depend on *how* you ask the judges.

## How APEE works

### Agents: three small models from different families

| Role | Model | Why this model |
|---|---|---|
| Analyst (leads hierarchical runs) | `qwen2.5-coder:3b` | Best analysis score (0.939) in the single-model benchmark below |
| Coder | `llama3.2:3b` | Best code-generation score (0.983) |
| Reviewer | `phi4-mini:3.8b` | Best code-review score (0.991) |

### Judges: two larger models from other families

`gpt-oss:20b` and `mistral-small3.2:24b`. Neither shares a model family with an agent, which avoids a judge grading its own family's writing. Both judges score every run, and their results are combined per metric.

### Six coordination patterns

| Pattern | What the agents do |
|---|---|
| Parallel | Work independently; the best result is kept |
| Sequential | Pipeline: analyze → code → review, each stage building on the last |
| Debate | Several rounds of argument and critique |
| Hierarchical | The leader plans and delegates, workers execute, the leader gives feedback and synthesizes |
| Consensus | Iterate until the answers agree semantically |
| Peer review | Work → review each other → revise, then a meta-review |

The [pattern guide](apee/coordination/PATTERNS.md) documents exactly what each pattern passes between agents.

### Twelve scenarios

Each scenario exercises one pattern:

| Scenario | Pattern | Task |
|---|---|---|
| Collaborative code review | Peer review | Review a code change and recommend fixes |
| Documentation sprint | Peer review | Write documentation for a module together |
| Research synthesis | Sequential | Produce a research summary |
| Cross-domain knowledge transfer | Sequential | Carry expertise from one domain into another |
| Constrained problem solving | Debate | Solve an optimization problem under resource limits |
| Creative design | Debate | Design a solution from different perspectives |
| Adversarial code review | Debate | One agent attacks the code for vulnerabilities, another defends it |
| Scalability test | Hierarchical | Coordinate a team with a lead and several workers |
| Error recovery | Hierarchical | Recover from injected failures |
| Conflict resolution | Consensus | Resolve a disagreement between agents |
| Emergent behavior | Parallel | Look for novel interaction patterns |
| Real-time incident response | Parallel | Coordinate a response under time pressure |

### Three levels of metrics

| Level | Scored by | Metrics |
|---|---|---|
| **L1: individual** | LLM judges | Goal alignment and semantic quality of each agent's output |
| **L2: collaborative** | LLM judges | Collaboration effectiveness and synthesis quality of the team |
| **L3: ecosystem** | Computed from the run | Efficiency, stability, throughput and adaptability |

```text
Overall APEE score = 0.30 × L1 + 0.45 × L2 + 0.25 × L3      (every score is 0–10)
```

### Four judging protocols

| Protocol | How it scores an agent's output | Judge models |
|---|---|---|
| **Basic ensemble** | Each judge scores each metric; the scores are combined | gpt-oss:20b and mistral-small3.2:24b |
| **Progressive deepening** | Starts with a quick check and only digs deeper (standard → deep) when the result is unclear | gpt-oss:20b |
| **Persona jury** | Four personas (skeptic, literalist, optimist, pragmatist) score independently; their scores are averaged | gpt-oss:20b |
| **Calibrated jury** | The judges first negotiate a task-specific rubric, then a skeptic and a pragmatist score against it | gpt-oss:20b and mistral-small3.2:24b |

## How the December 2025 runs were set up

On December 11, 2025 I ran the full evaluation four times, once per protocol. Two details matter for reading the results:

1. **Every run re-executed all 12 scenarios**, so each run judged a fresh set of agent outputs.
2. **Every run's headline "overall APEE score" comes from the basic ensemble.** The progressive, jury and calibrated protocols were run *in addition*, on the same outputs, and their per-agent scores were stored separately in the results.

The four overall scores therefore measure how much one pipeline varies between reruns, not how the protocols differ. The protocol comparison is the second table below.

## Result 1: rerunning the same pipeline moves scores a lot

Overall APEE score for each scenario in each run. Runs are labelled by the extra protocol that ran alongside the basic ensemble.

| Scenario | Pattern | Run 1 (basic) | Run 2 (+progressive) | Run 3 (+jury) | Run 4 (+calibrated) | Spread |
|---|---|---:|---:|---:|---:|---:|
| Adversarial code review | Debate | 8.0 | 7.4 | 7.0 | 7.7 | 1.0 |
| Constrained problem | Debate | 7.5 | 7.0 | 6.5 | 7.2 | 1.0 |
| Documentation sprint | Peer review | 7.3 | 7.4 | 7.2 | 7.5 | 0.3 |
| Research synthesis | Sequential | 7.3 | 7.9 | 7.2 | 7.5 | 0.7 |
| Creative design | Debate | 7.3 | 7.7 | 7.7 | 6.7 | 1.0 |
| Knowledge transfer | Sequential | 7.0 | 7.6 | 7.3 | 6.9 | 0.7 |
| Scalability test | Hierarchical | 6.9 | 7.6 | 7.7 | 7.1 | 0.8 |
| Real-time incident | Parallel | 6.9 | 8.0 | 7.5 | 7.2 | 1.1 |
| Collaborative code review | Peer review | 6.8 | 8.2 | 6.8 | 7.3 | 1.4 |
| Emergent behavior | Parallel | 6.6 | 6.5 | 6.3 | 6.3 | 0.3 |
| Error recovery | Hierarchical | 6.6 | 6.7 | 6.9 | 5.9 | 1.0 |
| Conflict resolution | Consensus | 6.6 | 8.0 | 7.1 | 7.1 | 1.4 |
| **Mean** | | **7.07** | **7.50** | **7.10** | **7.02** | |

| Run | Mean overall | L1 individual | L2 collaborative | L3 ecosystem | Best pattern in this run |
|---|---:|---:|---:|---:|---|
| 1 | 7.07 | 6.90 | 6.57 | 8.16 | Debate (7.62) |
| 2 | 7.50 | 6.91 | 7.42 | 8.34 | Consensus (8.00) |
| 3 | 7.10 | 6.79 | 6.76 | 8.07 | Hierarchical (7.31) |
| 4 | 7.02 | 6.62 | 6.74 | 8.02 | Peer review (7.43) |

The best-scoring pattern is different in every run. Consensus goes from *last* place in run 1 (6.58) to *first* in run 2 (8.00), and two of the twelve scenarios moved by 1.4 points. With a single run per configuration, differences smaller than about a point between scenarios or patterns aren't distinguishable from noise. (Patterns are averaged over only one or two scenarios each, which makes the ranking even less stable.)

Two more patterns hold across all four runs:

- **L3 is the highest and steadiest level** (8.0–8.3). It's computed rather than judged, so it lifts every overall score by roughly the same amount.
- **L2, the collaboration score, is the weakest and least stable level.** The emergent-behavior scenario had the lowest L2 in every run, between 4.25 and 5.5. Agents working independently in parallel rarely produced output the judges considered integrated.

## Result 2: the judging protocol matters more than the agents

Within each of runs 2–4, the same agent outputs were scored twice: by the basic ensemble (its per-agent L1 score) and by that run's protocol.

| Protocol | Protocol's mean score | Basic ensemble, same outputs | Difference |
|---|---:|---:|---:|
| Progressive deepening | **9.04** | 6.91 | +2.13 |
| Persona jury | **5.05** | 6.79 | −1.74 |
| Calibrated jury | **3.64** | 6.62 | −2.98 |

Scores are per-agent means on a 0–10 scale. What happened in each protocol:

- **Progressive deepening** stopped early in 38 of 40 evaluations, and 32 never went past the first "quick" check.
- **Persona jury:** the persona averages ranged from 7.6 (optimist) down to 4.7 (pragmatist), 4.2 (skeptic) and 3.4 (literalist), and the personas disagreed strongly in 36 of 40 evaluations.
- **Calibrated jury:** the negotiated rubrics were strict, and the two personas agreed closely (a mean spread of 1.0 point).

The same work scored anywhere from 3.6 to 9.0 out of 10 depending on how the judges were asked. That's a larger effect than any difference between scenarios, patterns or runs.

What this means in practice:

- **Never compare absolute scores across judging setups.** "7.5/10" only means something next to other scores produced by the identical protocol, judges and prompts.
- **Fail-fast protocols are lenient.** Progressive deepening saves tokens by stopping at the first confident verdict, and on these outputs its first verdict was nearly always a pass.
- **Personas make the spread visible.** The literalist and optimist differed by more than 4 points on average. That spread is useful information about how ambiguous a task is, and averaging it away hides it.

## Result 3: the two judges disagree systematically

| Run | mistral-small3.2:24b minus gpt-oss:20b (mean) | Mistral higher in |
|---|---:|---:|
| 1 | +0.87 | 11 of 12 |
| 2 | +0.40 | 9 of 12 |
| 3 | +0.66 | 11 of 12 |
| 4 | +0.92 | 12 of 12 |

Mistral was the more generous judge in 43 of 48 scenario evaluations, by 0.71 points on average. The ensemble uses "median" aggregation to resist outlier judges, but the median of two values is just their mean, so it offers no such protection. With systematic offsets like this, swapping one judge for another would shift every score in the study.

## The single-model benchmark behind the role assignments

Before the multi-agent runs, I benchmarked six small models on 19 single-agent tasks across 11 categories (December 10, 2025). Each model ran each task once and was scored heuristically: ROUGE overlap, keyword and constraint checks, and ground truth where available.

| Model | Mean quality | ± std | Mean latency | Categories it led |
|---|---:|---:|---:|---|
| phi4-mini:3.8b | **0.892** | 0.092 | 3,853 ms | code_review (0.991), qa_factual (0.919) |
| llama3.2:3b | 0.889 | 0.130 | 3,000 ms | code_generation (0.983), reasoning (0.909) |
| gemma3:4b | 0.876 | 0.121 | 3,681 ms | none (runner-up in code_debug and qa_factual) |
| qwen3:4b | 0.866 | 0.077 | 7,457 ms | summarization (0.912), code_explanation (0.893), instruction_following (0.840) |
| granite4:3b | 0.837 | 0.101 | **1,904 ms** | math (0.889) |
| qwen2.5-coder:3b | 0.829 | 0.138 | 2,386 ms | code_debug (0.965), qa_reasoning (0.957), analysis (0.939) |

No model led every category, which is why each role went to the model that was strongest at that role's task.

## What I'd change next

1. **Report each protocol's own score** as the headline for its run, instead of storing it next to the basic ensemble score.
2. **Repeat every configuration at least three times** and report means with confidence intervals. Result 1 shows why.
3. **Add a small set of outputs graded by a human** to calibrate protocols against, so a score of 7 can be related to something external.
4. **Use an odd number of judges, or correct for each judge's offset**, so the aggregate stops depending on which pair of judges was chosen.

## Run it yourself

Everything runs locally through [Ollama](https://ollama.com). The judges need a GPU that can hold a 20–24B model.

```bash
git clone https://github.com/ahjavid/technical-notes-blog.git
cd technical-notes-blog/posts/apee-evaluation-ecosystem
pip install -e ".[dev]"

# Agents and judges
ollama pull qwen2.5-coder:3b
ollama pull llama3.2:3b
ollama pull phi4-mini:3.8b
ollama pull gpt-oss:20b
ollama pull mistral-small3.2:24b

# One run with the basic ensemble; add a protocol with --mode
python examples/proper_apee_evaluation.py --mode basic   # progressive | jury | calibrated | all

# Unit tests (no Ollama needed)
pytest tests/
```

The [package documentation](README.md) covers the Python API, the other example scripts and the package layout.

## Limitations

- **One run per protocol**, so agent randomness and judge randomness can't be separated, and no confidence intervals are possible.
- **No human reference labels.** The protocols disagree, and there's no ground truth to say which one is right.
- **Two judges only**, with a systematic offset between them.
- **Small agents.** 3–4B-parameter models were chosen so the whole study runs on one machine. Stronger agents may behave differently.
- **Raw results for the single-model benchmark aren't in the repository.** Only the multi-agent runs are in [`data/`](data/README.md).
- **Hardware isn't recorded** in the result files, so latencies from different machines aren't comparable.

---

*Revision, September 2026:* the earlier version of this post compared the four runs as if they measured four evaluation modes, and concluded that progressive mode was "most effective" because its run averaged 7.50. In the evaluation script (`examples/proper_apee_evaluation.py`), every run's overall score comes from the basic ensemble. The protocols' own scores (9.04, 5.05, 3.64) are stored separately and are reported here for the first time. The per-scenario tables now match the committed JSON; several rows in the earlier tables didn't. The separate long-form study (`comprehensive_apee_study.md`) was merged into this page.
