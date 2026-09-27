# APEE: Adaptive Poly-Agentic Evaluation Ecosystem

A Python framework for evaluating teams of LLM agents with LLM judges. It runs small local models through collaborative scenarios in one of six coordination patterns, then has larger models from other families score the individual work, the collaboration and the overall run.

**Results and analysis:** [the blog post](index.md) walks through the December 2025 evaluation runs and what they show about LLM-as-a-judge scoring. This README covers using the package.

## Features

- **Six coordination patterns**: parallel, sequential (pipeline), debate, hierarchical, consensus and peer review ([details](apee/coordination/PATTERNS.md))
- **Twelve collaborative scenarios**, each exercising one pattern, from code review to incident response
- **Three levels of metrics**: individual (L1) and collaborative (L2) quality scored by LLM judges, and ecosystem health (L3) computed from the run
- **Four judging protocols**: a basic two-judge ensemble, progressive deepening, a four-persona jury, and a calibrated jury that negotiates a rubric first ([details](apee/evaluation/EVALUATION_PATTERNS.md))
- **Heuristic scorers** for fast checks without an LLM, plus a single-model benchmark suite (19 tasks in 11 categories)
- **Visualization, anomaly detection and a web dashboard** for inspecting results
- **Runs locally** on [Ollama](https://ollama.com); no API keys needed

## Quick start

Requires Python 3.10+ and a running Ollama server (`ollama serve`). The judges need a GPU that can hold a 20–24B model.

```bash
git clone https://github.com/ahjavid/technical-notes-blog.git
cd technical-notes-blog/posts/apee-evaluation-ecosystem
pip install -e ".[dev]"          # add ,viz for Plotly charts

# Agents: small models from different families
ollama pull qwen2.5-coder:3b
ollama pull llama3.2:3b
ollama pull phi4-mini:3.8b

# Judges: larger models from other families
ollama pull gpt-oss:20b
ollama pull mistral-small3.2:24b
```

### Run the evaluation

```bash
python examples/proper_apee_evaluation.py                      # basic two-judge ensemble
python examples/proper_apee_evaluation.py --mode progressive   # + progressive deepening
python examples/proper_apee_evaluation.py --mode jury          # + four-persona jury
python examples/proper_apee_evaluation.py --mode calibrated    # + calibrated jury
python examples/proper_apee_evaluation.py --mode all           # all four, one after another
```

Results are written to `data/`. Every run records the basic ensemble score as `overall_apee_score`. The extra protocol's own scores are stored under `advanced_evaluation`, as described in the [data guide](data/README.md).

### Other examples

| Script | What it does |
|---|---|
| [`examples/comprehensive_benchmark.py`](examples/comprehensive_benchmark.py) | Single-model benchmark across 19 tasks in 11 categories |
| [`examples/multi_model_evaluation.py`](examples/multi_model_evaluation.py) | Compare several Ollama models on the same tasks |
| [`examples/phase6_demo.py`](examples/phase6_demo.py) | Charts, anomaly detection and an HTML report from saved results |

## Using the API

```python
import asyncio

from apee import AgentRole, Coordinator, OllamaAgent, Task
from apee.evaluation.llm_evaluator import EnsembleEvaluator


async def main():
    # Order matters: the first agent leads hierarchical runs.
    agents = [
        OllamaAgent("analyst", AgentRole.ANALYZER, model="qwen2.5-coder:3b"),
        OllamaAgent("coder", AgentRole.EXECUTOR, model="llama3.2:3b"),
        OllamaAgent("reviewer", AgentRole.REVIEWER, model="phi4-mini:3.8b"),
    ]
    coordinator = Coordinator(agents=agents)
    task = Task(task_id="t1", description="Review this code for bugs: ...")

    # Pipeline: analyze → code → review
    results = await coordinator.run_pipeline(task, ["analyst", "coder", "reviewer"])

    # Other patterns
    # await coordinator.run_parallel(task)
    # await coordinator.run_debate(task, rounds=2)
    # await coordinator.run_hierarchical(task, leader_id="analyst")
    # await coordinator.run_consensus(task, max_rounds=3)
    # await coordinator.run_peer_review(task)

    evaluator = EnsembleEvaluator(
        judge_models=["gpt-oss:20b", "mistral-small3.2:24b"],
        aggregation="median",
    )
    # Build a CollaborativeTrace from the results, then:
    # evaluation = evaluator.evaluate_full(trace)
    # print(evaluation["overall_apee_score"])


asyncio.run(main())
```

[`examples/proper_apee_evaluation.py`](examples/proper_apee_evaluation.py) shows the complete flow, including how to build the `CollaborativeTrace`.

### Heuristic scoring (no LLM needed)

```python
from apee.evaluation.quality import CompositeScorer, HeuristicScorer

# `result` is an AgentResult from any coordinator run, `task` the Task it answered
quality = HeuristicScorer().score_sync(result, task)
print(quality.overall, quality.relevance, quality.completeness)

# Blend heuristics with a small LLM scorer (async)
blended = await CompositeScorer(heuristic_weight=0.4, llm_weight=0.6).score(result, task)
```

## Metrics

| Level | Scored by | Metrics |
|---|---|---|
| **L1: individual** | LLM judges | Goal alignment, semantic quality |
| **L2: collaborative** | LLM judges | Collaboration effectiveness, synthesis quality |
| **L3: ecosystem** | Computed | Efficiency, stability, throughput, adaptability |

```text
Overall APEE score = 0.30 × L1 + 0.45 × L2 + 0.25 × L3      (every score is 0–10)
```

## Package layout

```text
apee/
├── agents/          Agent base class and the Ollama implementation
├── coordination/    Coordinator with the six patterns (PATTERNS.md explains them)
├── evaluation/      LLM judges, heuristic scorers, advanced protocols (EVALUATION_PATTERNS.md)
├── benchmarks/      Single-model tasks (datasets.py) and the 12 collaborative scenarios (collaborative.py)
├── visualization/   Plotly and text charts, HTML export
├── anomaly/         Statistical anomaly detection and alerts
├── dashboard/       Web dashboard server and API client
├── utils/           Logging and helpers
├── models.py        Pydantic data models
└── cli.py           Command-line interface
examples/            The scripts listed above
tests/               Unit tests
data/                Results of the evaluation runs (see data/README.md)
```

## Tests

```bash
pytest tests/                       # 113 tests; 4 are skipped unless Ollama is running
pytest tests/test_coordinator.py    # the six coordination patterns
pytest tests/ --cov=apee            # with coverage (needs pytest-cov)
```

## Roadmap

- [x] Package structure, Ollama agents, six coordination patterns
- [x] Heuristic scoring and the single-model benchmark (19 tasks, 11 categories)
- [x] Twelve collaborative scenarios with three-level metrics
- [x] LLM-as-a-judge ensemble with judges from other model families
- [x] Visualization, anomaly detection and dashboard
- [x] Advanced protocols: progressive deepening, persona jury, calibrated jury
- [ ] Report each protocol's own score as the headline score for its mode
- [ ] Repeated runs with confidence intervals
- [ ] A human-labelled reference set for calibrating judges
- [ ] Publish to PyPI

## License

MIT. See [LICENSE](../../LICENSE).
