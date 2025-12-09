# Adaptive Poly-Agentic Evaluation Ecosystem (APEE)

*A comprehensive framework for evaluating and benchmarking multi-agent AI systems*

---

## 📖 Overview

The Adaptive Poly-Agentic Evaluation Ecosystem (APEE) is a framework for systematically evaluating multi-agent AI systems. It provides adaptive evaluation methodologies that dynamically assess agent interactions, collaboration patterns, and emergent behaviors in complex AI ecosystems.

### 🎯 Key Features
- **Poly-Agentic Collaboration**: Multiple agents working together (debate, pipeline, peer review)
- **Three-Tier Metrics**: Individual → Collaborative → Ecosystem evaluation
- **Adaptive Evaluation**: Dynamic pattern detection and criteria adjustment
- **Quality Scoring**: Multi-dimensional response quality assessment
- **Ollama Integration**: Ready-to-use local LLM agent implementation
- **6 Collaboration Patterns**: parallel, sequential, debate, consensus, hierarchical, peer_review

---

## 🏆 Latest Results

### Multi-Agent Collaborative Evaluation

**Configuration**: 3 agents × 6 scenarios using debate, pipeline, and parallel patterns

| Scenario | Pattern | L1 Individual | L2 Collaborative | L3 Ecosystem | Overall |
|----------|---------|---------------|------------------|--------------|---------|
| collab_code_review | peer_review | 0.61 | 0.93 | 1.00 | **0.87** |
| research_synthesis | sequential | 0.64 | 0.97 | 1.00 | **0.89** |
| constrained_problem | debate | 0.60 | 0.90 | 1.00 | **0.85** |
| emergent_behavior | parallel | 0.64 | 0.66 | 1.00 | **0.76** |
| scalability_test | hierarchical | 0.62 | 0.64 | 1.00 | **0.75** |
| conflict_resolution | consensus | 0.63 | 0.64 | 1.00 | **0.75** |

### Aggregate APEE Scores

| Metric | Score |
|--------|-------|
| **Level 1 (Individual)** | 0.624 |
| **Level 2 (Collaborative)** | 0.790 |
| **Level 3 (Ecosystem)** | 1.000 |
| **Overall APEE Score** | **0.813** |

### APEE Framework Compliance ✅

| Component | Status | Evidence |
|-----------|--------|----------|
| Multi-Agent Collaboration | ✅ | 3 agents working together |
| Debate Pattern | ✅ | 1 scenario |
| Pipeline Pattern | ✅ | 1 scenario |
| Level 1 Metrics (Individual) | ✅ | Response time, length, success |
| Level 2 Metrics (Collaborative) | ✅ | Coordination efficiency, participation |
| Level 3 Metrics (Ecosystem) | ✅ | Stability, latency, completion |
| Adaptive Engine | ✅ | Pattern detection active |
| Pattern Detection | ✅ | "convergent_behavior" detected |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Ollama running locally (`ollama serve`)
- A model pulled (e.g., `ollama pull qwen2.5-coder:3b`)

### Installation

```bash
# Clone the repository
git clone https://github.com/ahjavid/technical-notes-blog.git
cd technical-notes-blog/posts/apee-evaluation-ecosystem

# Install the package
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

### Run the Full Evaluation

```bash
# Activate your Python environment (optional)
source /path/to/your/venv/bin/activate

# Run the full evaluation example
python examples/full_evaluation.py
```

### Basic Usage

```python
import asyncio
from apee import OllamaAgent, Coordinator, Evaluator, Task, AgentRole

async def main():
    # Create specialized agents
    agents = [
        OllamaAgent("analyst", AgentRole.ANALYZER, model="qwen2.5-coder:3b"),
        OllamaAgent("coder", AgentRole.EXECUTOR, model="qwen2.5-coder:3b"),
    ]
    
    # Define tasks
    tasks = [
        Task(task_id="t1", description="Analyze REST vs GraphQL"),
        Task(task_id="t2", description="Write a Fibonacci function"),
    ]
    
    # Execute and evaluate
    coordinator = Coordinator(agents=agents)
    results = await coordinator.execute_parallel(tasks)
    
    evaluator = Evaluator()
    report = evaluator.evaluate(coordinator)
    print(f"Success Rate: {report.individual.success_rate:.1%}")

asyncio.run(main())
```

---

## 📦 Package Structure

```
apee/
├── __init__.py              # Package exports
├── models.py                # Pydantic data models
├── agents/
│   ├── base.py              # Abstract Agent class
│   └── ollama.py            # Ollama LLM implementation (7 models)
├── coordination/
│   └── coordinator.py       # Task distribution & execution modes
├── evaluation/
│   ├── evaluator.py         # Main evaluation engine
│   ├── quality.py           # Quality scoring (heuristic, LLM, composite)
│   ├── adaptive.py          # 🆕 Adaptive engine with pattern detection
│   └── report.py            # Report data models
├── benchmarks/
│   ├── datasets.py          # 19 scenarios, 11 categories
│   ├── runner.py            # Statistical benchmark runner
│   ├── analyzer.py          # Analysis with confidence intervals
│   └── collaborative.py     # 🆕 Multi-agent evaluation scenarios
└── utils/
    ├── logging.py           # Logging utilities
    └── helpers.py           # Helper functions

examples/
├── full_evaluation.py       # Basic evaluation demo
├── comprehensive_benchmark.py # Single-model benchmarks
└── multi_agent_evaluation.py  # 🆕 Full APEE multi-agent demo
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     APEE Framework                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   Agent A    │  │   Agent B    │  │   Agent N    │              │
│  │  (Analyzer)  │  │   (Coder)    │  │  (Reviewer)  │              │
│  │ qwen2.5:3b   │  │  gemma3:4b   │  │  qwen3:4b    │              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
│         │                 │                 │                      │
│         └────────────┬────┴─────────────────┘                      │
│                      │                                             │
│         ┌────────────▼─────────────┐                               │
│         │   Adaptive Coordinator   │                               │
│         │  • Parallel Execution    │                               │
│         │  • Pipeline Orchestration│                               │
│         │  • Debate Management     │                               │
│         │  • Consensus Building    │                               │
│         └────────────┬─────────────┘                               │
│                      │                                             │
│  ┌───────────────────┼───────────────────┐                         │
│  │                   │                   │                         │
│  ▼                   ▼                   ▼                         │
│ ┌─────────┐    ┌─────────────┐    ┌────────────┐                   │
│ │ Level 1 │    │   Level 2   │    │  Level 3   │                   │
│ │Individual│   │Collaborative│    │ Ecosystem  │                   │
│ │ Metrics │    │   Metrics   │    │  Metrics   │                   │
│ └────┬────┘    └──────┬──────┘    └─────┬──────┘                   │
│      │               │                  │                          │
│      └───────────────┼──────────────────┘                          │
│                      │                                             │
│         ┌────────────▼─────────────┐                               │
│         │    Adaptive Engine       │                               │
│         │  • Pattern Detection     │                               │
│         │  • Anomaly Detection     │                               │
│         │  • Criteria Adjustment   │                               │
│         └──────────────────────────┘                               │
└─────────────────────────────────────────────────────────────────────┘
```

### Three-Tier Evaluation Metrics

| Level | Focus | Metrics |
|-------|-------|---------|
| **L1: Individual** | Single agent performance | Success rate, latency, quality score |
| **L2: Collaborative** | Multi-agent interaction | Coordination efficiency, synergy, conflict rate |
| **L3: Ecosystem** | System-level health | Stability, scalability, throughput |

---

## 📊 Evaluation Dimensions

### Individual Agent Performance
| Metric | Description | Measurement |
|--------|-------------|-------------|
| Task Completion | Success rate per agent | Percentage |
| Response Quality | Multi-dimensional scoring | Score 0-1.0 |
| Latency | Processing time | Milliseconds |
| Token Usage | Output verbosity | Token count |

### Multi-Agent Collaboration
| Metric | Description | Measurement |
|--------|-------------|-------------|
| Response Consistency | Agreement across agents | Similarity score |
| Coordination Efficiency | Parallel vs sequential gain | Multiplier |
| Synergy Score | Combined vs individual performance | Ratio |

### Ecosystem Health
| Metric | Description | Measurement |
|--------|-------------|-------------|
| Throughput | Tasks processed per second | Tasks/sec |
| Total Execution Time | End-to-end latency | Milliseconds |
| Agent Utilization | Work distribution balance | Variance |

---

## 🔬 Quality Scoring (Phase 2)

APEE includes multiple scoring strategies:

```python
from apee.evaluation.quality import (
    HeuristicScorer,    # Rule-based scoring
    ComparativeScorer,  # Relative comparison
    LLMScorer,          # LLM-as-judge
    CompositeScorer,    # Weighted combination
)

# Create composite scorer
scorer = CompositeScorer([
    (HeuristicScorer(), 0.6),
    (ComparativeScorer(), 0.4),
])

# Score a result
score = scorer.score(result, task)
print(f"Overall: {score.overall_score:.2f}")
print(f"Completeness: {score.dimension_scores['completeness']:.2f}")
print(f"Latency: {score.dimension_scores['latency']:.2f}")
```

---

## 🛠️ Tech Stack

```yaml
Runtime:
  - Python 3.10+
  - asyncio for concurrent agents
  - httpx for HTTP client
  - Pydantic for data validation
  
LLM Backend:
  - Ollama (local, free, private)
  - 7 models tested (see benchmark below)
  
Testing:
  - pytest
  - pytest-asyncio
```

---

## 📊 Benchmark Results

Comprehensive evaluation across multiple Ollama models using the APEE framework.
Following LLM evaluation best practices from lm-evaluation-harness and DeepEval.

### Benchmark Methodology

- **19 evaluation scenarios** across 11 task categories
- **5 complexity levels**: trivial, easy, medium, hard, expert
- **Multiple runs per scenario** for statistical significance
- **Ground truth comparison** where available
- **Keyword/constraint validation**
- **Structured output checking**

### Model Pool

| Model            | Parameters | Purpose                   |
|------------------|------------|---------------------------|
| qwen2.5-coder:3b | 3B         | Fast code generation      |
| qwen2.5-coder:7b | 7B         | Balanced code generation  |
| qwen3:4b         | 4B         | Reasoning with thinking mode |
| qwen3:8b         | 8B         | Advanced reasoning        |
| gemma3:4b        | 4B         | Google's efficient model  |
| granite4:3b      | 3B         | IBM's enterprise model    |
| llama3.2:3b      | 3B         | Meta's efficient model    |

### Comprehensive Benchmark Results

**Configuration**: 5 models × 19 scenarios × 2 runs = 190 total evaluations

| Model             | Quality    | ±Std  | Success | Latency |
|-------------------|------------|-------|---------|---------|
| **qwen3:4b**      | **0.898**  | 0.047 | 100%    | 3777ms  |
| llama3.2:3b       | 0.879      | 0.133 | 100%    | 2446ms  |
| gemma3:4b         | 0.869      | 0.108 | 100%    | 3023ms  |
| granite4:3b       | 0.860      | 0.132 | 100%    | **1879ms** |
| qwen2.5-coder:3b  | 0.848      | 0.128 | 100%    | 2011ms  |

### Performance by Task Category

| Category              | qwen3:4b  | llama3.2:3b | gemma3:4b | granite4:3b | qwen2.5-coder:3b |
|-----------------------|-----------|-------------|-----------|-------------|------------------|
| analysis              | 0.899     | 0.963       | 0.856     | 0.921       | **0.964**        |
| code_debug            | 0.787     | 0.874       | 0.867     | 0.913       | **0.914**        |
| code_explanation      | 0.929     | 0.904       | **0.933** | 0.853       | 0.844            |
| code_generation       | 0.909     | **0.950**   | 0.898     | 0.849       | 0.869            |
| code_review           | 0.942     | 0.953       | **0.956** | 0.935       | 0.936            |
| instruction_following | **0.871** | 0.550       | 0.552     | 0.540       | 0.564            |
| math                  | 0.759     | 0.819       | 0.884     | **0.918**   | 0.661            |
| qa_factual            | 0.927     | 0.939       | 0.938     | **0.943**   | 0.909            |
| qa_reasoning          | 0.943     | 0.910       | 0.931     | 0.933       | **0.966**        |
| reasoning             | 0.934     | 0.883       | 0.889     | 0.892       | **0.950**        |
| summarization         | **0.912** | 0.837       | 0.777     | 0.674       | 0.749            |

### 🏆 Winners

| Metric | Winner | Value |
|--------|--------|-------|
| **Best Quality** | qwen3:4b | 0.898 |
| **Lowest Variance** | qwen3:4b | ±0.047 |
| **Fastest** | granite4:3b | 1879ms |
| **Most Efficient** | granite4:3b | quality/latency |

### Key Insights

1. **qwen3:4b dominates instruction_following** (0.87 vs 0.54-0.56) - significantly better than all others
2. **qwen2.5-coder:3b excels at reasoning tasks** - top in analysis, code_debug, qa_reasoning, reasoning
3. **llama3.2:3b best at code_generation** - highest score (0.95) for generating code
4. **granite4:3b leads math and qa_factual** - best for factual/mathematical tasks
5. **gemma3:4b top at code_explanation and code_review** - best for understanding existing code
6. **All models achieved 100% success rate** - all capable of completing tasks
7. **Each model has unique strengths** - no single model dominates all categories

### Running Your Own Benchmark

```bash
# Run the comprehensive single-model benchmark
python examples/comprehensive_benchmark.py

# Run the full APEE multi-agent evaluation
python examples/multi_agent_evaluation.py

# Or quick test with subset
python -c "
from apee.benchmarks import BenchmarkRunner, BenchmarkConfig, TaskCategory
import asyncio

config = BenchmarkConfig(
    models=['qwen2.5-coder:3b'],
    runs_per_scenario=1,
    categories=[TaskCategory.CODE_GENERATION],
)
result = asyncio.run(BenchmarkRunner().run(config))
print(result.quality_ranking)
"
```

### Multi-Agent Collaboration Patterns

| Pattern | Description | Use Case |
|---------|-------------|----------|
| `parallel` | All agents work independently | Diverse perspectives |
| `sequential` | Pipeline: output flows to next | Multi-stage analysis |
| `debate` | Agents argue/critique each other | Decision making |
| `consensus` | Agents must agree on output | Critical decisions |
| `hierarchical` | Leader delegates to workers | Complex task breakdown |
| `peer_review` | Each agent reviews others' work | Code review workflows |

---

## 📈 Roadmap

### Phase 1: Foundation ✅
- [x] Define APEE framework architecture
- [x] Create evaluation metric taxonomy
- [x] Build proper Python package structure
- [x] Implement Ollama agent integration
- [x] Create coordinator with multiple execution modes
- [x] Build evaluator with comprehensive metrics

### Phase 2: Quality Scoring ✅
- [x] Implement heuristic-based scoring
- [x] Add comparative scoring
- [x] Create composite scorer framework
- [x] Add LLM-as-judge scorer
- [x] Write unit tests for scoring (18 tests)

### Phase 3: Comprehensive Benchmarks ✅
- [x] Create benchmark dataset (19 scenarios, 11 categories)
- [x] Implement statistical analysis (mean, std, CI)
- [x] Test with 5+ different Ollama models
- [x] Document performance across categories
- [x] Run multiple iterations for significance
- [x] Generate comprehensive reports

### Phase 4: Full APEE Compliance ✅
- [x] Multi-agent collaborative scenarios
- [x] Three-tier metrics (Individual → Collaborative → Ecosystem)
- [x] Adaptive engine with pattern detection
- [x] 6 collaboration patterns implemented
- [x] Multi-agent evaluation demo
- [x] All 34 tests passing

### Phase 5: Future Enhancements
- [ ] Create visualization utilities
- [ ] Add more collaboration scenarios
- [ ] Implement advanced anomaly detection
- [ ] Web dashboard for results
- [ ] Publish to PyPI (optional)

---

## 🧪 Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=apee
```

---

## 🤝 Contributing

Contributions welcome in:
- Evaluation metric proposals
- Agent scenario design
- Additional scorer implementations
- Documentation improvements

---

## 📚 References

- Multi-Agent Systems: A Modern Approach
- LLM-as-a-Judge Evaluation Patterns
- Emergent Behavior in Complex Systems

---

## 📄 License

MIT License. See [LICENSE](../../LICENSE) for details.

---

**Status**: ✅ APEE Framework Complete (Phases 1-4)  
**Tests**: 34 passing  
**Models Tested**: 7 (qwen2.5-coder:3b/7b, qwen3:4b/8b, gemma3:4b, granite4:3b, llama3.2:3b)  
**Author**: [ahjavid](https://github.com/ahjavid)
