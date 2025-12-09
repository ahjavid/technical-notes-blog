# Adaptive Poly-Agentic Evaluation Ecosystem (APEE)

*A comprehensive framework for evaluating and benchmarking multi-agent AI systems*

---

## 📖 Overview

The Adaptive Poly-Agentic Evaluation Ecosystem (APEE) is a framework for systematically evaluating multi-agent AI systems. It provides adaptive evaluation methodologies that dynamically assess agent interactions, collaboration patterns, and emergent behaviors in complex AI ecosystems.

### 🎯 Key Features
- **Adaptive Evaluation**: Dynamic assessment that evolves with agent behavior
- **Poly-Agent Analysis**: Multi-agent interaction pattern recognition  
- **Ecosystem Metrics**: Holistic system-level performance measurement
- **Quality Scoring**: Multi-dimensional response quality assessment (Phase 2)
- **Ollama Integration**: Ready-to-use local LLM agent implementation

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
├── __init__.py           # Package exports
├── models.py             # Pydantic data models
├── agents/
│   ├── base.py           # Abstract Agent class
│   └── ollama.py         # Ollama LLM implementation
├── coordination/
│   └── coordinator.py    # Task distribution & execution
├── evaluation/
│   ├── evaluator.py      # Main evaluation engine
│   ├── quality.py        # Phase 2: Quality scoring
│   └── report.py         # Report data models
└── utils/
    ├── logging.py        # Logging utilities
    └── helpers.py        # Helper functions
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    APEE Framework                               │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   Agent A    │  │   Agent B    │  │   Agent N    │           │
│  │  (Analyzer)  │  │  (Executor)  │  │   (Critic)   │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
│         │                 │                 │                   │
│         └────────────┬────┴─────────────────┘                   │
│                      │                                          │
│         ┌────────────▼─────────────┐                            │
│         │   Adaptive Coordinator   │                            │
│         │  • Task Distribution     │                            │
│         │  • Parallel Execution    │                            │
│         │  • Pipeline Orchestration│                            │
│         └────────────┬─────────────┘                            │
│                      │                                          │
│         ┌────────────▼─────────────┐                            │
│         │   Evaluation Engine      │                            │
│         │  • Performance Metrics   │                            │
│         │  • Quality Scoring       │                            │
│         │  • Ecosystem Analysis    │                            │
│         └──────────────────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

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

| Model | Quality | ±Std | Success | Latency |
|-------|---------|------|---------|---------|
| **qwen3:4b** | **0.898** | 0.047 | 100% | 3777ms |
| llama3.2:3b | 0.879 | 0.133 | 100% | 2446ms |
| gemma3:4b | 0.869 | 0.108 | 100% | 3023ms |
| granite4:3b | 0.860 | 0.132 | 100% | **1879ms** |
| qwen2.5-coder:3b | 0.848 | 0.128 | 100% | 2011ms |

### Performance by Task Category

| Category | qwen3:4b | llama3.2:3b | gemma3:4b |
|----------|----------|-------------|-----------|
| code_generation | 0.895 | **0.947** | 0.883 |
| code_review | 0.921 | **0.970** | 0.942 |
| code_explanation | 0.912 | **0.926** | 0.895 |
| code_debug | 0.888 | 0.879 | **0.893** |
| reasoning | **0.901** | 0.880 | 0.902 |
| analysis | 0.909 | **0.935** | 0.884 |
| qa_factual | 0.911 | **0.948** | 0.908 |
| qa_reasoning | 0.938 | 0.921 | **0.967** |
| math | 0.886 | 0.811 | **0.895** |
| instruction_following | **0.860** | 0.538 | 0.640 |
| summarization | **0.871** | 0.817 | 0.769 |

### 🏆 Winners

| Metric | Winner | Value |
|--------|--------|-------|
| **Best Quality** | qwen3:4b | 0.898 |
| **Lowest Variance** | qwen3:4b | ±0.047 |
| **Fastest** | granite4:3b | 1879ms |
| **Most Efficient** | granite4:3b | quality/latency |

### Key Insights

1. **qwen3:4b achieved highest quality (0.898)** with lowest variance - most consistent
2. **llama3.2:3b excels at code tasks** - top performer in code_generation, code_review
3. **qwen3:4b dominates instruction_following** - significantly better than others (0.86 vs 0.54-0.64)
4. **granite4:3b offers best speed** - nearly 2x faster than qwen3:4b
5. **All models achieved 100% success rate** - all capable of completing tasks
6. **gemma3:4b strong at qa_reasoning and math** - best for analytical tasks

### Running Your Own Benchmark

```bash
# Run the comprehensive benchmark
python examples/comprehensive_benchmark.py

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
- [x] Write unit tests for scoring

### Phase 3: Real-World Testing ✅
- [x] Create benchmark dataset (19 scenarios, 11 categories)
- [x] Implement statistical analysis (mean, std, CI)
- [x] Test with 5+ different Ollama models
- [x] Document performance across categories
- [x] Run multiple iterations for significance
- [x] Generate comprehensive reports

### Phase 4: Polish & Share
- [x] Add CLI for running evaluations
- [x] Benchmark analyzer with comparisons
- [ ] Create visualization utilities
- [ ] Write comprehensive documentation
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

**Status**: 🚧 Active Development  
**Author**: [ahjavid](https://github.com/ahjavid)
