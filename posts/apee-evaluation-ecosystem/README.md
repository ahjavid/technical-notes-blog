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
  - Tested: qwen2.5-coder:3b, qwen3:8b, llama3.2:3b
  
Testing:
  - pytest
  - pytest-asyncio
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
- [x] Add LLM-as-judge scorer skeleton
- [x] Write unit tests for scoring

### Phase 3: Real-World Testing
- [ ] Test with different Ollama models
- [ ] Document performance across model sizes
- [ ] Create diverse evaluation scenarios
- [ ] Benchmark single vs multi-agent performance
- [ ] Publish findings

### Phase 4: Polish & Share
- [ ] Add CLI for running evaluations
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
