# Adaptive Poly-Agentic Evaluation Ecosystem (APEE)

*A comprehensive framework for evaluating and benchmarking multi-agent AI systems using LLM-as-a-Judge methodology*

---

## 📖 Overview

The Adaptive Poly-Agentic Evaluation Ecosystem (APEE) is a framework for systematically evaluating multi-agent AI systems. It uses **LLM-as-a-Judge** evaluation (inspired by CrewAI) where large language models evaluate agent outputs rather than simple heuristics, providing meaningful, nuanced scores.

### 🎯 Key Features
- **LLM-as-a-Judge Evaluation**: Large models (12-14B) evaluate smaller agent outputs
- **Ensemble Judges**: Multiple judge models from different families reduce bias
- **Poly-Agentic Collaboration**: Multiple agents working together with 6 patterns
- **Three-Tier Metrics**: Individual → Collaborative → Ecosystem evaluation
- **Role-Optimized Agents**: Agent selection based on benchmark strengths
- **6 Collaboration Patterns**:
  - `run_parallel()` - All agents work independently
  - `run_pipeline()` - Sequential output flow
  - `run_debate()` - Multi-round argument
  - `run_hierarchical()` - Leader delegates to workers
  - `run_consensus()` - Iterate until agreement
  - `run_peer_review()` - Work → review → revise

---

## 🏆 Latest Results (LLM-as-a-Judge)

### Configuration

**Agents** (small, diverse families - matched to role strengths):
| Role | Model | Family | Benchmark Strength |
|------|-------|--------|-------------------|
| Coder (Executor) | llama3.2:3b | Llama | code_generation: 0.950 |
| Analyst (Analyzer) | qwen2.5-coder:3b | Qwen | analysis: 0.964 |
| Reviewer | granite4:3b | Granite | code_review: 0.935 |

**Judges** (large, different families - no overlap with agents):
| Judge | Model | Size | Family |
|-------|-------|------|--------|
| Judge 1 | qwen3:14b | 14B | Qwen |
| Judge 2 | gemma3:12b | 12B | Gemma |

### Multi-Agent Collaborative Evaluation

| Scenario | Pattern | L1 Individual | L2 Collaborative | L3 Ecosystem | Overall |
|----------|---------|---------------|------------------|--------------|---------|
| collab_code_review | peer_review | 7.3/10 | 6.2/10 | 7.1/10 | **6.8/10** |
| research_synthesis | sequential | 7.5/10 | 6.0/10 | 7.9/10 | **6.9/10** |
| constrained_problem | debate | 7.1/10 | 5.6/10 | 7.1/10 | **6.5/10** |
| emergent_behavior | parallel | 7.9/10 | 5.0/10 | 8.0/10 | **6.6/10** |
| scalability_test | hierarchical | 7.2/10 | 5.8/10 | 7.9/10 | **6.7/10** |
| conflict_resolution | consensus | 8.6/10 | 7.0/10 | 8.1/10 | **7.8/10** |

### Ensemble Judge Agreement

```
Judge Models: qwen3:14b, gemma3:12b
Aggregation: median

Individual Judge Scores (code_review scenario):
  • qwen3:14b: Overall=6.62, L1=7.33, L2=6.25
  • gemma3:12b: Overall=6.98, L1=7.5, L2=6.5

Disagreement Metrics:
  • Overall StdDev: 0.25
  • Overall Range: 0.36
  • High Disagreement: ✅ No
```

### Detailed Metric Breakdown (Code Review Scenario)

```
Level 1 (Individual - per agent):
  • executor: Goal=9.0, Semantic=8.0
  • analyzer: Goal=8.0, Semantic=7.0
  • reviewer: Goal=7.0, Semantic=4.0

Level 2 (Collaborative):
  • Collaboration: 4.0/10
  • Synthesis: 8.0/10

Level 3 (Ecosystem):
  • Efficiency: 2.3/10
  • Stability: 10.0/10
  • Throughput: 10.0/10
  • Adaptability: 6.0/10
```

### Key Insights

1. **Consensus pattern wins**: 7.8/10 overall (highest) - agents agreeing works best
2. **Parallel has lowest L2**: 5.0/10 - independent work hurts collaboration score
3. **Debate struggles with collaboration**: L2=5.6/10
4. **Executor (llama3.2:3b) excels**: Goal=9.0, Semantic=8.0 (highest individual)
5. **Judges agree reasonably**: StdDev 0.25 indicates reliable evaluation
6. **All 6 patterns tested**: peer_review, sequential, debate, parallel, hierarchical, consensus

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Ollama running locally (`ollama serve`)
- Models pulled:
  ```bash
  # Agents (small, diverse)
  ollama pull llama3.2:3b
  ollama pull qwen2.5-coder:3b
  ollama pull granite4:3b
  
  # Judges (large, different families)
  ollama pull qwen3:14b
  ollama pull gemma3:12b
  ```

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

### Run the LLM-as-a-Judge Evaluation

```bash
# Run proper APEE evaluation with ensemble judges
python examples/proper_apee_evaluation.py
```

### Basic Usage

```python
import asyncio
from apee import OllamaAgent, Coordinator, Task, AgentRole
from apee.evaluation.llm_evaluator import EnsembleEvaluator

async def main():
    # Create role-optimized agents (small, diverse families)
    agents = [
        OllamaAgent("coder", AgentRole.EXECUTOR, model="llama3.2:3b"),
        OllamaAgent("analyst", AgentRole.ANALYZER, model="qwen2.5-coder:3b"),
        OllamaAgent("reviewer", AgentRole.REVIEWER, model="granite4:3b"),
    ]
    
    # Create ensemble evaluator (large judges, different families)
    evaluator = EnsembleEvaluator(
        judge_models=["qwen3:14b", "gemma3:12b"],
        aggregation="median",
    )
    
    # Coordinate and evaluate
    coordinator = Coordinator(agents=agents)
    task = Task(task_id="t1", description="Review this code for bugs")
    results = await coordinator.run_pipeline(task, ["coder", "analyst", "reviewer"])
    
    # Evaluate with LLM-as-a-Judge
    # ... build CollaborativeTrace from results
    # evaluation = evaluator.evaluate_full(trace)

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
│   └── ollama.py            # Ollama LLM implementation
├── coordination/
│   └── coordinator.py       # Task distribution & execution modes
├── evaluation/
│   ├── evaluator.py         # Heuristic evaluation engine
│   ├── llm_evaluator.py     # 🆕 LLM-as-a-Judge evaluators
│   ├── quality.py           # Quality scoring strategies
│   ├── adaptive.py          # Adaptive pattern detection
│   └── report.py            # Report data models
├── benchmarks/
│   ├── datasets.py          # 19 scenarios, 11 categories
│   ├── runner.py            # Statistical benchmark runner
│   ├── analyzer.py          # Analysis with confidence intervals
│   └── collaborative.py     # Multi-agent evaluation scenarios
└── utils/
    ├── logging.py           # Logging utilities
    └── helpers.py           # Helper functions

examples/
├── full_evaluation.py           # Basic evaluation demo
├── comprehensive_benchmark.py   # Single-model benchmarks
├── multi_agent_evaluation.py    # Multi-agent with heuristics
└── proper_apee_evaluation.py    # 🆕 LLM-as-a-Judge evaluation
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     APEE Framework                                  │
├─────────────────────────────────────────────────────────────────────┤
│  AGENTS (Small 3B models - diverse families)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │   Coder      │  │   Analyst    │  │   Reviewer   │               │
│  │  (Executor)  │  │  (Analyzer)  │  │  (Reviewer)  │               │
│  │ llama3.2:3b  │  │qwen2.5-coder │  │ granite4:3b  │               │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘               │
│         │                 │                 │                       │
│         └────────────┬────┴─────────────────┘                       │
│                      │                                              │
│         ┌────────────▼─────────────┐                                │
│         │      Coordinator         │                                │
│         │  • run_parallel()        │                                │
│         │  • run_pipeline()        │                                │
│         │  • run_debate()          │                                │
│         │  • run_hierarchical()    │                                │
│         │  • run_consensus()       │                                │
│         │  • run_peer_review()     │                                │
│         └────────────┬─────────────┘                                │
│                      │                                              │
│  JUDGES (Large 12-14B models - different families)                  │
│  ┌───────────────────────────────────────────────┐                  │
│  │  ┌──────────┐          ┌──────────┐           │                  │
│  │  │  Qwen    │    +     │  Gemma   │           │                  │
│  │  │  14B     │          │   12B    │           │                  │
│  │  └──────────┘          └──────────┘           │                  │
│  │         │ Ensemble Evaluation │               │                  │
│  │         └──────────┬──────────┘               │                  │
│  └───────────────────────────────────────────────┘                  │
│                      │                                              │
│  ┌───────────────────┼───────────────────┐                          │
│  │                   │                   │                          │
│  ▼                   ▼                   ▼                          │
│ ┌─────────┐    ┌─────────────┐    ┌────────────┐                    │
│ │ Level 1 │    │   Level 2   │    │  Level 3   │                    │
│ │Individual│   │Collaborative│    │ Ecosystem  │                    │
│ │ Metrics │    │   Metrics   │    │  Metrics   │                    │
│ │Goal,Sem.│    │Collab,Synth │    │Eff,Stab,Thr│                    │
│ └────┬────┘    └──────┬──────┘    └─────┬──────┘                    │
│      │               │                  │                           │
│      └───────────────┼──────────────────┘                           │
│                      │                                              │
│         ┌────────────▼─────────────┐                                │
│         │   Overall APEE Score     │                                │
│         │  (L1×0.30 + L2×0.45 +    │                                │
│         │   L3×0.25)               │                                │
│         └──────────────────────────┘                                │
└─────────────────────────────────────────────────────────────────────┘
```

### Three-Tier Evaluation Metrics

| Level | Focus | Metrics |
|-------|-------|---------|
| **L1: Individual** | Single agent performance | Goal alignment, semantic quality |
| **L2: Collaborative** | Multi-agent interaction | Collaboration effectiveness, synthesis quality |
| **L3: Ecosystem** | System-level health | Efficiency, stability, throughput, adaptability |

---

## 📊 Evaluation Dimensions

### Level 1: Individual Agent Performance (LLM-Evaluated)
| Metric | Description | Scale |
|--------|-------------|-------|
| Goal Alignment | Did agent achieve the task? | 0-10 |
| Semantic Quality | Is reasoning clear and logical? | 0-10 |

### Level 2: Multi-Agent Collaboration (LLM-Evaluated)
| Metric | Description | Scale |
|--------|-------------|-------|
| Collaboration Effectiveness | Did agents work well together? | 0-10 |
| Synthesis Quality | Is combined output coherent? | 0-10 |

### Level 3: Ecosystem Health (Computed)
| Metric | Description | Scale |
|--------|-------------|-------|
| Efficiency | Output quality per unit time | 0-10 |
| Stability | Inverse of conflicts | 0-10 |
| Throughput | Agents utilized effectively | 0-10 |
| Adaptability | Pattern appropriateness | 0-10 |

---

## 🔬 Evaluation Methods

APEE supports two evaluation approaches:

### 1. LLM-as-a-Judge (Recommended)
```python
from apee.evaluation.llm_evaluator import EnsembleEvaluator

# Ensemble of large judges from different families
evaluator = EnsembleEvaluator(
    judge_models=["qwen3:14b", "gemma3:12b"],
    aggregation="median",
)

# Evaluates: goal alignment, semantic quality, collaboration, synthesis
result = evaluator.evaluate_full(collaborative_trace)
print(f"Overall: {result['overall_apee_score']}/10")
```

### 2. Heuristic Scoring (Fast, no LLM needed)
```python
from apee.evaluation.quality import CompositeScorer, HeuristicScorer

scorer = CompositeScorer([
    (HeuristicScorer(), 1.0),
])
score = scorer.score(result, task)
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

Following LLM-as-a-Judge best practices, models are designated as either **agents** (being evaluated) or **judges** (doing evaluation).

#### Agent Models (Small, 3-4B)

| Model            | Family  | Params | Role Optimization | Benchmark Score |
|------------------|---------|--------|-------------------|-----------------|
| llama3.2:3b      | Llama   | 3B     | CODER             | code_gen=0.950  |
| qwen2.5-coder:3b | Qwen    | 3B     | ANALYZER          | analysis=0.964  |
| granite4:3b      | Granite | 3B     | REVIEWER          | code_review=0.935 |

#### Judge Models (Large, 12-14B+)

| Model      | Family | Params | Purpose                        |
|------------|--------|--------|--------------------------------|
| qwen3:14b  | Qwen   | 14B    | Primary judge, deep reasoning  |
| gemma3:12b | Gemma  | 12B    | Secondary judge, clear analysis |

#### Legacy Models (Available for Benchmarking)

| Model            | Family | Params | Notes                          |
|------------------|--------|--------|--------------------------------|
| qwen2.5-coder:7b | Qwen   | 7B     | Medium coding model            |
| qwen3:4b         | Qwen   | 4B     | Small reasoning model          |
| qwen3:8b         | Qwen   | 8B     | Medium reasoning model         |
| gemma3:4b        | Gemma  | 4B     | Small analysis model           |

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

| Pattern | Method | Description | Use Case |
|---------|--------|-------------|----------|
| `parallel` | `run_parallel()` | All agents work independently | Diverse perspectives |
| `sequential` | `run_pipeline()` | Output flows to next agent | Multi-stage analysis |
| `debate` | `run_debate()` | Multi-round argument/critique | Decision making |
| `hierarchical` | `run_hierarchical()` | Leader → workers → synthesis | Complex task breakdown |
| `consensus` | `run_consensus()` | Iterate until agreement | Critical decisions |
| `peer_review` | `run_peer_review()` | Work → review → revise | Code review workflows |

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

### Phase 5: LLM-as-a-Judge Evaluation ✅
- [x] Research CrewAI evaluation patterns
- [x] Implement LLM-based evaluators (Goal, Semantic, Collaboration, Synthesis)
- [x] Create EnsembleEvaluator with multiple judge models
- [x] Role-optimized agent selection based on benchmarks
- [x] Proper judge/agent family separation (no bias)
- [x] Judge size hierarchy (12-14B judges for 3B agents)

### Phase 6: Future Enhancements
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

**Status**: ✅ APEE Framework Complete (Phases 1-5)  
**Tests**: 34 passing  
**Agents**: llama3.2:3b, qwen2.5-coder:3b, granite4:3b (3B diverse families)  
**Judges**: qwen3:14b, gemma3:12b (12-14B evaluation models)  
**Patterns**: 6 (parallel, sequential, debate, hierarchical, consensus, peer_review)  
**Author**: [ahjavid](https://github.com/ahjavid)
