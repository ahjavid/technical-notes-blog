# Adaptive Poly-Agentic Evaluation Ecosystem (APEE)

*A comprehensive framework for evaluating and benchmarking multi-agent AI systems using LLM-as-a-Judge methodology*

---

## 📖 Overview

The Adaptive Poly-Agentic Evaluation Ecosystem (APEE) is a framework for systematically evaluating multi-agent AI systems. It uses **LLM-as-a-Judge** evaluation (inspired by CrewAI) where large language models evaluate agent outputs rather than simple heuristics, providing meaningful, nuanced scores.

### 🎯 Key Features
- **LLM-as-a-Judge Evaluation**: Large models (20-24B) evaluate smaller agent outputs
- **Ensemble Judges**: Multiple judge models from different families reduce bias
- **Poly-Agentic Collaboration**: Multiple agents working together with 6 patterns
- **Three-Tier Metrics**: Individual → Collaborative → Ecosystem evaluation
- **Role-Optimized Agents**: Agent selection based on benchmark strengths
- **6 Collaboration Patterns** (see [PATTERNS.md](apee/coordination/PATTERNS.md)):
  - `run_parallel()` - All agents work independently (best result selected)
  - `run_pipeline()` - Sequential: analyze → code → review
  - `run_debate()` - Multi-round parallel discussion
  - `run_hierarchical()` - Analyst leads, workers execute, leader synthesizes
  - `run_consensus()` - Iterate with semantic agreement detection
  - `run_peer_review()` - Work → review → revise (3 parallel phases)

---

## 🏆 Latest Results (LLM-as-a-Judge)

### Configuration

**Agents** (small, diverse families - ordered for optimal pattern execution):
| Role | Model | Family | Position | Benchmark Strength |
|------|-------|--------|----------|-------------------|
| Analyst (Analyzer) | qwen2.5-coder:3b | Qwen | 1st (Leader) | analysis: 0.964, reasoning: 0.950 |
| Coder (Executor) | llama3.2:3b | Llama | 2nd (Worker) | code_generation: 0.950 |
| Reviewer | granite4:3b | Granite | 3rd (Final) | code_review: 0.935 |

> **Note**: Agent order matters! Analyst is first for hierarchical (planning/synthesis), then coder, then reviewer for logical pipeline flow: analyze → code → review.

**Judges** (large, different families - no overlap with agents):
| Judge | Model | Size | Family |
|-------|-------|------|--------|
| Judge 1 | gpt-oss:20b | 20B | GPT-OSS |
| Judge 2 | mistral-small3.2:24b | 24B | Mistral |

### Multi-Agent Collaborative Evaluation

| Scenario | Pattern | L1 Individual | L2 Collaborative | L3 Ecosystem | Overall |
|----------|---------|---------------|------------------|--------------|---------|
| Research Synthesis | Sequential | 6.2/10 | 6.8/10 | 8.6/10 | **7.3/10** |
| Constrained Problem | Debate | 5.8/10 | 6.5/10 | 8.6/10 | **7.1/10** |
| Creative Collab | Debate | 5.0/10 | 6.8/10 | 8.6/10 | **7.1/10** |
| Realtime Collab | Parallel | 6.3/10 | 6.0/10 | 9.0/10 | **7.1/10** |
| Emergent Behavior | Parallel | 7.0/10 | 5.2/10 | 8.8/10 | **6.8/10** |
| Adversarial Review | Debate | 5.3/10 | 5.8/10 | 7.9/10 | **6.6/10** |
| Conflict Resolution | Consensus | 6.5/10 | 4.8/10 | 8.0/10 | **6.3/10** |
| Collab Code Review | Peer Review | 4.8/10 | 5.5/10 | 7.6/10 | **6.2/10** |
| Knowledge Transfer | Sequential | 6.2/10 | 5.0/10 | 8.2/10 | **6.2/10** |
| Doc Sprint | Peer Review | 5.3/10 | 5.2/10 | 7.7/10 | **6.1/10** |
| Scalability Test | Hierarchical | 5.8/10 | 4.5/10 | 8.0/10 | **6.0/10** |
| Error Recovery | Hierarchical | 4.2/10 | 5.0/10 | 8.0/10 | **5.9/10** |

### Ensemble Judge Agreement

```
Judge Models: gpt-oss:20b, mistral-small3.2:24b
Aggregation: median

Individual Judge Scores (code_review scenario):
  • gpt-oss:20b: Overall=6.52, L1=7.5, L2=6.0
  • mistral-small3.2:24b: Overall=6.82, L1=8.0, L2=6.0

Disagreement Metrics:
  • Overall StdDev: 0.21
  • Overall Range: 0.30
  • High Disagreement: ✅ No
```

### Detailed Metric Breakdown (Code Review Scenario)

```
Level 1 (Individual - per agent):
  • executor: Goal=7.0, Semantic=7.0
  • analyzer: Goal=5.0, Semantic=5.0
  • reviewer: Goal=9.0, Semantic=9.0

Level 2 (Collaborative):
  • Collaboration: 5.0/10
  • Synthesis: 7.0/10

Level 3 (Ecosystem):
  • Efficiency: 1.6/10
  • Stability: 10.0/10
  • Throughput: 10.0/10
  • Adaptability: 6.0/10
```

### Key Insights

1. **Research synthesis leads**: 7.3/10 overall (highest) - sequential pattern excels at structured analysis
2. **Debate and parallel tie for second**: 7.1/10 - constrained_problem, creative_collab, and realtime_collab
3. **Hierarchical struggles**: 6.0/10 average - leader-worker delegation needs improvement
4. **Error recovery lowest**: 5.9/10 - hierarchical pattern underperforms on recovery scenarios
5. **L2 Collaborative is the bottleneck**: Average 5.6/10 vs L1 (5.7/10) and L3 (8.3/10)
6. **L3 Ecosystem strongest**: All scenarios score 7.6-9.0/10 - system-level metrics consistently high
7. **Pattern ranking**: parallel/debate (6.9) > sequential (6.8) > consensus (6.3) > peer_review (6.2) > hierarchical (6.0)
8. **Score range 5.9-7.3**: Mean 6.6/10 across all 12 scenarios

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
  ollama pull gpt-oss:20b
  ollama pull mistral-small3.2:24b
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
# Run basic evaluation (default)
python examples/proper_apee_evaluation.py

# Run with advanced evaluation patterns
python examples/proper_apee_evaluation.py --mode progressive  # Fail-fast with 4 depth levels
python examples/proper_apee_evaluation.py --mode jury         # 4 persona judges (Skeptic, Literalist, Optimist, Pragmatist)
python examples/proper_apee_evaluation.py --mode calibrated   # Calibration + jury combined
python examples/proper_apee_evaluation.py --mode all          # Run all modes sequentially
```

### Advanced Evaluation Modes

| Mode | Pattern | Description | Best For |
|------|---------|-------------|----------|
| `basic` | Standard | Direct LLM-as-a-Judge evaluation | Quick assessment |
| `progressive` | Sequential | 4 depth levels with fail-fast | Large-scale screening |
| `jury` | Independent | 4 personas with distinct lenses | Subjective evaluations |
| `calibrated` | Combined | Calibration loop + jury | Novel tasks, ambiguous requirements |

### Evaluation Mode Comparison (December 2025)

| Mode | Avg Score | Min | Max | Characteristics |
|------|-----------|-----|-----|-----------------|
| Basic | 6.55 | 5.9 | 7.3 | Fastest, baseline scores |
| Progressive | 6.55 | 5.9 | 7.2 | Similar to basic, fail-fast efficiency |
| Jury | 6.29 | 5.1 | 7.1 | More critical, persona diversity |
| Calibrated | 6.25 | 5.5 | 6.8 | Most conservative, narrower range |

### Basic Usage

```python
import asyncio
from apee import OllamaAgent, Coordinator, Task, AgentRole
from apee.evaluation.llm_evaluator import EnsembleEvaluator

async def main():
    # Create role-optimized agents (order matters for patterns!)
    # Analyst first (leader for hierarchical), then coder, then reviewer
    agents = [
        OllamaAgent("analyst", AgentRole.ANALYZER, model="qwen2.5-coder:3b"),  # Leader
        OllamaAgent("coder", AgentRole.EXECUTOR, model="llama3.2:3b"),          # Worker
        OllamaAgent("reviewer", AgentRole.REVIEWER, model="granite4:3b"),       # Final
    ]
    
    # Create ensemble evaluator (large judges, different families)
    evaluator = EnsembleEvaluator(
        judge_models=["gpt-oss:20b", "mistral-small3.2:24b"],
        aggregation="median",
    )
    
    # Coordinate and evaluate
    coordinator = Coordinator(agents=agents)
    task = Task(task_id="t1", description="Review this code for bugs")
    
    # Pipeline flows: analyst → coder → reviewer (analyze → code → review)
    results = await coordinator.run_pipeline(task, ["analyst", "coder", "reviewer"])
    
    # Or use hierarchical: analyst leads, coder+reviewer work, analyst synthesizes
    # results = await coordinator.run_hierarchical(task, leader_id="analyst")
    
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
├── cli.py                   # Command-line interface
├── agents/
│   ├── base.py              # Abstract Agent class
│   └── ollama.py            # Ollama LLM implementation
├── coordination/
│   ├── coordinator.py       # Task distribution & execution modes
│   └── PATTERNS.md          # 📚 Detailed pattern documentation
├── evaluation/
│   ├── evaluator.py         # Heuristic evaluation engine
│   ├── llm_evaluator.py     # LLM-as-a-Judge evaluators
│   ├── quality.py           # Quality scoring strategies
│   ├── adaptive.py          # Adaptive pattern detection
│   └── report.py            # Report data models
├── benchmarks/
│   ├── datasets.py          # 19 scenarios, 11 categories
│   ├── runner.py            # Statistical benchmark runner
│   ├── analyzer.py          # Analysis with confidence intervals
│   └── collaborative.py     # 12 multi-agent scenarios
├── visualization/           # 🆕 Phase 6
│   ├── charts.py            # Plotly/text chart generation
│   └── export.py            # HTML/PNG export utilities
├── anomaly/                 # 🆕 Phase 6
│   ├── detector.py          # Statistical anomaly detection
│   ├── patterns.py          # Pattern analyzers
│   └── alerts.py            # Alert handling system
├── dashboard/               # 🆕 Phase 6
│   ├── server.py            # Web dashboard server
│   └── api.py               # Dashboard API client
└── utils/
    ├── logging.py           # Logging utilities
    └── helpers.py           # Helper functions

examples/
├── full_evaluation.py           # Basic evaluation demo
├── comprehensive_benchmark.py   # Single-model benchmarks
├── multi_agent_evaluation.py    # Multi-agent with heuristics
├── proper_apee_evaluation.py    # LLM-as-a-Judge evaluation
└── phase6_demo.py               # 🆕 Visualization & anomaly demo

tests/
├── test_benchmarks.py           # Benchmark tests
├── test_models.py               # Model tests
├── test_quality.py              # Quality scoring tests
└── test_coordinator.py          # 📋 Coordinator pattern tests (35 tests)
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     APEE Framework                                  │
├─────────────────────────────────────────────────────────────────────┤
│  AGENTS (Small 3B models - ordered for optimal pattern execution)   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │   Analyst    │  │    Coder     │  │   Reviewer   │               │
│  │  (Analyzer)  │  │  (Executor)  │  │  (Reviewer)  │               │
│  │qwen2.5-coder │  │ llama3.2:3b  │  │ granite4:3b  │               │
│  │  [LEADER]    │  │  [WORKER]    │  │   [FINAL]    │               │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘               │
│         │                 │                 │                       │
│         └────────────┬────┴─────────────────┘                       │
│                      │                                              │
│         ┌────────────▼─────────────┐                                │
│         │      Coordinator         │                                │
│         │  • run_parallel()        │ ← Best quality result          │
│         │  • run_pipeline()        │ ← analyze → code → review      │
│         │  • run_debate()          │ ← Multi-round parallel         │
│         │  • run_hierarchical()    │ ← Analyst leads workers        │
│         │  • run_consensus()       │ ← Semantic agreement detect    │
│         │  • run_peer_review()     │ ← 3 parallel phases            │
│         └────────────┬─────────────┘                                │
│                      │                                              │
│  JUDGES (Large 20-24B models - different families)                  │
│  ┌───────────────────────────────────────────────┐                  │
│  │  ┌──────────┐          ┌──────────┐           │                  │
│  │  │ GPT-OSS  │    +     │ Mistral  │           │                  │
│  │  │   20B    │          │   24B    │           │                  │
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
    judge_models=["gpt-oss:20b", "mistral-small3.2:24b"],
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

#### Judge Models (Large, 20-24B+)

| Model              | Family     | Params | Purpose                        |
|--------------------|------------|--------|--------------------------------|
| gpt-oss:20b        | GPT-OSS    | 20B    | Primary judge, deep reasoning  |
| mistral-small3.2:24b | Mistral  | 24B    | Secondary judge, clear analysis |

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

### Phase 6: Future Enhancements ✅
- [x] Create visualization utilities (charts, reports, exports)
- [x] Add more collaboration scenarios (12 scenarios total)
- [x] Implement advanced anomaly detection
- [x] Web dashboard for results
- [ ] Publish to PyPI (optional)

---

## 🖥️ Phase 6 Features

### Visualization Utilities

Create interactive charts and comprehensive reports:

```python
from apee.visualization import (
    MetricsVisualizer,
    create_evaluation_chart,
    create_comparison_chart,
    create_anomaly_heatmap,
    generate_report_html,
)

# Create visualizer
visualizer = MetricsVisualizer()

# Three-tier comparison chart
chart = visualizer.create_level_comparison(
    l1_scores={"goal": 8.5, "semantic": 7.0},
    l2_scores={"collaboration": 6.5, "synthesis": 7.5},
    l3_scores={"efficiency": 8.0, "stability": 9.0},
)

# Generate HTML report
generate_report_html(
    evaluation_result,
    title="My Evaluation Report",
    output_path="report.html"
)
```

### Advanced Anomaly Detection

Detect unusual patterns in evaluations:

```python
from apee.anomaly import (
    AnomalyDetector,
    AlertManager,
    ConsoleAlertHandler,
)

# Create detector and alert manager
detector = AnomalyDetector(window_size=50, z_threshold=3.0)
alerts = AlertManager()
alerts.add_handler(ConsoleAlertHandler())

# Check evaluation for anomalies
anomalies = detector.check_evaluation({
    "overall_apee_score": 3.0,  # Very low
    "l2_average": 1.0,  # Poor collaboration
})

# Process and display alerts
for anomaly in anomalies:
    alerts.process_anomaly(anomaly)
```

### Web Dashboard

Real-time monitoring of evaluations:

```python
from apee import create_dashboard, DashboardAPI

# Start dashboard server
dashboard = create_dashboard(port=8765)
# Opens browser to http://localhost:8765

# Push results programmatically
dashboard.add_evaluation({
    "overall_apee_score": 7.5,
    "scenario_id": "code_review",
    # ...
})

# Or use CLI
# $ apee-dashboard --port 8765
```

### New Collaboration Scenarios

12 scenarios covering diverse collaboration patterns:

| ID | Name | Pattern | Focus |
|----|------|---------|-------|
| 1 | Code Review | peer_review | Quality, coordination |
| 2 | Research Synthesis | sequential | Information flow |
| 3 | Constrained Problem | debate | Conflict resolution |
| 4 | Emergent Behavior | parallel | Diversity, consensus |
| 5 | Scalability Test | hierarchical | Coordination overhead |
| 6 | Conflict Resolution | consensus | Agreement time |
| 7 | Knowledge Transfer | sequential | Domain translation |
| 8 | Error Recovery | hierarchical | Fault tolerance |
| 9 | Creative Collaboration | debate | Idea synthesis |
| 10 | Real-Time Incident | parallel | Response time |
| 11 | Adversarial Review | debate | Security analysis |
| 12 | Documentation Sprint | peer_review | Consistency |

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
**Tests**: 69 passing  
**Agents**: llama3.2:3b, qwen2.5-coder:3b, granite4:3b (3B diverse families)  
**Judges**: gpt-oss:20b, mistral-small3.2:24b (20-24B evaluation models)  
**Patterns**: 6 (parallel, sequential, debate, hierarchical, consensus, peer_review)  
**Author**: [ahjavid](https://github.com/ahjavid)
