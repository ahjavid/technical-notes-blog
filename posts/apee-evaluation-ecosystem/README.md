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

## 🏆 Latest Results (LLM-as-a-Judge) - December 2025

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

---

### 📊 Evaluation Mode Summary

| Mode | Avg | Min | Max | L1 Avg | L2 Avg | L3 Avg | Characteristics |
|------|-----|-----|-----|--------|--------|--------|-----------------|
| **Basic** | 6.55 | 5.9 | 7.3 | 5.7 | 5.6 | 8.3 | Baseline reference |
| **Progressive** | 6.55 | 5.9 | 7.2 | 5.9 | 5.5 | 8.2 | Fail-fast efficiency |
| **Jury** | 6.29 | 5.1 | 7.1 | 5.7 | 5.2 | 7.9 | More critical, higher variance |
| **Calibrated** | 6.25 | 5.5 | 6.8 | 5.8 | 5.1 | 7.8 | Most conservative, narrow range |

---

### 🔵 Basic Mode Results

Standard LLM-as-a-Judge evaluation with ensemble median aggregation.

| Scenario | Pattern | L1 | L2 | L3 | Overall |
|----------|---------|-----|-----|-----|---------|
| Research Synthesis | Sequential | 6.2 | 6.8 | 8.6 | **7.3** |
| Constrained Problem | Debate | 5.8 | 6.5 | 8.6 | **7.1** |
| Creative Collab | Debate | 5.0 | 6.8 | 8.6 | **7.1** |
| Realtime Collab | Parallel | 6.3 | 6.0 | 9.0 | **7.1** |
| Emergent Behavior | Parallel | 7.0 | 5.2 | 8.8 | **6.8** |
| Adversarial Review | Debate | 5.3 | 5.8 | 7.9 | **6.6** |
| Conflict Resolution | Consensus | 6.5 | 4.8 | 8.0 | **6.3** |
| Collab Code Review | Peer Review | 4.8 | 5.5 | 7.6 | **6.2** |
| Knowledge Transfer | Sequential | 6.2 | 5.0 | 8.2 | **6.2** |
| Doc Sprint | Peer Review | 5.3 | 5.2 | 7.7 | **6.1** |
| Scalability Test | Hierarchical | 5.8 | 4.5 | 8.0 | **6.0** |
| Error Recovery | Hierarchical | 4.2 | 5.0 | 8.0 | **5.9** |

**Basic Mode Insights:**
- Research synthesis leads at 7.3 (sequential pattern excels)
- Debate and parallel patterns tie at 7.1
- Hierarchical pattern struggles (6.0 avg)

---

### 🟢 Progressive Mode Results

Sequential evaluation with 4 depth levels (SURFACE → TECHNICAL → COLLABORATIVE → COMPREHENSIVE) and fail-fast.

| Scenario | Pattern | L1 | L2 | L3 | Overall |
|----------|---------|-----|-----|-----|---------|
| Adversarial Review | Debate | 7.0 | 6.2 | 8.4 | **7.2** |
| Knowledge Transfer | Sequential | 5.8 | 6.8 | 8.5 | **7.1** |
| Realtime Collab | Parallel | 5.8 | 6.2 | 9.1 | **7.1** |
| Creative Collab | Debate | 4.8 | 6.8 | 7.9 | **6.9** |
| Research Synthesis | Sequential | 6.7 | 5.8 | 8.3 | **6.8** |
| Conflict Resolution | Consensus | 6.3 | 5.5 | 8.3 | **6.6** |
| Scalability Test | Hierarchical | 5.9 | 5.2 | 8.2 | **6.4** |
| Error Recovery | Hierarchical | 5.3 | 5.2 | 8.2 | **6.2** |
| Collab Code Review | Peer Review | 4.8 | 5.2 | 7.5 | **6.2** |
| Doc Sprint | Peer Review | 5.7 | 5.0 | 7.4 | **6.1** |
| Emergent Behavior | Parallel | 6.3 | 4.0 | 8.5 | **6.0** |
| Constrained Problem | Debate | 5.8 | 4.2 | 7.9 | **5.9** |

**Progressive Mode Insights:**
- Adversarial review jumps to top (7.2) - depth levels reveal hidden quality
- Knowledge transfer improves significantly (6.2 → 7.1)
- Constrained problem drops (7.1 → 5.9) - fails at deeper evaluation levels

---

### 🟡 Jury Mode Results

4 persona judges (SKEPTIC, LITERALIST, OPTIMIST, PRAGMATIST) with weighted voting.

| Scenario | Pattern | L1 | L2 | L3 | Overall |
|----------|---------|-----|-----|-----|---------|
| Research Synthesis | Sequential | 7.3 | 6.0 | 8.4 | **7.1** |
| Realtime Collab | Parallel | 5.8 | 5.8 | 8.8 | **6.9** |
| Creative Collab | Debate | 4.5 | 6.8 | 7.9 | **6.8** |
| Emergent Behavior | Parallel | 5.2 | 5.8 | 8.9 | **6.7** |
| Adversarial Review | Debate | 6.8 | 5.5 | 7.5 | **6.6** |
| Error Recovery | Hierarchical | 4.3 | 6.0 | 8.6 | **6.5** |
| Collab Code Review | Peer Review | 5.8 | 5.8 | 7.1 | **6.4** |
| Scalability Test | Hierarchical | 6.3 | 4.8 | 8.1 | **6.2** |
| Constrained Problem | Debate | 5.2 | 5.2 | 7.5 | **6.1** |
| Doc Sprint | Peer Review | 5.0 | 5.0 | 7.0 | **5.8** |
| Knowledge Transfer | Sequential | 6.0 | 3.0 | 7.5 | **5.2** |
| Conflict Resolution | Consensus | 5.5 | 3.2 | 7.0 | **5.1** |

**Jury Mode Insights:**
- More critical overall (avg 6.29 vs 6.55 basic)
- Consensus pattern penalized heavily (5.1) - SKEPTIC finds flaws
- Knowledge transfer drops (7.1 progressive → 5.2 jury) - LITERALIST strict

---

### 🔴 Calibrated Mode Results

Calibration loop + jury combined - judges negotiate rubric before scoring.

| Scenario | Pattern | L1 | L2 | L3 | Overall |
|----------|---------|-----|-----|-----|---------|
| Research Synthesis | Sequential | 6.8 | 5.5 | 8.2 | **6.8** |
| Knowledge Transfer | Sequential | 6.7 | 5.8 | 8.2 | **6.7** |
| Adversarial Review | Debate | 6.2 | 6.0 | 7.6 | **6.7** |
| Creative Collab | Debate | 5.7 | 6.0 | 7.8 | **6.6** |
| Collab Code Review | Peer Review | 7.2 | 5.3 | 7.2 | **6.5** |
| Error Recovery | Hierarchical | 6.0 | 5.0 | 8.1 | **6.3** |
| Conflict Resolution | Consensus | 6.8 | 4.8 | 7.4 | **6.2** |
| Realtime Collab | Parallel | 4.5 | 5.0 | 8.5 | **6.2** |
| Emergent Behavior | Parallel | 5.5 | 4.2 | 8.6 | **6.0** |
| Scalability Test | Hierarchical | 4.5 | 4.8 | 8.0 | **5.8** |
| Doc Sprint | Peer Review | 4.3 | 5.0 | 7.0 | **5.8** |
| Constrained Problem | Debate | 5.0 | 4.0 | 7.2 | **5.5** |

**Calibrated Mode Insights:**
- Narrowest score range (5.5-6.8) - calibration produces consistency
- Most conservative scores (avg 6.25)
- Sequential pattern most stable (research_synthesis and knowledge_transfer both 6.7+)

---

### 📈 Cross-Mode Comparison by Scenario

| Scenario | Basic | Progressive | Jury | Calibrated | Variance |
|----------|-------|-------------|------|------------|----------|
| research_synthesis | 7.3 | 6.8 | 7.1 | 6.8 | Low |
| adversarial_review | 6.6 | **7.2** | 6.6 | 6.7 | Low |
| realtime_collab | 7.1 | 7.1 | 6.9 | 6.2 | Medium |
| creative_collab | 7.1 | 6.9 | 6.8 | 6.6 | Low |
| knowledge_transfer | 6.2 | **7.1** | 5.2 | 6.7 | **High** |
| conflict_resolution | 6.3 | 6.6 | **5.1** | 6.2 | **High** |
| constrained_problem | **7.1** | 5.9 | 6.1 | **5.5** | **High** |
| emergent_behavior | 6.8 | 6.0 | 6.7 | 6.0 | Medium |
| error_recovery | 5.9 | 6.2 | 6.5 | 6.3 | Low |
| collab_code_review | 6.2 | 6.2 | 6.4 | 6.5 | Low |
| scalability_test | 6.0 | 6.4 | 6.2 | 5.8 | Low |
| doc_sprint | 6.1 | 6.1 | 5.8 | 5.8 | Low |

**High-Variance Scenarios** (modes disagree):
- `knowledge_transfer`: 5.2 (jury) to 7.1 (progressive) - evaluation uncertainty
- `conflict_resolution`: 5.1 (jury) to 6.6 (progressive) - persona sensitivity
- `constrained_problem`: 5.5 (calibrated) to 7.1 (basic) - calibration reveals issues

---

### Key Insights

1. **Research synthesis consistently strong**: 6.8-7.3 across all modes - sequential pattern reliable
2. **Mode selection matters**: Up to 2.0 point difference between modes for same scenario
3. **L2 Collaborative is the universal bottleneck**: 5.1-5.6 avg across all modes
4. **L3 Ecosystem most stable**: 7.8-8.3 avg across modes
5. **Jury mode most critical**: Lowest average (6.29), identifies weaknesses
6. **Calibrated mode most conservative**: Narrowest range, consistent scoring
7. **Progressive good for screening**: Similar to basic but with fail-fast efficiency
8. **High-variance scenarios need multiple modes**: knowledge_transfer, conflict_resolution, constrained_problem

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
│   ├── __init__.py
│   ├── base.py              # Abstract Agent class
│   └── ollama.py            # Ollama LLM implementation
├── coordination/
│   ├── __init__.py
│   ├── coordinator.py       # Task distribution & 6 execution patterns
│   └── PATTERNS.md          # 📚 Detailed pattern documentation
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py         # Heuristic evaluation engine
│   ├── llm_evaluator.py     # LLM-as-a-Judge evaluators (EnsembleEvaluator)
│   ├── advanced_patterns.py # 🆕 Progressive, Jury, Calibrated modes
│   ├── quality.py           # Quality scoring strategies
│   ├── adaptive.py          # Adaptive pattern detection
│   └── report.py            # Report data models
├── benchmarks/
│   ├── __init__.py
│   ├── datasets.py          # 19 scenarios, 11 categories
│   ├── runner.py            # Statistical benchmark runner
│   ├── analyzer.py          # Analysis with confidence intervals
│   └── collaborative.py     # 12 multi-agent scenarios
├── visualization/
│   ├── __init__.py
│   ├── charts.py            # Plotly/text chart generation
│   └── export.py            # HTML/PNG export utilities
├── anomaly/
│   ├── __init__.py
│   ├── detector.py          # Statistical anomaly detection
│   ├── patterns.py          # Pattern analyzers
│   └── alerts.py            # Alert handling system
├── dashboard/
│   ├── __init__.py
│   ├── server.py            # Web dashboard server
│   └── api.py               # Dashboard API client
└── utils/
    ├── __init__.py
    ├── logging.py           # Logging utilities
    └── helpers.py           # Helper functions

examples/
├── __init__.py
├── proper_apee_evaluation.py    # 🔥 Main evaluation (--mode basic|progressive|jury|calibrated|all)
├── advanced_evaluation_demo.py  # 🆕 Advanced patterns demo
├── full_evaluation.py           # Basic evaluation demo
├── comprehensive_benchmark.py   # Single-model benchmarks
├── multi_agent_evaluation.py    # Multi-agent with heuristics
├── multi_model_evaluation.py    # Multi-model comparison
└── phase6_demo.py               # Visualization & anomaly demo

tests/
├── __init__.py
├── test_advanced_patterns.py    # 🆕 Advanced patterns tests (38 tests)
├── test_coordinator.py          # Coordinator pattern tests (35 tests)
├── test_benchmarks.py           # Benchmark tests
├── test_models.py               # Model tests
└── test_quality.py              # Quality scoring tests

data/
├── apee_evaluation_results.json           # Basic mode results
├── apee_evaluation_results_progressive.json # Progressive mode results
├── apee_evaluation_results_jury.json      # Jury mode results
├── apee_evaluation_results_calibrated.json # Calibrated mode results
└── evaluation_report.html                 # HTML report
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

### Phase 6: Visualization & Dashboard ✅
- [x] Create visualization utilities (charts, reports, exports)
- [x] Add more collaboration scenarios (12 scenarios total)
- [x] Implement advanced anomaly detection
- [x] Web dashboard for results

### Phase 7: Advanced Evaluation Patterns ✅
- [x] Implement Progressive Deepening (4 depth levels with fail-fast)
- [x] Implement Jury with Personas (SKEPTIC, LITERALIST, OPTIMIST, PRAGMATIST)
- [x] Implement Calibration Loop (judges negotiate rubric)
- [x] Implement Calibrated Jury (combined pattern)
- [x] Add `--mode` CLI argument (basic, progressive, jury, calibrated, all)
- [x] Generate JSON results for all 4 evaluation modes
- [x] Write unit tests for advanced patterns (38 tests)
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

# Run all tests (107 total)
pytest tests/ -v

# Run specific test modules
pytest tests/test_advanced_patterns.py -v  # 38 tests - advanced evaluation patterns
pytest tests/test_coordinator.py -v        # 35 tests - coordination patterns
pytest tests/test_quality.py -v            # Quality scoring tests
pytest tests/test_benchmarks.py -v         # Benchmark tests
pytest tests/test_models.py -v             # Model tests

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
