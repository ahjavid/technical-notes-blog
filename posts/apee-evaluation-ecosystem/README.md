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

### 🔬 Evaluation Configuration

#### Agent Pool (Executors)
**Small, diverse models (3B parameters) - Role-optimized based on benchmark strengths**

| Role | Model | Family | Size | Strength | Score |
|------|-------|--------|------|----------|-------|
| **Coder** (Executor) | llama3.2:3b | Llama | 3B | Code Generation | 0.950 |
| **Analyst** (Analyzer) | qwen2.5-coder:3b | Qwen | 3B | Code Analysis | 0.964 |
| **Reviewer** (Critic) | granite4:3b | Granite | 3B | Code Review | 0.935 |

#### Judge Pool (Evaluators)
**Large models (20-24B parameters) - Different families to avoid bias**

| Judge | Model | Family | Size | Purpose |
|-------|-------|--------|------|---------|
| **Judge 1** | gpt-oss:20b | GPT-OSS | 20B | Primary evaluator - reasoning focus |
| **Judge 2** | mistral-small3.2:24b | Mistral | 24B | Secondary evaluator - quality focus |

**Aggregation Method:** Median (robust to outliers)  
**Judge Agreement:** StdDev=0.21, Range=0.30 ✅ **Excellent consensus**

---

### 📊 Performance Overview

#### Overall Scores by Collaboration Pattern

| Pattern | Scenarios | Avg Score | Best | Worst | Interpretation |
|---------|-----------|-----------|------|-------|----------------|
| 🥇 **Parallel** | 2 | **6.5/10** | 7.5 | 5.5 | Best for coordinated parallel work |
| 🥈 **Consensus** | 1 | **7.3/10** | 7.3 | 7.3 | Strong agreement-based collaboration |
| 🥉 **Debate** | 3 | **6.2/10** | 7.1 | 5.3 | Variable - needs constructive conflict |
| **Sequential** | 2 | **6.8/10** | 6.8 | 6.8 | Consistent pipeline performance |
| **Hierarchical** | 2 | **6.0/10** | 6.0 | 6.0 | Delegation works moderately well |
| **Peer Review** | 2 | **6.3/10** | 6.7 | 5.8 | Review cycles add value |

#### Top 5 Best-Performing Scenarios

| Rank | Scenario | Pattern | Overall | L1 | L2 | L3 | Why It Works |
|------|----------|---------|---------|----|----|----|--------------||
| 🥇 | **Realtime Collaboration** | parallel | **7.5/10** | 7.2 | 6.8 | 9.2 | Coordinated parallel work with high ecosystem efficiency |
| 🥈 | **Conflict Resolution** | consensus | **7.3/10** | 7.2 | 6.8 | 8.5 | Agents iterate until agreement - strong synthesis |
| 🥉 | **Adversarial Review** | debate | **7.1/10** | 6.9 | 6.8 | 7.8 | Constructive conflict improves output quality |
| 4 | **Research Synthesis** | sequential | **6.8/10** | 7.6 | 5.8 | 7.8 | Good individual work, pipeline flow effective |
| 5 | **Knowledge Transfer** | sequential | **6.8/10** | 6.5 | 6.0 | 8.4 | Sequential handoff preserves context |

#### Bottom 3 Struggling Scenarios

| Rank | Scenario | Pattern | Overall | L1 | L2 | L3 | Why It Struggles |
|------|----------|---------|---------|----|----|----|-----------------||
| 🔴 | **Constrained Problem** | debate | **5.3/10** | 6.4 | 3.8 | 6.8 | Debate fails under constraints - weak collaboration |
| ⚠️ | **Emergent Behavior** | parallel | **5.5/10** | 7.0 | 3.2 | 7.7 | Independent work lacks coordination |
| ⚠️ | **Doc Sprint** | peer_review | **5.8/10** | 6.4 | 4.8 | 6.9 | Review cycles slow documentation tasks |

---

### 📈 Detailed Results by Evaluation Level

#### Level 1: Individual Agent Performance
*How well each agent achieves its assigned task (goal alignment, semantic quality)*

| Scenario | Executor | Analyzer | Reviewer | Avg | Insight |
|----------|----------|----------|----------|-----|---------|
| collab_code_review | 7.0 | 5.0 | **9.0** | 7.0 | Reviewer excels at critique |
| research_synthesis | **8.0** | 7.0 | 8.0 | 7.7 | All agents perform well |
| constrained_problem | 6.0 | 6.5 | 6.5 | 6.3 | Constraints limit all agents |
| realtime_collab | 7.5 | 7.0 | **7.0** | 7.2 | Balanced team performance |

**Best Individual Agent:** Reviewer (granite4:3b) - Goal=9.0, Semantic=9.0 in code review

#### Level 2: Collaborative Effectiveness
*How well agents work together (collaboration quality, synthesis coherence)*

| Pattern | Best L2 | Worst L2 | Avg L2 | Observation |
|---------|---------|----------|--------|-------------|
| Consensus | **6.8** | 6.8 | 6.8 | Agreement-based = strong collaboration |
| Peer Review | **6.0** | 4.8 | 5.4 | Review helps but varies by task |
| Sequential | **6.0** | 5.8 | 5.9 | Pipeline handoffs work consistently |
| Debate | **6.8** | 3.8 | 5.5 | High variance - depends on conflict type |
| Hierarchical | **5.0** | 4.0 | 4.5 | Delegation creates coordination overhead |
| Parallel | **6.8** | 3.2 | 5.0 | Independent work = weak collaboration scores |

**Key Finding:** Patterns requiring explicit coordination (consensus, debate) score higher on L2 than independent work patterns (parallel, hierarchical)

#### Level 3: Ecosystem Performance
*System-level metrics (efficiency, stability, throughput, adaptability)*

**Example Breakdown (collab_code_review scenario):**
- Efficiency: 1.6/10 - Multi-agent communication overhead
- Stability: 10.0/10 - No crashes or failures
- Throughput: 10.0/10 - Task completed successfully
- Adaptability: 6.0/10 - Good pattern adaptation

**Observed L3 Scores Across Scenarios:**
| L3 Score | Scenarios | Observation |
|----------|-----------|-------------|
| **9.2/10** | realtime_collab | Highest ecosystem performance |
| **8.3-8.5/10** | scalability_test, conflict_resolution, knowledge_transfer, error_recovery | Strong ecosystem metrics |
| **7.5-7.8/10** | research_synthesis, adversarial_review, creative_collab, emergent_behavior | Good ecosystem performance |
| **6.8-6.9/10** | collab_code_review, constrained_problem, doc_sprint | Moderate ecosystem performance |

**Key Finding:** L3 scores remain high (6.8-9.2) across all scenarios despite likely low efficiency scores, suggesting that perfect stability and throughput compensate for multi-agent overhead

---

### 🔍 Complete Scenario Results

| # | Scenario | Pattern | L1 (Individual) | L2 (Collab) | L3 (Ecosystem) | Overall | Grade |
|---|----------|---------|----------------|-------------|----------------|---------|-------|
| 1 | Collaborative Code Review | peer_review | 7.5/10 | 6.0/10 | 6.9/10 | **6.7/10** | B- |
| 2 | Research Synthesis | sequential | 7.6/10 | 5.8/10 | 7.8/10 | **6.8/10** | B- |
| 3 | Constrained Problem Solving | debate | 6.4/10 | 3.8/10 | 6.8/10 | **5.3/10** | D+ |
| 4 | Emergent Behavior Detection | parallel | 7.0/10 | 3.2/10 | 7.7/10 | **5.5/10** | D+ |
| 5 | Agent Scalability Test | hierarchical | 6.9/10 | 4.0/10 | 8.3/10 | **6.0/10** | C |
| 6 | Conflict Resolution | consensus | 7.2/10 | 6.8/10 | 8.5/10 | **7.3/10** | B |
| 7 | Cross-Domain Knowledge Transfer | sequential | 6.5/10 | 6.0/10 | 8.4/10 | **6.8/10** | B- |
| 8 | Collaborative Error Recovery | hierarchical | 5.6/10 | 5.0/10 | 8.4/10 | **6.0/10** | C |
| 9 | Creative Collaborative Design | debate | 5.6/10 | 5.8/10 | 7.5/10 | **6.2/10** | C |
| 10 | Realtime Incident Response | parallel | 7.2/10 | 6.8/10 | 9.2/10 | **7.5/10** | B+ |
| 11 | Adversarial Code Review | debate | 6.9/10 | 6.8/10 | 7.8/10 | **7.1/10** | B |
| 12 | Documentation Sprint | peer_review | 6.4/10 | 4.8/10 | 6.9/10 | **5.8/10** | D+ |

**Overall Average:** 6.3/10 (C+)  
**Score Range:** 5.3 - 7.5 (2.2 point spread)  
**Most Consistent:** Sequential pattern (both 6.8/10)  
**Most Variable:** Parallel pattern (5.5 to 7.5)

---

### 💡 Key Insights & Recommendations

#### ✅ What Works Well

1. **Realtime Coordination** (7.5/10) - When parallel agents coordinate in real-time, they achieve the highest scores. The ecosystem efficiency reaches 9.2/10.

2. **Consensus-Building** (7.3/10) - Iterating until agents agree produces strong collaboration (L2=6.8) and ecosystem performance (L3=8.5).

3. **Constructive Conflict** (7.1/10) - Adversarial review with constructive debate improves quality without sacrificing collaboration.

4. **Reviewer Agent Dominance** - granite4:3b consistently achieves the highest individual scores (Goal=9.0, Semantic=9.0), especially in critique tasks.

5. **Perfect Reliability** - All 12 scenarios completed with 10/10 stability and throughput. Zero crashes or failures.

#### ⚠️ What Needs Improvement

1. **Debate Under Constraints** (5.3/10) - The debate pattern fails when agents face constraints (L2=3.8). Agents argue rather than collaborate.

2. **Independent Parallel Work** (5.5/10) - When parallel agents work independently without coordination, collaboration scores drop dramatically (L2=3.2).

3. **Multi-Agent Efficiency** (~1.6-2.5/10) - All patterns suffer from low efficiency due to multi-agent communication overhead. This is a systemic issue.

4. **Documentation Tasks** (5.8/10) - Peer review cycles slow down documentation sprints. Simpler tasks may not benefit from multi-agent approaches.

5. **Hierarchical Delegation** (6.0/10 avg) - Leader-worker patterns create coordination overhead (L2=4.0-5.0) without clear benefits.

#### 🎯 Recommendations

| Pattern | When to Use | When to Avoid |
|---------|-------------|---------------|
| **Parallel** | Time-sensitive tasks with clear coordination | Independent subtasks without communication |
| **Consensus** | High-stakes decisions requiring agreement | Time-constrained tasks |
| **Debate** | Complex problems benefiting from multiple perspectives | Constrained problems with limited solution space |
| **Sequential** | Tasks with natural pipeline structure | Highly interactive or iterative problems |
| **Hierarchical** | Large-scale delegation with clear leader | Small teams or tasks requiring peer collaboration |
| **Peer Review** | Quality-critical outputs needing validation | Simple documentation or routine tasks |

#### 🔬 Judge Reliability Analysis

**Ensemble Agreement:** Excellent (StdDev=0.21, Range=0.30)

| Judge | Model | Avg Score | Bias | Reliability |
|-------|-------|-----------|------|-------------|
| Judge 1 | gpt-oss:20b | 6.52/10 | -0.19 (stricter) | High |
| Judge 2 | mistral-small3.2:24b | 6.82/10 | +0.11 (lenient) | High |

**Inter-Judge Agreement:** 0.30 point average difference (excellent for ensemble evaluation)  
**No High Disagreement Cases:** Both judges aligned on scenario quality ✅

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
        judge_models=["gpt-oss:20b", "mistral-small3.2:24b"],
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
├── cli.py                   # Command-line interface
├── agents/
│   ├── base.py              # Abstract Agent class
│   └── ollama.py            # Ollama LLM implementation
├── coordination/
│   └── coordinator.py       # Task distribution & execution modes
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
**Tests**: 34 passing  
**Agents**: llama3.2:3b, qwen2.5-coder:3b, granite4:3b (3B diverse families)  
**Judges**: qwen3:14b, gemma3:12b (12-14B evaluation models)  
**Patterns**: 6 (parallel, sequential, debate, hierarchical, consensus, peer_review)  
**Author**: [ahjavid](https://github.com/ahjavid)
