# Multi-Agent Coordination Patterns

This document explains the coordination patterns available in APEE's `Coordinator` class, their implementation details, and guidance on when to use each pattern.

## Table of Contents

1. [Overview](#overview)
2. [Pattern Comparison](#pattern-comparison)
3. [Detailed Pattern Documentation](#detailed-pattern-documentation)
   - [Parallel](#1-parallel-pattern)
   - [Pipeline](#2-pipeline-pattern)
   - [Debate](#3-debate-pattern)
   - [Hierarchical](#4-hierarchical-pattern)
   - [Consensus](#5-consensus-pattern)
   - [Peer Review](#6-peer-review-pattern)
4. [Implementation Architecture](#implementation-architecture)
5. [Configuration Options](#configuration-options)
6. [Best Practices](#best-practices)

---

## Overview

The Coordinator supports six distinct multi-agent coordination patterns, each designed for different collaboration scenarios:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     APEE Coordination Patterns                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Independent          Sequential           Iterative          Hybrid         │
│  ───────────          ──────────           ─────────          ──────         │
│  • Parallel           • Pipeline           • Debate           • Hierarchical │
│                                            • Consensus        • Peer Review  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Pattern Comparison

| Pattern | Execution Model | Agent Awareness | Rounds | Best For |
|---------|-----------------|-----------------|--------|----------|
| **Parallel** | All at once | None | 1 | Diverse perspectives |
| **Pipeline** | Sequential | Previous outputs | 1 | Multi-stage workflows |
| **Debate** | Parallel per round | All previous | N | Exploring trade-offs |
| **Hierarchical** | Plan → Parallel → Synthesize | Leader ↔ Workers | 3 phases | Complex decomposition |
| **Consensus** | Parallel per round + early exit | All previous | 1-N | Critical decisions |
| **Peer Review** | 3 parallel phases | Assigned reviewer | 3 phases | Quality assurance |

### Latency vs. Quality Trade-off

```
        High Quality
             ▲
             │    ┌─────────────┐
             │    │  Consensus  │
             │    └─────────────┘
             │         ┌─────────────┐
             │         │ Peer Review │
             │         └─────────────┘
             │    ┌──────────┐
             │    │  Debate  │
             │    └──────────┘
             │              ┌──────────────┐
             │              │ Hierarchical │
             │              └──────────────┘
             │    ┌──────────┐
             │    │ Pipeline │
             │    └──────────┘
             │                        ┌──────────┐
             │                        │ Parallel │
             │                        └──────────┘
             └────────────────────────────────────────► Low Latency
```

---

## Detailed Pattern Documentation

### 1. Parallel Pattern

**Method:** `run_parallel(task: Task) -> list[AgentResult]`

#### How It Works

```
                    ┌─────────┐
                    │  Task   │
                    └────┬────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
    ┌─────────┐    ┌─────────┐    ┌─────────┐
    │ Agent A │    │ Agent B │    │ Agent C │
    └────┬────┘    └────┬────┘    └────┬────┘
         │               │               │
         ▼               ▼               ▼
    ┌─────────┐    ┌─────────┐    ┌─────────┐
    │Result A │    │Result B │    │Result C │
    └─────────┘    └─────────┘    └─────────┘
```

All agents receive the **exact same task** and work **independently** with no knowledge of each other's responses.

#### Implementation Details

```python
results = await asyncio.gather(*[
    agent.execute(task) for agent in self.agents.values()
], return_exceptions=True)
```

- Uses `asyncio.gather()` for true parallel execution
- `return_exceptions=True` prevents one failure from canceling others
- Failed agents return `AgentResult` with `success=False`

#### When to Use

✅ **Good for:**
- Getting diverse perspectives on the same problem
- Brainstorming and idea generation
- Redundancy (multiple agents solve same problem)
- Benchmarking agent performance on identical tasks

❌ **Not suitable for:**
- Tasks requiring collaboration or building on others' work
- Sequential workflows with dependencies

#### Example

```python
coordinator = Coordinator([analyzer, coder, reviewer])
task = Task(task_id="design", description="Design a REST API for user management")
results = await coordinator.run_parallel(task)
# Returns 3 independent designs from each agent
```

---

### 2. Pipeline Pattern

**Method:** `run_pipeline(task: Task, agent_order: list[str], pass_context: bool = True, stop_on_failure: bool = False) -> list[AgentResult]`

#### How It Works

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│  Task   │────▶│ Agent A │────▶│ Agent B │────▶│ Agent C │
└─────────┘     └────┬────┘     └────┬────┘     └────┬────┘
                     │               │               │
                     ▼               ▼               ▼
                ┌─────────┐    ┌─────────┐    ┌─────────┐
                │Output A │───▶│Output B │───▶│Output C │
                │(context)│    │(context)│    │ (final) │
                └─────────┘    └─────────┘    └─────────┘
```

Agents execute **sequentially**. Each agent's output is added to the context for the next agent.

#### Implementation Details

- Each agent receives an `accumulated_context` dictionary
- Previous outputs stored as `{role}_output` keys (e.g., `analyzer_output`)
- Output truncated to `output_truncation_limit` (default: 1000 chars)
- Optional `stop_on_failure` halts pipeline on first error

#### Context Accumulation

```python
# After Agent A (analyzer) completes:
context = {
    "analyzer_output": "Analysis results...",
}

# After Agent B (coder) completes:
context = {
    "analyzer_output": "Analysis results...",
    "coder_output": "Implementation code...",
}
```

#### When to Use

✅ **Good for:**
- Code review workflows (write → review → fix)
- Multi-stage analysis (research → analyze → summarize)
- Document processing pipelines
- Any workflow with clear sequential dependencies

❌ **Not suitable for:**
- Tasks that can be parallelized
- Exploratory tasks without clear sequence

#### Example

```python
coordinator = Coordinator([planner, coder, reviewer])
task = Task(task_id="feature", description="Implement user authentication")

results = await coordinator.run_pipeline(
    task,
    agent_order=["planner_1", "coder_1", "reviewer_1"],
    pass_context=True,
    stop_on_failure=True  # Stop if any step fails
)
```

---

### 3. Debate Pattern

**Method:** `run_debate(task: Task, rounds: int = 2, agent_ids: Optional[list[str]] = None) -> list[AgentResult]`

#### How It Works

```
Round 1:                          Round 2:
┌─────────┐                       ┌─────────┐
│  Task   │                       │  Task   │
└────┬────┘                       └────┬────┘
     │                                 │
     ├─────────────┐                   ├─────────────┐
     ▼             ▼                   ▼             ▼
┌─────────┐  ┌─────────┐         ┌─────────┐  ┌─────────┐
│ Agent A │  │ Agent B │         │ Agent A │  │ Agent B │
└────┬────┘  └────┬────┘         └────┬────┘  └────┬────┘
     │             │                  │            │
     ▼             ▼                  ▼            ▼
┌─────────┐  ┌─────────┐          ┌─────────┐  ┌─────────┐
│Response │  │Response │  ──────▶ │Response │  │Response │
│   A1    │  │   B1    │ context  │   A2    │  │   B2    │
└─────────┘  └─────────┘          └─────────┘  └─────────┘
```

Multiple **rounds** where agents respond **in parallel**, but each round includes **all previous responses** as context.

#### Implementation Details

- Agents execute in parallel within each round (`asyncio.gather()`)
- Previous responses stored in `context["previous_responses"]` as dict
- Task description prefixed with round info: `"Round 2/3: {description}"`
- Responses truncated to `context_truncation_limit` (default: 500 chars)

#### Context Structure

```python
# Round 2 context includes:
{
    "debate_round": 2,
    "total_rounds": 3,
    "previous_responses": {
        "agent_a": "In round 1, I argued that...",
        "agent_b": "My initial position was..."
    }
}
```

#### When to Use

✅ **Good for:**
- Exploring trade-offs and alternatives
- Decision-making with multiple perspectives
- Adversarial testing of ideas
- Reaching refined conclusions through iteration

❌ **Not suitable for:**
- Tasks with clear right/wrong answers
- Time-sensitive operations (multiple rounds add latency)

#### Example

```python
coordinator = Coordinator([optimist, pessimist, pragmatist])
task = Task(
    task_id="decision",
    description="Should we migrate to microservices architecture?"
)

results = await coordinator.run_debate(task, rounds=3)
# 9 results total: 3 agents × 3 rounds
```

---

### 4. Hierarchical Pattern

**Method:** `run_hierarchical(task: Task, leader_id: str, worker_ids: Optional[list[str]] = None) -> list[AgentResult]`

#### How It Works

```
Phase 1: Planning              Phase 2: Execution           Phase 3: Synthesis
┌─────────────────┐            ┌─────────────────┐          ┌──────────────────┐
│                 │            │                 │          │                  │
│  ┌───────────┐  │            │    ┌───────┐    │          │  ┌────────────┐  │
│  │  Leader   │  │            │    │Leader │    │          │  │  Leader    │  │
│  │  (Plan)   │  │            │    │(idle) │    │          │  │(Synthesize)│  │
│  └─────┬─────┘  │            │    └───────┘    │          │  └─────┬──────┘  │
│        │        │            │                 │          │        ▲         │
│        ▼        │            │  ┌─────────────┐│          │        │         │
│   ┌─────────┐   │  ────────▶ │  │  Workers   ││ ───────▶ │   ┌────┴────┐    │
│   │  Plan   │   │            │  │  (parallel) ││          │   │ Results │    │
│   └─────────┘   │            │  └─────────────┘│          │   └─────────┘    │
│                 │            │                 │          │                  │
└─────────────────┘            └─────────────────┘          └──────────────────┘
```

A **leader agent** plans, **workers execute in parallel**, then leader **synthesizes** results.

#### Implementation Details

**Phase 1 - Planning:**
```python
planning_task = Task(
    description="As team leader, analyze this task and create a plan...",
    context={"role": "leader", "phase": "planning", "worker_count": N}
)
```

**Phase 2 - Execution:**
- Workers receive leader's plan in context
- All workers execute in parallel (`asyncio.gather()`)
- Failed workers don't block others

**Phase 3 - Synthesis:**
```python
synthesis_task = Task(
    description="Synthesize worker outputs into a final cohesive result...",
    context={"role": "leader", "phase": "synthesis"}
)
```

#### When to Use

✅ **Good for:**
- Complex tasks requiring decomposition
- Large projects with parallelizable subtasks
- Scenarios needing both breadth (workers) and coherence (leader)
- Simulating team dynamics

❌ **Not suitable for:**
- Simple tasks that don't benefit from decomposition
- When all agents should have equal authority

#### Example

```python
coordinator = Coordinator([architect, frontend_dev, backend_dev, db_dev])
task = Task(
    task_id="feature",
    description="Build a real-time notification system"
)

results = await coordinator.run_hierarchical(
    task,
    leader_id="architect_1",
    worker_ids=["frontend_1", "backend_1", "db_1"]
)
# Returns: [plan, frontend_work, backend_work, db_work, synthesis]
```

---

### 5. Consensus Pattern

**Method:** `run_consensus(task: Task, max_rounds: int = 3, agreement_threshold: float = 0.8, agent_ids: Optional[list[str]] = None) -> list[AgentResult]`

#### How It Works

```
Round 1                    Round 2                    Round 3 (if needed)
┌──────────────┐          ┌──────────────┐           ┌──────────────┐
│   Parallel   │          │   Parallel   │           │   Parallel   │
│   Execution  │          │   Execution  │           │   Execution  │
└──────┬───────┘          └──────┬───────┘           └──────┬───────┘
       │                         │                          │
       ▼                         ▼                          ▼
┌──────────────┐          ┌──────────────┐           ┌──────────────┐
│    Check     │    No    │    Check     │    No     │    Check     │
│  Consensus   │─────────▶│  Consensus   │──────────▶│  Consensus  │
└──────┬───────┘          └──────┬───────┘           └──────┬───────┘
       │ Yes                     │ Yes                      │
       ▼                         ▼                          ▼
   ┌───────┐                 ┌───────┐                  ┌───────┐
   │ EXIT  │                 │ EXIT  │                  │ EXIT  │
   └───────┘                 └───────┘                  └───────┘
```

Like debate, but with **automatic consensus detection** and **early termination**.

#### Implementation Details

**Consensus Detection** uses three signals:

1. **Explicit Markers** (strongest):
   - Agents are prompted to start with "I AGREE" or "I DISAGREE"
   - Direct parsing of response beginning

2. **Sentiment Analysis** (negation-aware):
   ```python
   # Correctly handles:
   "I agree" → positive
   "I don't agree" → negative (detects negation)
   "I disagree" → negative
   "I don't disagree" → positive (double negative)
   ```

3. **Output Similarity** (Jaccard):
   - Compares word overlap between all response pairs
   - High similarity suggests convergence even without explicit agreement

**Combined Score:**
```python
combined_score = 0.7 * explicit_agreement + 0.3 * output_similarity
consensus_reached = combined_score >= agreement_threshold
```

#### Prompt Enhancement

Agents receive explicit instructions:
```
"If you agree with the consensus, clearly state 'I AGREE' at the start.
If you disagree, clearly state 'I DISAGREE' at the start and explain your reasoning."
```

#### When to Use

✅ **Good for:**
- Critical decisions requiring validation
- Tasks where agreement indicates correctness
- Reducing bias through multi-agent validation
- High-stakes outputs

❌ **Not suitable for:**
- Creative tasks where diversity is valued
- Tasks with no clear "right" answer
- Time-critical operations

#### Example

```python
coordinator = Coordinator([validator_1, validator_2, validator_3])
task = Task(
    task_id="verify",
    description="Is this SQL query safe from injection attacks?"
)

results = await coordinator.run_consensus(
    task,
    max_rounds=3,
    agreement_threshold=0.8  # 80% agreement required
)
# Early exit if consensus reached before max_rounds
```

---

### 6. Peer Review Pattern

**Method:** `run_peer_review(task: Task, agent_ids: Optional[list[str]] = None) -> list[AgentResult]`

#### How It Works

```
Phase 1: Work              Phase 2: Review            Phase 3: Revise
┌─────────────────┐       ┌─────────────────┐        ┌─────────────────┐
│    Parallel     │       │    Parallel     │        │    Parallel     │
│   (all work)    │       │  (all review)   │        │  (all revise)   │
└────────┬────────┘       └────────┬────────┘        └────────┬────────┘
         │                         │                          │
         ▼                         ▼                          ▼
   ┌───────────┐            ┌───────────┐             ┌───────────┐
   │ A writes  │            │ A reviews │             │ A revises │
   │ B writes  │            │ B reviews │             │ B revises │
   │ C writes  │            │ C reviews │             │ C revises │
   └───────────┘            └───────────┘             └───────────┘

Review Assignment (circular):
A reviews B's work → B reviews C's work → C reviews A's work
```

Three-phase workflow: **Work → Review → Revise**, all phases execute in parallel.

#### Implementation Details

**Circular Review Assignment:**
```python
# For agent at index i:
reviews_work_of = (i + 1) % n_agents  # Agent i reviews agent i+1
gets_feedback_from = (i - 1) % n_agents  # Agent i-1 reviewed agent i
```

**Phase Contexts:**
```python
# Phase 1: Initial work
context = {"phase": "initial_work"}

# Phase 2: Review
context = {"phase": "peer_review", "reviewing": "agent_b"}

# Phase 3: Revision
context = {"phase": "revision", "feedback_from": "agent_c"}
```

#### Result Structure

For 3 agents, returns 9 results:
```
[
    initial_A, initial_B, initial_C,      # Phase 1
    review_A, review_B, review_C,          # Phase 2
    revision_A, revision_B, revision_C     # Phase 3
]
```

#### When to Use

✅ **Good for:**
- Code review workflows
- Document review and editing
- Quality assurance processes
- Academic-style peer review
- Catching errors through external review

❌ **Not suitable for:**
- Tasks without reviewable outputs
- Time-sensitive operations (3 full phases)
- Small tasks where review overhead exceeds value

#### Example

```python
coordinator = Coordinator([coder_1, coder_2, coder_3])
task = Task(
    task_id="implement",
    description="Implement a thread-safe cache with LRU eviction"
)

results = await coordinator.run_peer_review(task)
# Each coder writes, reviews another's work, then revises based on feedback
```

---

## Implementation Architecture

### Shared Infrastructure

All patterns share common infrastructure:

```python
class Coordinator:
    def __init__(
        self,
        agents: Sequence[Agent],
        context_truncation_limit: int = 500,   # For context sharing
        output_truncation_limit: int = 1000    # For output storage
    ):
        self.agents: dict[str, Agent] = {a.agent_id: a for a in agents}
        self.results: list[AgentResult] = []
        self.execution_history: list[dict] = []
```

### Error Handling

All parallel patterns use consistent error handling:

```python
results = await asyncio.gather(*[...], return_exceptions=True)

for result in results:
    if isinstance(result, Exception):
        result = AgentResult(
            success=False,
            error=str(result),
            # ... other fields
        )
```

### Execution Logging

Every pattern logs execution details:

```python
self._log_execution(
    pattern="debate",
    task=task,
    results=results,
    total_time_s=elapsed
)
```

---

## Configuration Options

### Coordinator-Level

| Parameter | Default | Description |
|-----------|---------|-------------|
| `context_truncation_limit` | 500 | Max chars when sharing context between agents |
| `output_truncation_limit` | 1000 | Max chars when storing outputs |

### Pattern-Level

| Pattern | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| Pipeline | `pass_context` | `True` | Pass outputs to subsequent agents |
| Pipeline | `stop_on_failure` | `False` | Halt on first failure |
| Debate | `rounds` | `2` | Number of debate rounds |
| Consensus | `max_rounds` | `3` | Maximum rounds before giving up |
| Consensus | `agreement_threshold` | `0.8` | Required agreement ratio |

---

## Best Practices

### 1. Choose the Right Pattern

```
Need diverse perspectives?          → Parallel
Have sequential dependencies?       → Pipeline
Exploring trade-offs?               → Debate
Complex task decomposition?         → Hierarchical
Critical decision validation?       → Consensus
Quality assurance workflow?         → Peer Review
```

### 2. Agent Role Matching

| Pattern | Recommended Agent Roles |
|---------|------------------------|
| Parallel | Mixed roles for diversity |
| Pipeline | Complementary roles (Planner → Coder → Reviewer) |
| Debate | Similar capabilities, different "personalities" |
| Hierarchical | One Planner/Synthesizer + multiple Executors |
| Consensus | Similar roles for valid comparison |
| Peer Review | Same role (all can work AND review) |

### 3. Resource Considerations

- **Parallel/Debate/Consensus**: API rate limits may apply with many agents
- **Pipeline**: Total latency = sum of all agent latencies
- **Hierarchical**: 3 sequential phases minimum
- **Peer Review**: 3 phases × N agents = 3N API calls

### 4. Truncation Tuning

```python
# For code-heavy tasks (need more context)
coordinator = Coordinator(
    agents,
    context_truncation_limit=1000,
    output_truncation_limit=2000
)

# For summarization tasks (less context needed)
coordinator = Coordinator(
    agents,
    context_truncation_limit=300,
    output_truncation_limit=500
)
```

---

## Appendix: Quick Reference

```python
from apee.coordination import Coordinator
from apee.models import Task

# Initialize
coordinator = Coordinator([agent1, agent2, agent3])
task = Task(task_id="demo", description="Solve this problem")

# Parallel - Independent execution
results = await coordinator.run_parallel(task)

# Pipeline - Sequential with context passing
results = await coordinator.run_pipeline(task, ["a1", "a2", "a3"])

# Debate - Multi-round discussion
results = await coordinator.run_debate(task, rounds=3)

# Hierarchical - Leader delegates to workers
results = await coordinator.run_hierarchical(task, leader_id="a1")

# Consensus - Iterate until agreement
results = await coordinator.run_consensus(task, max_rounds=3)

# Peer Review - Work → Review → Revise
results = await coordinator.run_peer_review(task)
```
