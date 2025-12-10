# APEE Evaluation Patterns

This document explains the **evaluation-specific** coordination patterns available in APEE's evaluation module. These patterns focus on **measuring quality** and **reducing evaluation bias** rather than task execution.

## Table of Contents

1. [Overview](#overview)
2. [Pattern Taxonomy](#pattern-taxonomy)
3. [Detailed Pattern Documentation](#detailed-pattern-documentation)
   - [Jury with Personas](#1-jury-with-personas-independent-pattern)
   - [Calibration Loop](#2-calibration-loop-iterative-pattern)
   - [Progressive Deepening](#3-progressive-deepening-sequential-pattern)
   - [Calibrated Jury](#4-calibrated-jury-combined-pattern)
4. [Implementation Details](#implementation-details)
5. [Usage Examples](#usage-examples)
6. [Best Practices](#best-practices)

---

## Overview

APEE implements evaluation-specific patterns that go beyond standard LLM-as-Judge approaches. These patterns address common evaluation challenges:

- **Single-perspective bias**: One judge may have systematic biases
- **Inconsistent rubrics**: Different judges interpret criteria differently
- **Subjectivity in ambiguous tasks**: Novel tasks lack clear evaluation standards
- **Cost inefficiency**: Full evaluation on obvious pass/fail wastes compute

```
┌──────────────────────────────────────────────────────────────────────────────┐
│               APEE Evaluation Patterns                                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Independent          Sequential           Iterative          Hybrid         │
│  (Concurrent)         (Linear Funnel)      (Cyclical)         (Combined)     │
│  ───────────          ──────────           ─────────          ──────         │
│  • Jury/Voting        • Progressive        • Calibration      • Calibrated   │
│    with Personas        Deepening            Loop               Jury         │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Pattern Taxonomy

| Pattern | Category | Description | When to Use |
|---------|----------|-------------|-------------|
| **Jury with Personas** | Independent | Multiple judges with distinct evaluation lenses (Skeptic, Literalist, Optimist, Pragmatist) | Subjective evaluations, contentious outputs |
| **Calibration Loop** | Iterative | Judges negotiate rubric criteria before scoring | Novel tasks, ambiguous requirements |
| **Progressive Deepening** | Sequential | Fail-fast evaluation with escalating depth | High-volume evaluation, cost optimization |
| **Calibrated Jury** | Hybrid | Calibration + Personas combined | High-stakes evaluations |

### Quality vs. Cost Trade-off

```
        High Quality
             ▲
             │         ┌──────────────────┐
             │         │ Calibrated Jury  │
             │         └──────────────────┘
             │    ┌─────────────────┐
             │    │ Calibration Loop│
             │    └─────────────────┘
             │              ┌────────────────────┐
             │              │ Jury with Personas │
             │              └────────────────────┘
             │                   ┌──────────────────────────┐
             │                   │ Progressive Deepening    │
             │                   │ (adaptive cost/quality)  │
             │                   └──────────────────────────┘
             │                              ┌────────────────┐
             │                              │ Single Judge   │
             │                              │   (baseline)   │
             │                              └────────────────┘
             └────────────────────────────────────────────────► Low Cost
```

---

## Detailed Pattern Documentation

### 1. Jury with Personas (Independent Pattern)

**Class:** `JuryEvaluator`

#### Concept

Instead of using multiple judge **models**, use one model with multiple judge **personas**. Each persona brings a different evaluation lens:

| Persona | Focus | Tendency |
|---------|-------|----------|
| **Skeptic** | Flaws, edge cases, failure modes | Lower scores, identifies risks |
| **Literalist** | Strict requirement adherence | Penalizes assumptions |
| **Optimist** | Creative solutions, potential | Higher scores, sees value |
| **Pragmatist** | Real-world utility, maintainability | Practical assessment |

#### How It Works

```
                    ┌──────────────┐
                    │ Agent Output │
                    └──────┬───────┘
                           │
       ┌───────────────────┼───────────────────┐
       ▼                   ▼                   ▼
┌────────────┐      ┌────────────┐      ┌────────────┐
│  SKEPTIC   │      │ LITERALIST │      │  OPTIMIST  │
│   Judge    │      │   Judge    │      │   Judge    │
│  Score: 6  │      │  Score: 5  │      │  Score: 8  │
└────────────┘      └────────────┘      └────────────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           ▼
                  ┌────────────────┐
                  │  Aggregation   │
                  │  (mean/median) │
                  │   Score: 6.3   │
                  └────────────────┘
                           │
                           ▼
                  ┌────────────────┐
                  │  Disagreement  │
                  │   Analysis     │
                  │  Range: 3.0    │
                  └────────────────┘
```

#### Key Benefits

1. **Reduces single-perspective bias** without needing multiple LLM models
2. **Disagreement tracking** identifies contentious outputs
3. **Interpretable feedback** from each perspective

#### Configuration

```python
from apee.evaluation import JuryEvaluator, JudgePersona

# Use all 4 personas (default)
jury = JuryEvaluator(
    model="qwen2.5-coder:7b",
    base_url="http://localhost:11434",
    aggregation="weighted_mean"  # or "mean", "median"
)

# Use specific personas
jury = JuryEvaluator(
    model="qwen2.5-coder:7b",
    personas=[JudgePersona.SKEPTIC, JudgePersona.PRAGMATIST]
)
```

---

### 2. Calibration Loop (Iterative Pattern)

**Class:** `CalibrationLoop`

#### Concept

Before evaluating, judges **negotiate the evaluation rubric**. This ensures consistent interpretation of criteria, especially for novel or ambiguous tasks.

#### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                     CALIBRATION PHASE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Round 1: Propose Criteria                                      │
│  ┌──────────┐     ┌──────────┐                                  │
│  │ Judge A  │     │ Judge B  │                                  │
│  │ Proposes │     │ Proposes │                                  │
│  │ Criteria │     │ Criteria │                                  │
│  └────┬─────┘     └────┬─────┘                                  │
│       │                │                                        │
│       └───────┬────────┘                                        │
│               ▼                                                 │
│       ┌──────────────┐                                          │
│       │  Synthesize  │                                          │
│       │   Rubric     │                                          │
│       └──────┬───────┘                                          │
│              │                                                  │
│  Round 2: Agreement Check                                       │
│              ▼                                                  │
│       ┌──────────────┐                                          │
│       │ Agreement    │                                          │
│       │ >= 70%? ─────┼──► YES ──► Use Rubric                    │
│       └──────┬───────┘                                          │
│              │ NO                                               │
│              ▼                                                  │
│       ┌──────────────┐                                          │
│       │   Refine     │──► Loop (max 3 rounds)                   │
│       │   Rubric     │                                          │
│       └──────────────┘                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     EVALUATION PHASE                            │
│  (Use calibrated rubric for consistent evaluation)              │
└─────────────────────────────────────────────────────────────────┘
```

#### Calibrated Rubric Structure

```python
CalibratedRubric:
  task_type: "code_review"
  criteria:
    - name: "correctness"
      description: "Code produces expected output"
      weight: 0.4
      score_anchors:
        2: "Major bugs, doesn't run"
        5: "Runs but edge cases fail"
        8: "All cases pass"
    - name: "maintainability"
      description: "Code is readable and maintainable"
      weight: 0.3
      score_anchors:
        2: "Unreadable, no structure"
        5: "Readable but could improve"
        8: "Clean, well-documented"
  calibration_notes: "Agreed to weight correctness over style"
  agreed_by: ["judge1", "judge2"]
```

#### Key Benefits

1. **Consistent rubric** across all judges
2. **Explicit criteria** reduce subjectivity
3. **Score anchors** ensure scoring consistency
4. **Rubric caching** for repeated evaluations of same task type

#### Configuration

```python
from apee.evaluation import CalibrationLoop

calibrator = CalibrationLoop(
    judge_models=["qwen2.5-coder:7b", "llama3.2:3b"],
    base_url="http://localhost:11434",
    max_calibration_rounds=3,
    agreement_threshold=0.7
)

# Calibrate for a specific task
rubric = calibrator.calibrate(
    task_description="Review this Python function for bugs",
    task_type="code_review"
)
```

---

### 3. Progressive Deepening (Sequential Pattern)

**Class:** `ProgressiveDeepening`

#### Concept

A **fail-fast optimization** that runs evaluation at increasing depth levels. Obvious pass/fail cases terminate early, saving 60-80% tokens on high-volume evaluations.

| Depth Level | Method | Token Cost | When to Stop |
|-------------|--------|------------|--------------|
| **QUICK** | Heuristics (no LLM) | 0 | Score ≥9 (pass) or ≤2 (fail) |
| **STANDARD** | Single LLM pass | ~500 | Score ≥8 (pass) or ≤3 (fail) |
| **DEEP** | 2-persona mini-jury | ~1000 | Score ≥7 (pass) or ≤4 (fail) |
| **COMPREHENSIVE** | Full 4-persona jury | ~2000 | Always completes |

#### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│              PROGRESSIVE DEEPENING WORKFLOW                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐                                               │
│  │ Agent Output │                                               │
│  └──────┬───────┘                                               │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────────────┐                                       │
│  │  Level 1: QUICK      │  (Heuristics - no LLM)                │
│  │  ─────────────────── │                                       │
│  │  • Length check      │                                       │
│  │  • Keyword matching  │                                       │
│  │  • Format validation │                                       │
│  └──────────┬───────────┘                                       │
│             │                                                   │
│    ┌────────┴────────┐                                          │
│    │   Score ≥ 9?    │──► YES ──► EARLY PASS (save ~3500 tok)   │
│    │   Score ≤ 2?    │──► YES ──► EARLY FAIL (save ~3500 tok)   │
│    └────────┬────────┘                                          │
│             │ UNCERTAIN                                         │
│             ▼                                                   │
│  ┌──────────────────────┐                                       │
│  │  Level 2: STANDARD   │  (Single LLM evaluation)              │
│  │  ─────────────────── │                                       │
│  │  • Full prompt eval  │                                       │
│  │  • Strengths/weaks   │                                       │
│  └──────────┬───────────┘                                       │
│             │                                                   │
│    ┌────────┴────────┐                                          │
│    │   Score ≥ 8?    │──► YES ──► EARLY PASS (save ~3000 tok)   │
│    │   Score ≤ 3?    │──► YES ──► EARLY FAIL (save ~3000 tok)   │
│    └────────┬────────┘                                          │
│             │ UNCERTAIN                                         │
│             ▼                                                   │
│  ┌──────────────────────┐                                       │
│  │  Level 3: DEEP       │  (2-persona mini-jury)                │
│  │  ─────────────────── │                                       │
│  │  • Skeptic view      │                                       │
│  │  • Pragmatist view   │                                       │
│  └──────────┬───────────┘                                       │
│             │                                                   │
│    ┌────────┴────────┐                                          │
│    │   Score ≥ 7?    │──► YES ──► EARLY PASS (save ~2000 tok)   │
│    │   Score ≤ 4?    │──► YES ──► EARLY FAIL (save ~2000 tok)   │
│    └────────┬────────┘                                          │
│             │ STILL UNCERTAIN                                   │
│             ▼                                                   │
│  ┌──────────────────────┐                                       │
│  │ Level 4: COMPREHENSIVE│  (Full 4-persona jury)               │
│  │  ─────────────────── │                                       │
│  │  • All personas      │                                       │
│  │  • Disagreement      │                                       │
│  │  • Final score       │                                       │
│  └──────────┬───────────┘                                       │
│             │                                                   │
│             ▼                                                   │
│     ┌───────────────┐                                           │
│     │ FINAL RESULT  │                                           │
│     └───────────────┘                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Key Benefits

1. **Cost optimization**: 60-80% token savings on obvious cases
2. **Maintains quality**: Edge cases still get full evaluation
3. **Configurable thresholds**: Adjust pass/fail thresholds per use case
4. **Tracking**: Know exactly which depth level was reached

#### Configuration

```python
from apee.evaluation import create_progressive_evaluator

# Default: use all depth levels
evaluator = create_progressive_evaluator(model="qwen2.5-coder:7b")

# Limit max depth for speed
evaluator = create_progressive_evaluator(
    model="qwen2.5-coder:7b",
    max_depth="deep"  # Don't go to comprehensive
)

# Custom thresholds (more lenient passing)
evaluator = create_progressive_evaluator(
    model="qwen2.5-coder:7b",
    custom_thresholds={
        "standard": (7.5, 3.5),  # (pass_threshold, fail_threshold)
        "deep": (6.5, 4.5),
    }
)

result = evaluator.evaluate(trace)

# Result contains:
print(f"Final Score: {result.final_score.score}")
print(f"Depth Reached: {result.depth_reached}")
print(f"Early Termination: {result.early_termination}")
print(f"Termination Reason: {result.termination_reason}")
print(f"Tokens Saved: {result.tokens_saved_estimate}")
print(f"Depth Scores: {result.depth_scores}")
```

#### Heuristic Checks (Quick Level)

The quick level runs without LLM calls:

| Check | Weight | Logic |
|-------|--------|-------|
| Output exists | Pass/Fail | Must be non-empty, >10 chars |
| Adequate length | +1.0 | Output ≥ 50% of expected length |
| Meets expected length | +1.0 | Output ≥ expected length |
| Keyword coverage >30% | +1.0 | Task keywords found in output |
| High keyword coverage >60% | +1.0 | Most task keywords found |
| Has structure | +0.5 | Contains code blocks or paragraphs |
| Multiple error indicators | -2.0 | "error", "failed", etc. present |

---

### 4. Calibrated Jury (Combined Pattern)

**Class:** `CalibratedJuryEvaluator`

#### Concept

The **recommended approach** for high-quality evaluation: combine Calibration Loop + Jury with Personas.

1. First, multiple judges calibrate on a shared rubric
2. Then, a jury of personas evaluates using the calibrated rubric

#### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    CALIBRATED JURY WORKFLOW                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 1: CALIBRATION                                            │
│  ┌──────────┐     ┌──────────┐                                  │
│  │ Judge 1  │ ←──→│ Judge 2  │  Negotiate rubric                │
│  └────┬─────┘     └────┬─────┘                                  │
│       └───────┬────────┘                                        │
│               ▼                                                 │
│       ┌──────────────┐                                          │
│       │  Calibrated  │                                          │
│       │    Rubric    │                                          │
│       └──────┬───────┘                                          │
│              │                                                  │
│  Step 2: JURY EVALUATION (using rubric)                         │
│              │                                                  │
│    ┌─────────┼─────────┬─────────────┐                          │
│    ▼         ▼         ▼             ▼                          │
│ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────────┐                         │
│ │Skept.│ │Liter.│ │Optim.│ │Pragmtst. │                         │
│ │ 6.0  │ │ 5.5  │ │ 8.0  │ │   7.0    │                         │
│ └──────┘ └──────┘ └──────┘ └──────────┘                         │
│    │         │         │             │                          │
│    └─────────┴─────────┴─────────────┘                          │
│                    │                                            │
│                    ▼                                            │
│            ┌──────────────┐                                     │
│            │ Final: 6.6   │                                     │
│            │ + Breakdown  │                                     │
│            │ + Rubric info│                                     │
│            └──────────────┘                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Configuration

```python
from apee.evaluation import CalibratedJuryEvaluator, JudgePersona

evaluator = CalibratedJuryEvaluator(
    judge_models=["qwen2.5-coder:7b", "llama3.2:3b"],
    base_url="http://localhost:11434",
    personas=[JudgePersona.SKEPTIC, JudgePersona.PRAGMATIST],
    max_calibration_rounds=2
)

result = evaluator.evaluate(trace, task_type="code_review")

# Result includes:
# - aggregated_score: Final score from jury
# - persona_scores: Score from each persona
# - disagreement: Persona disagreement metrics
# - calibration: Rubric used, criteria, calibration notes
```

---

## Implementation Details

### Persona System Prompts

Each persona has a distinct system prompt modifier:

```python
SKEPTIC:
"You are a SKEPTICAL evaluator. Your role is to:
- Look for potential flaws, edge cases, and failure modes
- Question assumptions made by the agent
- Identify what could go wrong with this output
..."

LITERALIST:
"You are a LITERALIST evaluator. Your role is to:
- Evaluate STRICTLY against the stated requirements
- Do not give credit for 'close enough' or 'probably meant'
- Check if EVERY requirement is explicitly addressed
..."
```

### Aggregation Methods

| Method | Formula | Best For |
|--------|---------|----------|
| `mean` | Average of all scores | General use |
| `median` | Middle value | Outlier resistance |
| `weighted_mean` | Sum(score × weight) / Sum(weights) | Prioritized personas |

### Disagreement Tracking

```python
disagreement = {
    "stdev": 1.2,           # Standard deviation
    "range": 3.0,           # Max - Min
    "high_disagreement": True,  # Range > 3.0
    "main_disagreement": "skeptic (5.0) vs optimist (8.0)"
}
```

---

## Usage Examples

### Basic Jury Evaluation

```python
from apee.evaluation import create_jury_evaluator
from apee.evaluation.llm_evaluator import ExecutionTrace

# Create trace from agent execution
trace = ExecutionTrace(
    agent_id="agent_1",
    agent_role="coder",
    task_description="Implement a binary search function",
    final_output="def binary_search(arr, target): ...",
    duration_seconds=5.0,
    token_count=200
)

# Evaluate with jury
jury = create_jury_evaluator(model="qwen2.5-coder:7b")
result = jury.evaluate(trace)

print(f"Final Score: {result['aggregated_score'].score}/10")
print(f"High Disagreement: {result['disagreement']['high_disagreement']}")
for persona, data in result['persona_scores'].items():
    print(f"  {persona}: {data['score']}/10")
```

### Full Calibrated Evaluation

```python
from apee.evaluation import create_calibrated_evaluator

evaluator = create_calibrated_evaluator(
    judge_models=["qwen2.5-coder:7b", "llama3.2:3b"],
    personas=["skeptic", "pragmatist"]
)

result = evaluator.evaluate(trace, task_type="code_implementation")

# Access calibration details
print(f"Rubric criteria: {result['calibration']['criteria']}")
print(f"Calibration rounds: {result['calibration']['calibration_rounds']}")
```

---

## Best Practices

### 1. Choose the Right Pattern

| Scenario | Recommended Pattern |
|----------|-------------------|
| Quick evaluation, single perspective OK | Single Judge (baseline) |
| High-volume, cost-conscious | **Progressive Deepening** |
| Need multiple perspectives, low cost | Jury with Personas |
| Novel task, unclear criteria | Calibration Loop |
| High-stakes, maximum quality | Calibrated Jury |

### 2. Progressive Deepening for Volume

For high-volume evaluation (100+ outputs):
```python
# Use progressive deepening to optimize cost
evaluator = create_progressive_evaluator(
    model="qwen2.5-coder:7b",
    custom_thresholds={
        "quick": (9.5, 1.5),  # Very strict quick check
        "standard": (8.0, 3.0),
    }
)

# Track savings
total_saved = 0
for trace in traces:
    result = evaluator.evaluate(trace)
    total_saved += result.tokens_saved_estimate
    
print(f"Total tokens saved: {total_saved}")
```

### 3. Persona Selection

- **Code review**: Skeptic + Pragmatist (catch bugs, ensure practicality)
- **Creative tasks**: Optimist + Literalist (value creativity but check requirements)
- **Security analysis**: Skeptic + Literalist (thorough, requirement-focused)

### 4. Handle High Disagreement

When `high_disagreement` is True (range > 3 points):
1. Review the output manually
2. Check if task requirements were ambiguous
3. Consider re-calibrating the rubric

### 5. Rubric Caching

Calibration is cached by task type. For consistent evaluations:
```python
# First call calibrates
result1 = evaluator.evaluate(trace1, task_type="code_review")

# Subsequent calls reuse cached rubric
result2 = evaluator.evaluate(trace2, task_type="code_review")

# Force re-calibration if needed
result3 = evaluator.evaluate(trace3, task_type="code_review", force_recalibrate=True)
```

---

## Related Documentation

- [Coordination Patterns](PATTERNS.md) - Agent execution patterns
- [LLM Evaluator](../evaluation/llm_evaluator.py) - Base evaluator classes
- [Ensemble Evaluator](../evaluation/llm_evaluator.py#EnsembleEvaluator) - Multi-model evaluation
