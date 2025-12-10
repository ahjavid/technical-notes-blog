"""
Advanced APEE Evaluation Patterns Demo

Demonstrates the 3 newly implemented evaluation patterns:
1. Progressive Deepening - Fail-fast evaluation (saves 60-80% tokens)
2. Jury with Personas - Multi-perspective evaluation (4 personas)
3. Calibration Loop - Negotiated rubric evaluation

Requirements:
- Ollama running locally with models:
  - qwen2.5-coder:3b
  - llama3.2:3b (for calibration)

Usage:
    python examples/advanced_evaluation_demo.py
    python examples/advanced_evaluation_demo.py --pattern jury
    python examples/advanced_evaluation_demo.py --pattern progressive
    python examples/advanced_evaluation_demo.py --pattern calibration
    python examples/advanced_evaluation_demo.py --all
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from apee.evaluation.advanced_patterns import (
    # Progressive Deepening
    ProgressiveDeepening,
    ProgressiveResult,
    EvaluationDepth,
    create_progressive_evaluator,
    # Jury with Personas
    JuryEvaluator,
    JudgePersona,
    PERSONA_CONFIGS,
    create_jury_evaluator,
    # Calibration Loop
    CalibrationLoop,
    CalibratedRubric,
    CalibratedJuryEvaluator,
    create_calibrated_evaluator,
)
from apee.evaluation.llm_evaluator import ExecutionTrace, CollaborativeTrace


# =============================================================================
# RESULTS COLLECTOR (for JSON export)
# =============================================================================

class ResultsCollector:
    """Collects evaluation results for JSON export."""
    
    def __init__(self):
        self.results: Dict[str, Any] = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "version": "1.0",
                "patterns_tested": []
            },
            "progressive_deepening": None,
            "jury_with_personas": None,
            "calibration_loop": None,
            "calibrated_jury": None,
            "summary": {}
        }
    
    def add_progressive_result(self, good_result: 'ProgressiveResult', bad_result: 'ProgressiveResult'):
        """Store progressive deepening results."""
        self.results["progressive_deepening"] = {
            "pattern": "Progressive Deepening",
            "description": "Fail-fast evaluation with escalating depth",
            "good_trace": {
                "score": good_result.final_score.score,
                "depth_reached": good_result.depth_reached.value,
                "early_termination": good_result.early_termination,
                "termination_reason": good_result.termination_reason,
                "depth_scores": good_result.depth_scores,
                "time_seconds": good_result.total_time_seconds,
                "tokens_saved_estimate": good_result.tokens_saved_estimate
            },
            "bad_trace": {
                "score": bad_result.final_score.score,
                "depth_reached": bad_result.depth_reached.value,
                "early_termination": bad_result.early_termination,
                "termination_reason": bad_result.termination_reason,
                "depth_scores": bad_result.depth_scores,
                "time_seconds": bad_result.total_time_seconds
            }
        }
        self.results["metadata"]["patterns_tested"].append("progressive_deepening")
    
    def add_jury_result(self, full_jury_result: Dict, focused_jury_result: Dict):
        """Store jury with personas results."""
        self.results["jury_with_personas"] = {
            "pattern": "Jury with Personas",
            "description": "Multi-perspective evaluation with 4 personas",
            "full_jury": {
                "personas": list(full_jury_result['persona_scores'].keys()),
                "aggregated_score": full_jury_result['aggregated_score'].score,
                "per_persona_scores": {
                    p: d['score'] for p, d in full_jury_result['persona_scores'].items()
                },
                "disagreement": full_jury_result['disagreement']
            },
            "focused_jury": {
                "personas": list(focused_jury_result['persona_scores'].keys()),
                "aggregated_score": focused_jury_result['aggregated_score'].score,
                "per_persona_scores": {
                    p: d['score'] for p, d in focused_jury_result['persona_scores'].items()
                }
            }
        }
        self.results["metadata"]["patterns_tested"].append("jury_with_personas")
    
    def add_calibration_result(self, rubric: 'CalibratedRubric'):
        """Store calibration loop results."""
        self.results["calibration_loop"] = {
            "pattern": "Calibration Loop",
            "description": "Judges negotiate rubric before scoring",
            "rubric": {
                "task_type": rubric.task_type,
                "calibration_rounds": rubric.calibration_rounds,
                "agreed_by": rubric.agreed_by,
                "criteria": [
                    {
                        "name": c.name,
                        "weight": c.weight,
                        "description": c.description,
                        "score_anchors": c.score_anchors
                    } for c in rubric.criteria
                ],
                "calibration_notes": rubric.calibration_notes
            }
        }
        self.results["metadata"]["patterns_tested"].append("calibration_loop")
    
    def add_calibrated_jury_result(self, result: Dict):
        """Store calibrated jury results."""
        self.results["calibrated_jury"] = {
            "pattern": "Calibrated Jury",
            "description": "Calibration + Personas for maximum quality",
            "aggregated_score": result['aggregated_score'].score,
            "calibration": result['calibration'],
            "per_persona_scores": {
                p: d['score'] for p, d in result['persona_scores'].items()
            },
            "disagreement": result['disagreement']
        }
        self.results["metadata"]["patterns_tested"].append("calibrated_jury")
    
    def generate_summary(self):
        """Generate summary statistics."""
        scores = []
        if self.results["progressive_deepening"]:
            scores.append(self.results["progressive_deepening"]["good_trace"]["score"])
        if self.results["jury_with_personas"]:
            scores.append(self.results["jury_with_personas"]["full_jury"]["aggregated_score"])
        if self.results["calibrated_jury"]:
            scores.append(self.results["calibrated_jury"]["aggregated_score"])
        
        self.results["summary"] = {
            "patterns_run": len(self.results["metadata"]["patterns_tested"]),
            "average_score": sum(scores) / len(scores) if scores else 0,
            "scores_collected": scores
        }
    
    def save(self, output_path: Optional[str] = None) -> str:
        """Save results to JSON file."""
        self.generate_summary()
        
        if output_path is None:
            # Default to data directory with timestamp
            data_dir = Path(__file__).parent.parent / "data"
            data_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = data_dir / f"advanced_evaluation_results_{timestamp}.json"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        return str(output_path)


# Global collector instance
_collector: Optional[ResultsCollector] = None


def get_collector() -> ResultsCollector:
    """Get or create the global results collector."""
    global _collector
    if _collector is None:
        _collector = ResultsCollector()
    return _collector


def reset_collector():
    """Reset the global results collector."""
    global _collector
    _collector = ResultsCollector()


# =============================================================================
# SAMPLE DATA
# =============================================================================

def create_good_trace() -> ExecutionTrace:
    """Create a high-quality execution trace for testing."""
    return ExecutionTrace(
        agent_id="security_analyzer_001",
        agent_role="analyzer",
        task_description="""Analyze the following Python function for security vulnerabilities 
and provide a detailed report with severity levels and fixes:

def process_user_input(data, user_id):
    result = json.loads(data)
    query = f"SELECT * FROM users WHERE id = {user_id}"
    db.execute(query)
    return open(result['filename']).read()
""",
        expected_output="A comprehensive security analysis with vulnerability identification and remediation",
        final_output="""
## Security Analysis Report

### Executive Summary
The provided code contains **3 critical** and **1 high** severity vulnerabilities that must be addressed before deployment.

---

### Critical Vulnerabilities

#### 1. SQL Injection (CRITICAL - CVSS 9.8)
**Location:** Line 3 - String interpolation in SQL query
**Risk:** Attackers can manipulate `user_id` to execute arbitrary SQL commands, potentially accessing, modifying, or deleting all database records.

**Vulnerable Code:**
```python
query = f"SELECT * FROM users WHERE id = {user_id}"
```

**Fix:**
```python
query = "SELECT * FROM users WHERE id = ?"
db.execute(query, [user_id])
```

#### 2. Path Traversal (CRITICAL - CVSS 9.1)
**Location:** Line 4 - Unvalidated file path from user input
**Risk:** Attackers can read arbitrary files on the system using paths like `../../etc/passwd`

**Vulnerable Code:**
```python
return open(result['filename']).read()
```

**Fix:**
```python
import os
safe_path = os.path.basename(result['filename'])
allowed_dir = '/app/uploads/'
full_path = os.path.join(allowed_dir, safe_path)
if not os.path.realpath(full_path).startswith(allowed_dir):
    raise ValueError("Invalid file path")
return open(full_path).read()
```

#### 3. Unsafe JSON Deserialization (CRITICAL - CVSS 8.6)
**Location:** Line 2 - `json.loads()` on untrusted input without validation
**Risk:** While `json.loads` is safer than `pickle`, the parsed data is used unsafely afterward

**Fix:**
```python
from pydantic import BaseModel, validator

class UserInput(BaseModel):
    filename: str
    
    @validator('filename')
    def validate_filename(cls, v):
        if '..' in v or v.startswith('/'):
            raise ValueError('Invalid filename')
        return v

validated = UserInput.parse_raw(data)
```

---

### High Severity Issues

#### 4. Missing Error Handling (HIGH - CVSS 7.5)
**Risk:** Unhandled exceptions leak stack traces and system information

**Fix:**
```python
try:
    result = json.loads(data)
except json.JSONDecodeError as e:
    logger.error(f"Invalid JSON input: {e}")
    raise HTTPException(400, "Invalid input format")
```

---

### Recommendations
1. Implement input validation layer using Pydantic or similar
2. Use parameterized queries for ALL database operations
3. Add comprehensive logging for security events
4. Implement rate limiting on this endpoint
5. Add automated security scanning to CI/CD pipeline

### OWASP Classification
- A03:2021 – Injection
- A01:2021 – Broken Access Control
- A04:2021 – Insecure Design
""",
        duration_seconds=8.5,
        token_count=850
    )


def create_poor_trace() -> ExecutionTrace:
    """Create a low-quality execution trace for testing early termination."""
    return ExecutionTrace(
        agent_id="lazy_analyzer",
        agent_role="analyzer",
        task_description="Analyze the code for security vulnerabilities and provide detailed fixes",
        expected_output="A comprehensive security analysis",
        final_output="Looks fine to me.",
        duration_seconds=0.5,
        token_count=5
    )


def create_medium_trace() -> ExecutionTrace:
    """Create a medium-quality trace for testing standard evaluation."""
    return ExecutionTrace(
        agent_id="basic_analyzer",
        agent_role="analyzer",
        task_description="Review the authentication system for security issues",
        expected_output="Security review with findings and recommendations",
        final_output="""
## Security Review

Found some issues:
1. Password stored in plain text - should hash with bcrypt
2. No rate limiting on login endpoint
3. Session tokens don't expire

Recommendations:
- Add password hashing
- Implement rate limiting
- Set session expiry
""",
        duration_seconds=3.0,
        token_count=120
    )


# =============================================================================
# DEMO FUNCTIONS
# =============================================================================

def demo_progressive_deepening():
    """Demonstrate Progressive Deepening pattern."""
    print("\n" + "="*70)
    print("🔄 PROGRESSIVE DEEPENING DEMO")
    print("   Fail-fast evaluation with escalating depth")
    print("="*70)
    
    evaluator = create_progressive_evaluator(
        model="qwen2.5-coder:3b",
        max_depth="comprehensive"
    )
    
    print("\n📊 Depth Level Thresholds:")
    print("   QUICK:         Pass >= 9.0, Fail <= 2.0 (heuristics only)")
    print("   STANDARD:      Pass >= 8.0, Fail <= 3.0 (single LLM)")
    print("   DEEP:          Pass >= 7.0, Fail <= 4.0 (2 personas)")
    print("   COMPREHENSIVE: Always returns final score (full jury)")
    
    # Test 1: Poor output (should fail fast at QUICK)
    print("\n" + "-"*50)
    print("Test 1: Poor Quality Output")
    print("-"*50)
    poor_trace = create_poor_trace()
    print(f"   Input: '{poor_trace.final_output}'")
    
    result = evaluator.evaluate(poor_trace)
    print(f"\n   ✅ Result:")
    print(f"      Score: {result.final_score.score}/10")
    print(f"      Depth Reached: {result.depth_reached.value}")
    print(f"      Early Termination: {result.early_termination}")
    print(f"      Reason: {result.termination_reason}")
    print(f"      Tokens Saved: ~{result.tokens_saved_estimate}")
    print(f"      Time: {result.total_time_seconds:.2f}s")
    
    # Test 2: Good output (may pass at STANDARD or DEEP)
    print("\n" + "-"*50)
    print("Test 2: High Quality Output")
    print("-"*50)
    good_trace = create_good_trace()
    print(f"   Input: {len(good_trace.final_output)} chars security analysis")
    
    good_result = evaluator.evaluate(good_trace)
    print(f"\n   ✅ Result:")
    print(f"      Score: {good_result.final_score.score}/10")
    print(f"      Depth Reached: {good_result.depth_reached.value}")
    print(f"      Early Termination: {good_result.early_termination}")
    print(f"      Reason: {good_result.termination_reason}")
    print(f"      Depth Scores: {good_result.depth_scores}")
    print(f"      Time: {good_result.total_time_seconds:.2f}s")
    
    # Store results
    poor_trace = create_poor_trace()
    poor_result = evaluator.evaluate(poor_trace)
    get_collector().add_progressive_result(good_result, poor_result)
    
    print("\n💡 Key Insight: Progressive Deepening saves tokens on obvious")
    print("   pass/fail cases while maintaining quality for edge cases.")


def demo_jury_with_personas():
    """Demonstrate Jury with Personas pattern."""
    print("\n" + "="*70)
    print("👥 JURY WITH PERSONAS DEMO")
    print("   Multi-perspective evaluation reduces single-viewpoint bias")
    print("="*70)
    
    print("\n📊 Available Personas:")
    for persona, config in PERSONA_CONFIGS.items():
        print(f"   {persona.value.upper()}: {config.focus_areas}")
    
    # Full jury (all 4 personas)
    print("\n" + "-"*50)
    print("Test 1: Full Jury (4 Personas)")
    print("-"*50)
    
    jury = create_jury_evaluator(
        model="qwen2.5-coder:3b",
        personas=None  # All 4
    )
    
    trace = create_good_trace()
    result = jury.evaluate(trace)
    
    print(f"\n   ✅ Results:")
    print(f"      Aggregated Score: {result['aggregated_score'].score}/10")
    print(f"      Aggregation Method: {jury.aggregation}")
    print(f"\n   📋 Per-Persona Breakdown:")
    
    for persona, data in result['persona_scores'].items():
        score = data['score'] if data['score'] else 'N/A'
        print(f"      {persona.upper():12} → {score}/10")
        if data['feedback']:
            preview = data['feedback'][:80].replace('\n', ' ')
            print(f"         {preview}...")
    
    print(f"\n   🔍 Disagreement Analysis:")
    print(f"      Standard Deviation: {result['disagreement']['stdev']:.2f}")
    print(f"      Score Range: {result['disagreement']['range']:.1f}")
    print(f"      High Disagreement: {result['disagreement']['high_disagreement']}")
    print(f"      Main Gap: {result['disagreement']['main_disagreement']}")
    
    # Focused jury (2 personas)
    print("\n" + "-"*50)
    print("Test 2: Focused Jury (Skeptic + Pragmatist)")
    print("-"*50)
    
    focused_jury = create_jury_evaluator(
        model="qwen2.5-coder:3b",
        personas=["skeptic", "pragmatist"]
    )
    
    result2 = focused_jury.evaluate(trace)
    print(f"\n   ✅ Focused Score: {result2['aggregated_score'].score}/10")
    for persona, data in result2['persona_scores'].items():
        print(f"      {persona}: {data['score']}/10")
    
    # Store results
    get_collector().add_jury_result(result, result2)
    
    print("\n💡 Key Insight: Different personas catch different issues.")
    print("   Skeptics find flaws, Optimists see potential, Pragmatists")
    print("   assess real-world utility. Combine for balanced evaluation.")


def demo_calibration_loop():
    """Demonstrate Calibration Loop pattern."""
    print("\n" + "="*70)
    print("🔧 CALIBRATION LOOP DEMO")
    print("   Judges negotiate rubric before scoring")
    print("="*70)
    
    print("\n📊 Calibration Process:")
    print("   1. Each judge proposes evaluation criteria")
    print("   2. Proposals are synthesized into unified rubric")
    print("   3. Judges vote on agreement (threshold: 70%)")
    print("   4. If disagreement, refine and iterate (max 3 rounds)")
    
    print("\n" + "-"*50)
    print("Test 1: Calibrate for Security Analysis")
    print("-"*50)
    
    calibrator = CalibrationLoop(
        judge_models=["qwen2.5-coder:3b", "llama3.2:3b"],
        max_calibration_rounds=2,
        agreement_threshold=0.6
    )
    
    rubric = calibrator.calibrate(
        task_description="""Evaluate a security vulnerability analysis report for:
- Completeness of vulnerability identification
- Quality of remediation suggestions  
- Accuracy of severity assessments
- Clarity of explanation""",
        task_type="security_analysis"
    )
    
    print(f"\n   ✅ Calibrated Rubric:")
    print(f"      Task Type: {rubric.task_type}")
    print(f"      Calibration Rounds: {rubric.calibration_rounds}")
    print(f"      Agreed By: {rubric.agreed_by}")
    
    print(f"\n   📋 Negotiated Criteria ({len(rubric.criteria)}):")
    for c in rubric.criteria:
        print(f"      • {c.name} (weight: {c.weight:.2f})")
        if c.description:
            print(f"        {c.description[:60]}...")
        if c.score_anchors:
            for score, anchor in sorted(c.score_anchors.items())[:2]:
                print(f"        Score {score}: {anchor[:50]}...")
    
    if rubric.calibration_notes:
        print(f"\n   📝 Calibration Notes:")
        print(f"      {rubric.calibration_notes[:200]}...")
    
    # Store results
    get_collector().add_calibration_result(rubric)
    
    print("\n💡 Key Insight: Calibration ensures judges share the same")
    print("   understanding of what 'good' means for this specific task.")


def demo_calibrated_jury():
    """Demonstrate combined Calibrated Jury pattern."""
    print("\n" + "="*70)
    print("⭐ CALIBRATED JURY DEMO (Best Practice)")
    print("   Calibration + Personas = Maximum evaluation quality")
    print("="*70)
    
    print("\n📊 Workflow:")
    print("   Step 1: Multiple judges calibrate on shared rubric")
    print("   Step 2: Jury of personas evaluate using rubric")
    print("   Step 3: Aggregate with disagreement tracking")
    
    print("\n" + "-"*50)
    print("Running Full Calibrated Jury Evaluation")
    print("-"*50)
    
    evaluator = create_calibrated_evaluator(
        judge_models=["qwen2.5-coder:3b", "llama3.2:3b"],
        personas=["skeptic", "pragmatist"]  # Focus on critical assessment
    )
    
    trace = create_good_trace()
    result = evaluator.evaluate(trace, task_type="security_analysis")
    
    print(f"\n   ✅ FINAL SCORE: {result['aggregated_score'].score}/10")
    
    print(f"\n   📋 Calibration Used:")
    print(f"      Task Type: {result['calibration']['task_type']}")
    print(f"      Criteria: {[c['name'] for c in result['calibration']['criteria']]}")
    print(f"      Rounds: {result['calibration']['calibration_rounds']}")
    
    print(f"\n   👥 Persona Scores:")
    for persona, data in result['persona_scores'].items():
        print(f"      {persona}: {data['score']}/10")
    
    print(f"\n   🔍 Disagreement: {result['disagreement']}")
    
    # Store results
    get_collector().add_calibrated_jury_result(result)
    
    print("\n💡 Key Insight: Calibrated Jury is recommended for high-stakes")
    print("   evaluations where consistency and multiple perspectives matter.")


def demo_all():
    """Run all demos."""
    print("\n" + "#"*70)
    print("#  APEE ADVANCED EVALUATION PATTERNS - COMPLETE DEMO")
    print("#"*70)
    print("""
This demo showcases the 3 newly implemented evaluation patterns:

1. PROGRESSIVE DEEPENING (Sequential)
   - Fail-fast with escalating depth
   - Saves 60-80% tokens on obvious cases

2. JURY WITH PERSONAS (Independent)  
   - 4 perspectives: Skeptic, Literalist, Optimist, Pragmatist
   - Reduces single-viewpoint bias

3. CALIBRATION LOOP (Iterative)
   - Judges negotiate rubric before scoring
   - Ensures consistent evaluation standards

4. CALIBRATED JURY (Hybrid - Best Practice)
   - Combines Calibration + Personas
   - Maximum quality for high-stakes evaluation
""")
    
    demo_progressive_deepening()
    demo_jury_with_personas()
    demo_calibration_loop()
    demo_calibrated_jury()
    
    print("\n" + "#"*70)
    print("#  ALL DEMOS COMPLETE!")
    print("#"*70)
    print("""
Summary of When to Use Each Pattern:

| Pattern               | Best For                              | Token Cost |
|-----------------------|---------------------------------------|------------|
| Progressive Deepening | High-volume, cost-conscious           | Low-Medium |
| Jury with Personas    | Subjective tasks, need perspectives   | Medium     |
| Calibration Loop      | Novel tasks, unclear criteria         | Medium     |
| Calibrated Jury       | High-stakes, maximum quality          | High       |

For production use:
- Start with Progressive Deepening for bulk evaluation
- Use Calibrated Jury for final/important decisions
""")


def main():
    parser = argparse.ArgumentParser(
        description="Demo APEE Advanced Evaluation Patterns"
    )
    parser.add_argument(
        "--pattern",
        choices=["progressive", "jury", "calibration", "calibrated", "all"],
        default="all",
        help="Which pattern to demo (default: all)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path for JSON report (default: data/advanced_evaluation_results_<timestamp>.json)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to JSON"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would run without LLM calls"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("🏃 Dry run mode - showing available demos")
        print("\nAvailable patterns:")
        print("  --pattern progressive  Progressive Deepening (fail-fast)")
        print("  --pattern jury         Jury with Personas (4 perspectives)")
        print("  --pattern calibration  Calibration Loop (negotiated rubric)")
        print("  --pattern calibrated   Calibrated Jury (best practice)")
        print("  --pattern all          Run all demos")
        print("\nTo run: python examples/advanced_evaluation_demo.py --pattern all")
        return
    
    # Reset collector for fresh run
    reset_collector()
    
    if args.pattern == "progressive":
        demo_progressive_deepening()
    elif args.pattern == "jury":
        demo_jury_with_personas()
    elif args.pattern == "calibration":
        demo_calibration_loop()
    elif args.pattern == "calibrated":
        demo_calibrated_jury()
    else:
        demo_all()
    
    # Save results unless --no-save
    if not args.no_save:
        output_path = get_collector().save(args.output)
        print(f"\n📄 Results saved to: {output_path}")


if __name__ == "__main__":
    main()
