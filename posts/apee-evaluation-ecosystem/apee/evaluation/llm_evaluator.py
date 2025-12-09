"""
LLM-as-a-Judge Evaluators for APEE Framework.

Following the CrewAI pattern: Use LLMs to evaluate agent outputs
rather than simple heuristics. This provides meaningful, nuanced
evaluation that can understand context, reasoning quality, and 
goal alignment.

Reference: https://github.com/crewAIInc/crewAI/tree/main/lib/crewai/src/crewai/experimental/evaluation
"""

import json
import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field

import httpx


class MetricCategory(Enum):
    """Categories of evaluation metrics."""
    GOAL_ALIGNMENT = "goal_alignment"
    SEMANTIC_QUALITY = "semantic_quality"
    REASONING_EFFICIENCY = "reasoning_efficiency"
    COLLABORATION_EFFECTIVENESS = "collaboration_effectiveness"
    SYNTHESIS_QUALITY = "synthesis_quality"
    CONFLICT_RESOLUTION = "conflict_resolution"


class EvaluationScore(BaseModel):
    """Result of an evaluation with score and feedback."""
    score: Optional[float] = Field(
        default=5.0,
        description="Numeric score from 0-10 where 0 is worst and 10 is best",
        ge=0.0,
        le=10.0,
    )
    feedback: str = Field(
        default="",
        description="Detailed feedback explaining the evaluation score"
    )
    subcategory_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown scores for subcategories"
    )
    raw_response: Optional[str] = Field(
        default=None,
        description="Raw response from the evaluator LLM"
    )
    
    def __str__(self) -> str:
        if self.score is None:
            return f"Score: N/A - {self.feedback}"
        return f"Score: {self.score:.1f}/10 - {self.feedback}"


class ExecutionTrace(BaseModel):
    """Trace of agent execution for evaluation."""
    agent_id: str
    agent_role: str
    task_description: str
    expected_output: Optional[str] = None
    final_output: str
    llm_calls: list[dict] = Field(default_factory=list)
    tool_uses: list[dict] = Field(default_factory=list)
    messages_exchanged: list[dict] = Field(default_factory=list)
    duration_seconds: float = 0.0
    token_count: int = 0


class CollaborativeTrace(BaseModel):
    """Trace of multi-agent collaboration for evaluation."""
    scenario_id: str
    scenario_description: str
    collaboration_pattern: str  # debate, pipeline, parallel, etc.
    participating_agents: list[str]
    agent_traces: list[ExecutionTrace]
    coordination_messages: list[dict] = Field(default_factory=list)
    conflicts_detected: list[dict] = Field(default_factory=list)
    final_synthesized_output: Optional[str] = None
    total_duration_seconds: float = 0.0


def extract_json_from_response(response: str) -> dict[str, Any]:
    """Extract JSON from LLM response, handling markdown code blocks."""
    # Try to find JSON in code blocks first
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to find raw JSON
    try:
        # Find the first { and last }
        start = response.find('{')
        end = response.rfind('}') + 1
        if start != -1 and end > start:
            return json.loads(response[start:end])
    except json.JSONDecodeError:
        pass
    
    # Return empty dict if extraction fails
    return {}


class BaseEvaluator(ABC):
    """Base class for LLM-based evaluators."""
    
    def __init__(
        self, 
        model: str = "qwen2.5-coder:7b",
        base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.base_url = base_url
    
    @property
    @abstractmethod
    def metric_category(self) -> MetricCategory:
        """The category of metric this evaluator computes."""
        pass
    
    @abstractmethod
    def evaluate(
        self,
        trace: ExecutionTrace | CollaborativeTrace,
    ) -> EvaluationScore:
        """Evaluate the trace and return a score."""
        pass
    
    def _call_llm(self, messages: list[dict]) -> str:
        """Call the evaluation LLM."""
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.1}  # Low temp for consistent evaluation
                }
            )
            response.raise_for_status()
            return response.json()["message"]["content"]


class GoalAlignmentEvaluator(BaseEvaluator):
    """Evaluates how well agent output aligns with task goals."""
    
    @property
    def metric_category(self) -> MetricCategory:
        return MetricCategory.GOAL_ALIGNMENT
    
    def evaluate(self, trace: ExecutionTrace) -> EvaluationScore:
        messages = [
            {
                "role": "system",
                "content": """You are an expert evaluator assessing how well an AI agent's output aligns with its assigned task goal.

Score the agent's goal alignment on a scale from 0-10 where:
- 0-2: Complete misalignment, agent did not understand or attempt the task goal
- 3-4: Poor alignment, agent attempted but missed most requirements  
- 5-6: Partial alignment, agent addressed some but not all requirements
- 7-8: Good alignment, agent satisfied most task requirements with minor gaps
- 9-10: Excellent alignment, agent fully satisfied all task requirements

Consider:
1. Did the agent correctly interpret the task goal?
2. Did the final output directly address the requirements?
3. Did the agent focus on relevant aspects of the task?
4. Did the agent provide all requested information or deliverables?
5. Is the output actionable and useful for the stated goal?

Return your evaluation as JSON with fields:
- "score": number (0-10)
- "feedback": string (detailed explanation)
- "strengths": list of strings (what the agent did well)
- "gaps": list of strings (what was missing or could be improved)
"""
            },
            {
                "role": "user",
                "content": f"""
Agent Role: {trace.agent_role}
Task Description: {trace.task_description}
Expected Output: {trace.expected_output or "Not specified"}

Agent's Final Output:
{trace.final_output[:2000]}{"..." if len(trace.final_output) > 2000 else ""}

Evaluate how well the agent's output aligns with the assigned task goal.
"""
            }
        ]
        
        try:
            response = self._call_llm(messages)
            data = extract_json_from_response(response)
            
            return EvaluationScore(
                score=float(data.get("score", 5.0)),
                feedback=data.get("feedback", response),
                subcategory_scores={
                    "interpretation": float(data.get("interpretation_score", data.get("score", 5.0))),
                    "completeness": float(data.get("completeness_score", data.get("score", 5.0))),
                },
                raw_response=response
            )
        except Exception as e:
            return EvaluationScore(
                score=None,
                feedback=f"Evaluation failed: {e}",
                raw_response=str(e)
            )


class SemanticQualityEvaluator(BaseEvaluator):
    """Evaluates the semantic quality and reasoning of agent output."""
    
    @property
    def metric_category(self) -> MetricCategory:
        return MetricCategory.SEMANTIC_QUALITY
    
    def evaluate(self, trace: ExecutionTrace) -> EvaluationScore:
        messages = [
            {
                "role": "system",
                "content": """You are an expert evaluator assessing the semantic quality of an AI agent's output.

Score the semantic quality on a scale from 0-10 where:
- 0-2: Completely incoherent, confusing, or logically flawed output
- 3-4: Poor quality with significant issues in clarity or logic
- 5-6: Moderately clear and logical output with some issues
- 7-8: Good quality with clear reasoning and organization
- 9-10: Exceptionally clear, coherent, and logically sound output

Evaluate these subcategories (each 0-10):
1. Structure: Is the output well-organized with clear sections/flow?
2. Clarity: Is the language precise and easy to understand?
3. Reasoning: Is the logic sound and well-supported?
4. Depth: Does the output show thorough analysis/understanding?
5. Coherence: Is the output internally consistent without contradictions?

Return your evaluation as JSON with fields:
- "score": number (overall 0-10)
- "scores": {"structure": n, "clarity": n, "reasoning": n, "depth": n, "coherence": n}
- "feedback": string (detailed explanation)
"""
            },
            {
                "role": "user",
                "content": f"""
Agent Role: {trace.agent_role}
Task: {trace.task_description}

Agent's Output:
{trace.final_output[:2500]}{"..." if len(trace.final_output) > 2500 else ""}

Evaluate the semantic quality and reasoning of this output.
"""
            }
        ]
        
        try:
            response = self._call_llm(messages)
            data = extract_json_from_response(response)
            
            scores = data.get("scores", {})
            return EvaluationScore(
                score=float(data.get("score", 5.0)),
                feedback=data.get("feedback", response),
                subcategory_scores={
                    "structure": float(scores.get("structure", 5.0)),
                    "clarity": float(scores.get("clarity", 5.0)),
                    "reasoning": float(scores.get("reasoning", 5.0)),
                    "depth": float(scores.get("depth", 5.0)),
                    "coherence": float(scores.get("coherence", 5.0)),
                },
                raw_response=response
            )
        except Exception as e:
            return EvaluationScore(
                score=None,
                feedback=f"Evaluation failed: {e}",
                raw_response=str(e)
            )


class CollaborationEffectivenessEvaluator(BaseEvaluator):
    """Evaluates how effectively agents collaborated."""
    
    @property
    def metric_category(self) -> MetricCategory:
        return MetricCategory.COLLABORATION_EFFECTIVENESS
    
    def evaluate(self, trace: CollaborativeTrace) -> EvaluationScore:
        # Build summary of agent contributions
        agent_summaries = []
        for agent_trace in trace.agent_traces:
            summary = f"- {agent_trace.agent_role}: {len(agent_trace.final_output)} chars output"
            agent_summaries.append(summary)
        
        messages = [
            {
                "role": "system",
                "content": """You are an expert evaluator assessing how effectively multiple AI agents collaborated on a task.

Score the collaboration effectiveness on a scale from 0-10 where:
- 0-2: No meaningful collaboration, agents worked in isolation or conflicted
- 3-4: Poor collaboration, minimal interaction or value-add from multi-agent setup
- 5-6: Moderate collaboration, some synergy but significant coordination issues
- 7-8: Good collaboration, agents complemented each other with minor gaps
- 9-10: Excellent collaboration, clear synergy and value from multi-agent approach

Evaluate these subcategories (each 0-10):
1. Coordination: Did agents work together smoothly without conflicts?
2. Synergy: Did the combined output exceed what one agent could produce?
3. Role Utilization: Did each agent contribute uniquely based on their role?
4. Information Flow: Was knowledge effectively shared between agents?
5. Pattern Fit: Was the collaboration pattern appropriate for the task?

Return your evaluation as JSON with fields:
- "score": number (overall 0-10)
- "scores": {"coordination": n, "synergy": n, "role_utilization": n, "information_flow": n, "pattern_fit": n}
- "feedback": string (detailed explanation)
- "collaboration_insights": string (what worked well/poorly in the collaboration)
"""
            },
            {
                "role": "user",
                "content": f"""
Collaboration Scenario: {trace.scenario_description}
Collaboration Pattern: {trace.collaboration_pattern}
Number of Agents: {len(trace.participating_agents)}
Agents: {', '.join(trace.participating_agents)}

Agent Contributions:
{chr(10).join(agent_summaries)}

Messages Exchanged: {len(trace.coordination_messages)}
Conflicts Detected: {len(trace.conflicts_detected)}

Final Synthesized Output:
{(trace.final_synthesized_output or "Not available")[:1500]}{"..." if trace.final_synthesized_output and len(trace.final_synthesized_output) > 1500 else ""}

Evaluate how effectively the agents collaborated on this task.
"""
            }
        ]
        
        try:
            response = self._call_llm(messages)
            data = extract_json_from_response(response)
            
            scores = data.get("scores", {})
            return EvaluationScore(
                score=float(data.get("score", 5.0)),
                feedback=data.get("feedback", response),
                subcategory_scores={
                    "coordination": float(scores.get("coordination", 5.0)),
                    "synergy": float(scores.get("synergy", 5.0)),
                    "role_utilization": float(scores.get("role_utilization", 5.0)),
                    "information_flow": float(scores.get("information_flow", 5.0)),
                    "pattern_fit": float(scores.get("pattern_fit", 5.0)),
                },
                raw_response=response
            )
        except Exception as e:
            return EvaluationScore(
                score=None,
                feedback=f"Evaluation failed: {e}",
                raw_response=str(e)
            )


class SynthesisQualityEvaluator(BaseEvaluator):
    """Evaluates the quality of synthesized multi-agent output."""
    
    @property
    def metric_category(self) -> MetricCategory:
        return MetricCategory.SYNTHESIS_QUALITY
    
    def evaluate(self, trace: CollaborativeTrace) -> EvaluationScore:
        # Collect individual outputs
        individual_outputs = []
        for agent_trace in trace.agent_traces:
            individual_outputs.append(
                f"[{agent_trace.agent_role}]:\n{agent_trace.final_output[:500]}..."
            )
        
        messages = [
            {
                "role": "system",
                "content": """You are an expert evaluator assessing the quality of synthesized output from multiple AI agents.

Score the synthesis quality on a scale from 0-10 where:
- 0-2: No synthesis, just concatenation or one agent's output
- 3-4: Poor synthesis, major perspectives or contributions lost
- 5-6: Moderate synthesis, basic integration but lacking depth
- 7-8: Good synthesis, perspectives well integrated with minor gaps  
- 9-10: Excellent synthesis, seamless integration with emergent insights

Evaluate these subcategories (each 0-10):
1. Integration: How well are different perspectives combined?
2. Coherence: Is the synthesized output internally consistent?
3. Completeness: Are all important contributions represented?
4. Added Value: Does synthesis provide insights beyond individual outputs?
5. Balance: Are different viewpoints fairly represented?

Return your evaluation as JSON with fields:
- "score": number (overall 0-10)
- "scores": {"integration": n, "coherence": n, "completeness": n, "added_value": n, "balance": n}
- "feedback": string (detailed explanation)
"""
            },
            {
                "role": "user",
                "content": f"""
Task: {trace.scenario_description}
Number of Contributors: {len(trace.agent_traces)}

Individual Agent Outputs (truncated):
{chr(10).join(individual_outputs)}

Final Synthesized Output:
{(trace.final_synthesized_output or "Not available")[:2000]}

Evaluate how well the individual contributions were synthesized into the final output.
"""
            }
        ]
        
        try:
            response = self._call_llm(messages)
            data = extract_json_from_response(response)
            
            scores = data.get("scores", {})
            return EvaluationScore(
                score=float(data.get("score", 5.0)),
                feedback=data.get("feedback", response),
                subcategory_scores={
                    "integration": float(scores.get("integration", 5.0)),
                    "coherence": float(scores.get("coherence", 5.0)),
                    "completeness": float(scores.get("completeness", 5.0)),
                    "added_value": float(scores.get("added_value", 5.0)),
                    "balance": float(scores.get("balance", 5.0)),
                },
                raw_response=response
            )
        except Exception as e:
            return EvaluationScore(
                score=None,
                feedback=f"Evaluation failed: {e}",
                raw_response=str(e)
            )


class APEEEvaluator:
    """
    Main APEE evaluator that orchestrates multiple evaluation dimensions.
    
    Implements the three-tier APEE evaluation:
    - Level 1 (Individual): Goal alignment, semantic quality per agent
    - Level 2 (Collaborative): Collaboration effectiveness, synthesis quality
    - Level 3 (Ecosystem): System health, adaptability metrics
    """
    
    def __init__(
        self,
        model: str = "qwen2.5-coder:7b",
        base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.base_url = base_url
        
        # Initialize evaluators
        self.goal_evaluator = GoalAlignmentEvaluator(model, base_url)
        self.semantic_evaluator = SemanticQualityEvaluator(model, base_url)
        self.collaboration_evaluator = CollaborationEffectivenessEvaluator(model, base_url)
        self.synthesis_evaluator = SynthesisQualityEvaluator(model, base_url)
    
    def evaluate_individual(
        self, 
        trace: ExecutionTrace
    ) -> dict[MetricCategory, EvaluationScore]:
        """Evaluate individual agent performance (Level 1)."""
        return {
            MetricCategory.GOAL_ALIGNMENT: self.goal_evaluator.evaluate(trace),
            MetricCategory.SEMANTIC_QUALITY: self.semantic_evaluator.evaluate(trace),
        }
    
    def evaluate_collaborative(
        self,
        trace: CollaborativeTrace
    ) -> dict[MetricCategory, EvaluationScore]:
        """Evaluate multi-agent collaboration (Level 2)."""
        return {
            MetricCategory.COLLABORATION_EFFECTIVENESS: self.collaboration_evaluator.evaluate(trace),
            MetricCategory.SYNTHESIS_QUALITY: self.synthesis_evaluator.evaluate(trace),
        }
    
    def evaluate_full(
        self,
        collaborative_trace: CollaborativeTrace
    ) -> dict[str, Any]:
        """
        Run full APEE evaluation on a collaborative scenario.
        
        Returns three-tier metrics:
        - Level 1: Individual agent scores (averaged across agents)
        - Level 2: Collaborative metrics
        - Level 3: Ecosystem metrics (derived from L1+L2)
        """
        # Level 1: Evaluate each agent individually
        individual_scores = {}
        for agent_trace in collaborative_trace.agent_traces:
            agent_metrics = self.evaluate_individual(agent_trace)
            individual_scores[agent_trace.agent_role] = {
                cat.value: score.model_dump() for cat, score in agent_metrics.items()
            }
        
        # Average L1 scores across agents
        l1_avg = self._average_individual_scores(individual_scores)
        
        # Level 2: Evaluate collaboration
        collaborative_metrics = self.evaluate_collaborative(collaborative_trace)
        l2_scores = {
            cat.value: score.model_dump() for cat, score in collaborative_metrics.items()
        }
        l2_avg = sum(
            s.score for s in collaborative_metrics.values() if s.score is not None
        ) / max(1, len([s for s in collaborative_metrics.values() if s.score is not None]))
        
        # Level 3: Ecosystem metrics (derived)
        l3_scores = self._compute_ecosystem_metrics(
            l1_avg, l2_avg, collaborative_trace
        )
        
        # Compute overall APEE score (weighted)
        # L1: 30%, L2: 45% (collaborative focus), L3: 25%
        overall = (l1_avg * 0.30 + l2_avg * 0.45 + l3_scores["overall"] * 0.25)
        
        return {
            "level1_individual": {
                "scores_by_agent": individual_scores,
                "average": round(l1_avg, 2),
            },
            "level2_collaborative": {
                "scores": l2_scores,
                "average": round(l2_avg, 2),
            },
            "level3_ecosystem": l3_scores,
            "overall_apee_score": round(overall, 2),
            "metadata": {
                "scenario_id": collaborative_trace.scenario_id,
                "pattern": collaborative_trace.collaboration_pattern,
                "num_agents": len(collaborative_trace.participating_agents),
                "duration_seconds": collaborative_trace.total_duration_seconds,
            }
        }
    
    def _average_individual_scores(
        self, 
        individual_scores: dict[str, dict]
    ) -> float:
        """Compute average score across all agents and metrics."""
        all_scores = []
        for agent_metrics in individual_scores.values():
            for metric in agent_metrics.values():
                if metric.get("score") is not None:
                    all_scores.append(metric["score"])
        return sum(all_scores) / max(1, len(all_scores))
    
    def _compute_ecosystem_metrics(
        self,
        l1_avg: float,
        l2_avg: float,
        trace: CollaborativeTrace
    ) -> dict[str, Any]:
        """Compute Level 3 ecosystem metrics."""
        # Efficiency: Output quality per unit time
        total_output = sum(
            len(at.final_output) for at in trace.agent_traces
        )
        efficiency = min(10.0, total_output / max(1, trace.total_duration_seconds) / 100)
        
        # Stability: Inverse of conflicts
        stability = max(0, 10.0 - len(trace.conflicts_detected) * 2)
        
        # Throughput: Agents utilized effectively
        active_agents = len([
            at for at in trace.agent_traces 
            if len(at.final_output) > 50
        ])
        throughput = (active_agents / max(1, len(trace.agent_traces))) * 10
        
        # Adaptability: Pattern appropriateness (based on L2 score)
        adaptability = l2_avg
        
        overall = (efficiency + stability + throughput + adaptability) / 4
        
        return {
            "efficiency": round(efficiency, 2),
            "stability": round(stability, 2),
            "throughput": round(throughput, 2),
            "adaptability": round(adaptability, 2),
            "overall": round(overall, 2),
        }


def create_default_evaluator(
    model: str = "qwen2.5-coder:7b",
    base_url: str = "http://localhost:11434"
) -> APEEEvaluator:
    """Factory function to create a default APEE evaluator."""
    return APEEEvaluator(model=model, base_url=base_url)


class EnsembleEvaluator:
    """
    Ensemble evaluator using multiple model families to reduce bias.
    
    Research shows that using the same model as both executor and judge
    leads to self-preference bias. This ensemble approach:
    1. Uses models from DIFFERENT families as judges
    2. Aggregates scores (mean, median, or weighted)
    3. Detects high disagreement (potential evaluation uncertainty)
    
    Best practice: Use models from different families than the agents being evaluated.
    E.g., If agents use Qwen, use Llama+Gemma as judges.
    """
    
    def __init__(
        self,
        judge_models: list[str],
        base_url: str = "http://localhost:11434",
        aggregation: str = "median",  # mean, median, weighted
    ):
        """
        Initialize ensemble evaluator.
        
        Args:
            judge_models: List of model names from DIFFERENT families
                         e.g., ["llama3.2:3b", "gemma3:4b"]
            base_url: Ollama API URL
            aggregation: How to combine scores - "mean", "median", or "weighted"
        """
        if len(judge_models) < 2:
            raise ValueError("Ensemble requires at least 2 judge models from different families")
        
        self.judge_models = judge_models
        self.base_url = base_url
        self.aggregation = aggregation
        
        # Create evaluators for each judge model
        self.evaluators = [
            APEEEvaluator(model=model, base_url=base_url)
            for model in judge_models
        ]
    
    def evaluate_full(
        self,
        collaborative_trace: CollaborativeTrace
    ) -> dict[str, Any]:
        """
        Run ensemble evaluation using multiple judge models.
        
        Returns aggregated scores plus disagreement metrics.
        """
        # Collect evaluations from all judges
        judge_results = []
        for i, evaluator in enumerate(self.evaluators):
            try:
                result = evaluator.evaluate_full(collaborative_trace)
                result["_judge_model"] = self.judge_models[i]
                judge_results.append(result)
            except Exception as e:
                # Log but continue with other judges
                print(f"Judge {self.judge_models[i]} failed: {e}")
        
        if not judge_results:
            raise RuntimeError("All judge models failed")
        
        # Aggregate scores
        aggregated = self._aggregate_results(judge_results)
        
        # Compute disagreement metrics
        disagreement = self._compute_disagreement(judge_results)
        
        return {
            **aggregated,
            "ensemble_metadata": {
                "judge_models": self.judge_models,
                "aggregation_method": self.aggregation,
                "num_successful_judges": len(judge_results),
                "disagreement": disagreement,
                "individual_judge_scores": [
                    {
                        "model": r["_judge_model"],
                        "overall": r["overall_apee_score"],
                        "l1": r["level1_individual"]["average"],
                        "l2": r["level2_collaborative"]["average"],
                    }
                    for r in judge_results
                ]
            }
        }
    
    def _aggregate_results(
        self, 
        results: list[dict]
    ) -> dict[str, Any]:
        """Aggregate results from multiple judges."""
        import statistics
        
        # Extract key scores
        overall_scores = [r["overall_apee_score"] for r in results]
        l1_scores = [r["level1_individual"]["average"] for r in results]
        l2_scores = [r["level2_collaborative"]["average"] for r in results]
        l3_scores = [r["level3_ecosystem"]["overall"] for r in results]
        
        # Aggregate based on method
        if self.aggregation == "median":
            agg_fn = statistics.median
        elif self.aggregation == "mean":
            agg_fn = statistics.mean
        else:  # weighted - weight by inverse variance (more confident judges weighted higher)
            def agg_fn(scores):
                if len(scores) <= 1:
                    return scores[0] if scores else 0
                mean = statistics.mean(scores)
                variance = statistics.variance(scores)
                if variance < 0.1:  # Low variance = high agreement
                    return mean
                # Weight towards median when high variance
                return (mean + statistics.median(scores)) / 2
        
        # Use first result as template, update with aggregated scores
        base = results[0].copy()
        base["overall_apee_score"] = round(agg_fn(overall_scores), 2)
        base["level1_individual"]["average"] = round(agg_fn(l1_scores), 2)
        base["level2_collaborative"]["average"] = round(agg_fn(l2_scores), 2)
        base["level3_ecosystem"]["overall"] = round(agg_fn(l3_scores), 2)
        
        return base
    
    def _compute_disagreement(
        self, 
        results: list[dict]
    ) -> dict[str, float]:
        """Compute disagreement metrics between judges."""
        import statistics
        
        overall_scores = [r["overall_apee_score"] for r in results]
        l1_scores = [r["level1_individual"]["average"] for r in results]
        l2_scores = [r["level2_collaborative"]["average"] for r in results]
        
        def safe_stdev(scores):
            if len(scores) < 2:
                return 0.0
            return statistics.stdev(scores)
        
        def score_range(scores):
            return max(scores) - min(scores)
        
        return {
            "overall_stdev": round(safe_stdev(overall_scores), 2),
            "overall_range": round(score_range(overall_scores), 2),
            "l1_stdev": round(safe_stdev(l1_scores), 2),
            "l2_stdev": round(safe_stdev(l2_scores), 2),
            "high_disagreement": score_range(overall_scores) > 2.0,  # Flag if >2 point spread
        }


def create_ensemble_evaluator(
    judge_models: list[str] = None,
    base_url: str = "http://localhost:11434",
    aggregation: str = "median",
) -> EnsembleEvaluator:
    """
    Factory to create an ensemble evaluator with sensible defaults.
    
    Default uses Llama and Gemma families as judges (different from Qwen agents).
    """
    if judge_models is None:
        # Default: Use different model families than typical agents
        judge_models = ["llama3.2:3b", "gemma3:4b"]
    
    return EnsembleEvaluator(
        judge_models=judge_models,
        base_url=base_url,
        aggregation=aggregation,
    )

