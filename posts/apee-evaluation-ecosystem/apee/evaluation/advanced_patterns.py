"""
Advanced APEE Evaluation Patterns.

Implements evaluation-specific coordination patterns from the APEE taxonomy:

1. Calibration Loop (Iterative Pattern):
   - Judges negotiate rubric criteria BEFORE scoring
   - Ensures consistent evaluation standards across judges
   - Reduces subjectivity in ambiguous tasks

2. Jury with Personas (Independent Pattern):
   - Multiple judges with distinct evaluation personas
   - Skeptic, Literalist, Optimist perspectives
   - Aggregate scoring reduces single-perspective bias

3. Progressive Deepening (Sequential Pattern):
   - Fail-fast evaluation with escalating depth
   - Quick checks first, deep analysis only if needed
   - Saves compute on obvious pass/fail cases

Reference: APEE Pattern Matrix (Independent, Sequential, Iterative, Hybrid)
"""

import asyncio
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

import httpx
from pydantic import BaseModel, Field

from apee.evaluation.llm_evaluator import (
    EvaluationScore,
    ExecutionTrace,
    CollaborativeTrace,
    extract_json_from_response,
)


# =============================================================================
# JUDGE PERSONAS (Jury Pattern)
# =============================================================================

class JudgePersona(str, Enum):
    """
    Distinct evaluation personas for the Jury pattern.
    
    Each persona brings a different perspective:
    - SKEPTIC: Focuses on flaws, edge cases, potential failures
    - LITERALIST: Strict adherence to requirements, no assumptions
    - OPTIMIST: Values creative solutions, potential over perfection
    - PRAGMATIST: Focuses on practical utility and real-world applicability
    """
    SKEPTIC = "skeptic"
    LITERALIST = "literalist"
    OPTIMIST = "optimist"
    PRAGMATIST = "pragmatist"


@dataclass
class PersonaConfig:
    """Configuration for a judge persona."""
    persona: JudgePersona
    system_prompt_modifier: str
    weight: float = 1.0  # Voting weight (default equal)
    focus_areas: list[str] = field(default_factory=list)


# Pre-defined persona configurations
PERSONA_CONFIGS: dict[JudgePersona, PersonaConfig] = {
    JudgePersona.SKEPTIC: PersonaConfig(
        persona=JudgePersona.SKEPTIC,
        system_prompt_modifier="""You are a SKEPTICAL evaluator. Your role is to:
- Look for potential flaws, edge cases, and failure modes
- Question assumptions made by the agent
- Identify what could go wrong with this output
- Be critical but fair - acknowledge genuine strengths
- Focus on: error handling, edge cases, robustness, security concerns

When in doubt, err on the side of caution. A score of 7 from you means "solid work with no major concerns."
Lower scores indicate specific issues you've identified.""",
        weight=1.0,
        focus_areas=["error_handling", "edge_cases", "robustness", "security"]
    ),
    
    JudgePersona.LITERALIST: PersonaConfig(
        persona=JudgePersona.LITERALIST,
        system_prompt_modifier="""You are a LITERALIST evaluator. Your role is to:
- Evaluate STRICTLY against the stated requirements
- Do not give credit for "close enough" or "probably meant"
- Check if EVERY requirement is explicitly addressed
- Penalize assumptions not justified by the task description
- Focus on: requirement coverage, specification adherence, completeness

Score based purely on what was asked vs what was delivered. No partial credit for good intentions.""",
        weight=1.0,
        focus_areas=["requirements", "specification", "completeness", "accuracy"]
    ),
    
    JudgePersona.OPTIMIST: PersonaConfig(
        persona=JudgePersona.OPTIMIST,
        system_prompt_modifier="""You are an OPTIMISTIC evaluator. Your role is to:
- Recognize creative solutions and innovative approaches
- Value potential and direction over perfection
- Acknowledge effort and partial progress toward goals
- Look for strengths that could be built upon
- Focus on: creativity, innovation, potential, positive aspects

Be genuinely positive but not naive. A score of 8+ should indicate real merit, not just good vibes.""",
        weight=1.0,
        focus_areas=["creativity", "innovation", "potential", "strengths"]
    ),
    
    JudgePersona.PRAGMATIST: PersonaConfig(
        persona=JudgePersona.PRAGMATIST,
        system_prompt_modifier="""You are a PRAGMATIC evaluator. Your role is to:
- Focus on practical utility and real-world applicability
- Ask "Would this actually work in production?"
- Value maintainability, clarity, and simplicity
- Consider the cost-benefit of complexity
- Focus on: practicality, maintainability, usability, clarity

Score based on whether this output would be useful in practice, not just theoretically correct.""",
        weight=1.0,
        focus_areas=["practicality", "maintainability", "usability", "clarity"]
    ),
}


# =============================================================================
# CALIBRATION RUBRIC
# =============================================================================

class RubricCriterion(BaseModel):
    """A single criterion in the evaluation rubric."""
    name: str = Field(description="Short name for the criterion")
    description: str = Field(description="What this criterion measures")
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Weight in final score")
    score_anchors: dict[int, str] = Field(
        default_factory=dict,
        description="Score anchors: {score: description_of_what_that_score_means}"
    )


class CalibratedRubric(BaseModel):
    """
    A rubric that has been calibrated/agreed upon by multiple judges.
    
    The calibration process ensures judges share a common understanding
    of what each score level means for the specific task at hand.
    """
    task_type: str = Field(description="Type of task this rubric is for")
    criteria: list[RubricCriterion] = Field(default_factory=list)
    calibration_notes: str = Field(default="", description="Notes from calibration discussion")
    agreed_by: list[str] = Field(default_factory=list, description="Judge models that agreed")
    calibration_rounds: int = Field(default=0, description="Number of calibration iterations")


# =============================================================================
# CALIBRATION LOOP IMPLEMENTATION
# =============================================================================

class CalibrationLoop:
    """
    Implements the Calibration Loop pattern.
    
    Before evaluating, judges discuss and agree on:
    1. What criteria matter for this specific task
    2. What each score level (0-10) means for each criterion
    3. How to weight different criteria
    
    This ensures consistent evaluation even for novel or ambiguous tasks.
    
    Algorithm:
    1. Each judge proposes criteria and score anchors
    2. Judges review each other's proposals
    3. A synthesis step merges proposals into unified rubric
    4. If agreement threshold not met, iterate
    5. Once calibrated, rubric is used for all evaluations
    """
    
    def __init__(
        self,
        judge_models: list[str],
        base_url: str = "http://localhost:11434",
        max_calibration_rounds: int = 3,
        agreement_threshold: float = 0.7,
    ):
        """
        Initialize calibration loop.
        
        Args:
            judge_models: List of LLM models to use as judges
            base_url: Ollama API URL
            max_calibration_rounds: Max iterations for reaching agreement
            agreement_threshold: Minimum agreement level (0-1) to accept rubric
        """
        if len(judge_models) < 2:
            raise ValueError("Calibration requires at least 2 judges")
        
        self.judge_models = judge_models
        self.base_url = base_url
        self.max_rounds = max_calibration_rounds
        self.agreement_threshold = agreement_threshold
        
        # Cache calibrated rubrics by task type
        self._rubric_cache: dict[str, CalibratedRubric] = {}
    
    def _call_llm(self, model: str, messages: list[dict]) -> str:
        """Call an LLM model."""
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.3}  # Moderate temp for creative but stable output
                }
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
    
    async def _call_llm_async(self, model: str, messages: list[dict]) -> str:
        """Async LLM call."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.3}
                }
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
    
    def calibrate(
        self,
        task_description: str,
        task_type: str = "general",
        example_output: Optional[str] = None,
    ) -> CalibratedRubric:
        """
        Run calibration loop to create agreed-upon rubric.
        
        Args:
            task_description: Description of the task to evaluate
            task_type: Category of task (e.g., "code_review", "analysis")
            example_output: Optional example output to calibrate against
            
        Returns:
            CalibratedRubric with agreed criteria and score anchors
        """
        # Check cache first
        cache_key = f"{task_type}:{hash(task_description[:100])}"
        if cache_key in self._rubric_cache:
            return self._rubric_cache[cache_key]
        
        # Phase 1: Each judge proposes criteria
        proposals = self._gather_proposals(task_description, task_type, example_output)
        
        # Phase 2: Synthesize into unified rubric
        rubric = self._synthesize_rubric(proposals, task_description, task_type)
        
        # Phase 3: Validate agreement
        for round_num in range(self.max_rounds):
            agreement = self._check_agreement(rubric, task_description)
            
            if agreement >= self.agreement_threshold:
                rubric.calibration_rounds = round_num + 1
                rubric.agreed_by = self.judge_models
                break
            
            # Refine rubric based on disagreements
            rubric = self._refine_rubric(rubric, task_description)
        
        # Cache and return
        self._rubric_cache[cache_key] = rubric
        return rubric
    
    def _gather_proposals(
        self,
        task_description: str,
        task_type: str,
        example_output: Optional[str],
    ) -> list[dict]:
        """Each judge proposes evaluation criteria."""
        proposals = []
        
        example_section = ""
        if example_output:
            example_section = f"""

Example Output to Consider:
{example_output[:1000]}{"..." if len(example_output) > 1000 else ""}
"""
        
        for model in self.judge_models:
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert evaluator helping to establish evaluation criteria.

Your task is to propose 3-5 evaluation criteria for assessing AI agent outputs.
For each criterion, define:
1. Name (short identifier)
2. Description (what it measures)
3. Weight (0.0-1.0, total should sum to ~1.0)
4. Score anchors (what scores 2, 5, 8 look like for this criterion)

Return as JSON:
{
    "criteria": [
        {
            "name": "criterion_name",
            "description": "what this measures",
            "weight": 0.25,
            "score_anchors": {
                "2": "description of poor performance",
                "5": "description of adequate performance",
                "8": "description of excellent performance"
            }
        }
    ],
    "reasoning": "why these criteria matter for this task"
}"""
                },
                {
                    "role": "user",
                    "content": f"""Task Type: {task_type}

Task Description:
{task_description}
{example_section}
Propose evaluation criteria appropriate for this specific task."""
                }
            ]
            
            try:
                response = self._call_llm(model, messages)
                data = extract_json_from_response(response)
                data["_model"] = model
                proposals.append(data)
            except Exception as e:
                proposals.append({
                    "_model": model,
                    "_error": str(e),
                    "criteria": []
                })
        
        return proposals
    
    def _synthesize_rubric(
        self,
        proposals: list[dict],
        task_description: str,
        task_type: str,
    ) -> CalibratedRubric:
        """Merge proposals into unified rubric."""
        # Use first judge to synthesize (could rotate)
        synthesizer = self.judge_models[0]
        
        # Format proposals for synthesis
        proposals_text = ""
        for i, prop in enumerate(proposals):
            model = prop.get("_model", f"Judge {i+1}")
            criteria = prop.get("criteria", [])
            proposals_text += f"\n\n--- {model}'s Proposal ---\n"
            for c in criteria:
                proposals_text += f"- {c.get('name', 'unnamed')}: {c.get('description', 'no desc')} (weight: {c.get('weight', 0.25)})\n"
        
        messages = [
            {
                "role": "system",
                "content": """You are synthesizing multiple judges' proposals into a unified evaluation rubric.

Your task:
1. Identify common themes across proposals
2. Merge similar criteria (avoid duplication)
3. Resolve conflicts by taking the more specific definition
4. Ensure weights sum to approximately 1.0
5. Create clear, unambiguous score anchors

Return as JSON:
{
    "criteria": [
        {
            "name": "criterion_name",
            "description": "merged description",
            "weight": 0.25,
            "score_anchors": {
                "2": "poor performance description",
                "5": "adequate performance description", 
                "8": "excellent performance description"
            }
        }
    ],
    "calibration_notes": "notes on how proposals were merged and any conflicts resolved"
}"""
            },
            {
                "role": "user",
                "content": f"""Task: {task_description}

Judge Proposals:
{proposals_text}

Synthesize these into a unified rubric with 3-5 criteria."""
            }
        ]
        
        try:
            response = self._call_llm(synthesizer, messages)
            data = extract_json_from_response(response)
            
            criteria = []
            for c in data.get("criteria", []):
                criteria.append(RubricCriterion(
                    name=c.get("name", "unnamed"),
                    description=c.get("description", ""),
                    weight=float(c.get("weight", 0.25)),
                    score_anchors={
                        int(k): v for k, v in c.get("score_anchors", {}).items()
                    }
                ))
            
            return CalibratedRubric(
                task_type=task_type,
                criteria=criteria,
                calibration_notes=data.get("calibration_notes", ""),
            )
        except Exception as e:
            # Fallback to default rubric
            return self._default_rubric(task_type)
    
    def _check_agreement(
        self,
        rubric: CalibratedRubric,
        task_description: str,
    ) -> float:
        """Check how much judges agree with the synthesized rubric."""
        agreements = []
        
        rubric_text = self._format_rubric_for_display(rubric)
        
        for model in self.judge_models:
            messages = [
                {
                    "role": "system",
                    "content": """You are reviewing a proposed evaluation rubric.

Rate your agreement with this rubric on a scale of 0-10:
- 0-3: Major disagreements, would evaluate very differently
- 4-6: Some concerns but acceptable
- 7-10: Agree with criteria and score anchors

Return JSON: {"agreement": <0-10>, "concerns": "brief description of any concerns"}"""
                },
                {
                    "role": "user",
                    "content": f"""Task: {task_description}

Proposed Rubric:
{rubric_text}

Rate your agreement with this rubric."""
                }
            ]
            
            try:
                response = self._call_llm(model, messages)
                data = extract_json_from_response(response)
                agreement = float(data.get("agreement", 5)) / 10.0
                agreements.append(agreement)
            except Exception:
                agreements.append(0.5)  # Default to neutral on error
        
        return statistics.mean(agreements) if agreements else 0.5
    
    def _refine_rubric(
        self,
        rubric: CalibratedRubric,
        task_description: str,
    ) -> CalibratedRubric:
        """Refine rubric based on judge feedback."""
        # Collect concerns from all judges
        concerns = []
        rubric_text = self._format_rubric_for_display(rubric)
        
        for model in self.judge_models:
            messages = [
                {
                    "role": "system",
                    "content": """Identify specific improvements needed for this rubric.
Return JSON: {"improvements": ["specific improvement 1", "specific improvement 2"]}"""
                },
                {
                    "role": "user",
                    "content": f"""Task: {task_description}

Current Rubric:
{rubric_text}

What specific improvements would make this rubric better?"""
                }
            ]
            
            try:
                response = self._call_llm(model, messages)
                data = extract_json_from_response(response)
                concerns.extend(data.get("improvements", []))
            except Exception:
                pass
        
        if not concerns:
            return rubric
        
        # Apply improvements
        messages = [
            {
                "role": "system",
                "content": """Improve the rubric based on feedback.
Return the improved rubric in the same JSON format as before."""
            },
            {
                "role": "user",
                "content": f"""Current Rubric:
{rubric_text}

Improvement Suggestions:
{chr(10).join(f"- {c}" for c in concerns[:5])}

Return improved rubric as JSON."""
            }
        ]
        
        try:
            response = self._call_llm(self.judge_models[0], messages)
            data = extract_json_from_response(response)
            
            criteria = []
            for c in data.get("criteria", rubric.criteria):
                if isinstance(c, RubricCriterion):
                    criteria.append(c)
                else:
                    criteria.append(RubricCriterion(
                        name=c.get("name", "unnamed"),
                        description=c.get("description", ""),
                        weight=float(c.get("weight", 0.25)),
                        score_anchors={
                            int(k): v for k, v in c.get("score_anchors", {}).items()
                        }
                    ))
            
            rubric.criteria = criteria
            rubric.calibration_notes += f"\nRefined based on: {'; '.join(concerns[:3])}"
        except Exception:
            pass
        
        return rubric
    
    def _format_rubric_for_display(self, rubric: CalibratedRubric) -> str:
        """Format rubric as readable text."""
        lines = [f"Task Type: {rubric.task_type}\n"]
        
        for c in rubric.criteria:
            lines.append(f"## {c.name} (weight: {c.weight})")
            lines.append(f"   {c.description}")
            for score, anchor in sorted(c.score_anchors.items()):
                lines.append(f"   Score {score}: {anchor}")
            lines.append("")
        
        if rubric.calibration_notes:
            lines.append(f"Notes: {rubric.calibration_notes}")
        
        return "\n".join(lines)
    
    def _default_rubric(self, task_type: str) -> CalibratedRubric:
        """Fallback default rubric."""
        return CalibratedRubric(
            task_type=task_type,
            criteria=[
                RubricCriterion(
                    name="goal_alignment",
                    description="How well the output addresses the task goal",
                    weight=0.4,
                    score_anchors={2: "Misses the goal", 5: "Partially addresses", 8: "Fully addresses"}
                ),
                RubricCriterion(
                    name="quality",
                    description="Overall quality and correctness",
                    weight=0.35,
                    score_anchors={2: "Poor quality", 5: "Acceptable", 8: "High quality"}
                ),
                RubricCriterion(
                    name="clarity",
                    description="Clarity and organization of output",
                    weight=0.25,
                    score_anchors={2: "Confusing", 5: "Understandable", 8: "Very clear"}
                ),
            ],
            calibration_notes="Default rubric (calibration failed)",
        )


# =============================================================================
# JURY WITH PERSONAS IMPLEMENTATION
# =============================================================================

class JuryEvaluator:
    """
    Implements the Jury/Voting pattern with distinct personas.
    
    Multiple judges with different perspectives (Skeptic, Literalist, Optimist, 
    Pragmatist) evaluate the same output. The final score is aggregated across
    all perspectives, reducing single-viewpoint bias.
    
    Algorithm:
    1. Each persona evaluates independently using their lens
    2. Scores are collected with persona-specific feedback
    3. Aggregation combines scores (weighted voting)
    4. Disagreement is tracked to identify contentious outputs
    """
    
    def __init__(
        self,
        model: str = "qwen2.5-coder:7b",
        base_url: str = "http://localhost:11434",
        personas: Optional[list[JudgePersona]] = None,
        aggregation: str = "weighted_mean",  # mean, median, weighted_mean
    ):
        """
        Initialize jury evaluator.
        
        Args:
            model: LLM model to use (same model, different personas)
            base_url: Ollama API URL
            personas: Which personas to include (default: all 4)
            aggregation: How to combine scores
        """
        self.model = model
        self.base_url = base_url
        self.aggregation = aggregation
        
        # Default to all personas
        self.personas = personas or list(JudgePersona)
        self.persona_configs = {p: PERSONA_CONFIGS[p] for p in self.personas}
    
    def _call_llm(self, messages: list[dict]) -> str:
        """Call LLM."""
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.2}  # Low temp for consistent evaluation
                }
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
    
    def evaluate(
        self,
        trace: Union[ExecutionTrace, CollaborativeTrace],
        rubric: Optional[CalibratedRubric] = None,
    ) -> dict[str, Any]:
        """
        Evaluate using all jury personas.
        
        Args:
            trace: Execution or collaborative trace to evaluate
            rubric: Optional calibrated rubric (if using Calibration Loop)
            
        Returns:
            Dict with aggregated score, per-persona scores, and disagreement metrics
        """
        persona_scores: dict[JudgePersona, EvaluationScore] = {}
        
        # Each persona evaluates
        for persona in self.personas:
            config = self.persona_configs[persona]
            score = self._evaluate_with_persona(trace, config, rubric)
            persona_scores[persona] = score
        
        # Aggregate scores
        aggregated = self._aggregate_scores(persona_scores)
        
        # Compute disagreement
        disagreement = self._compute_disagreement(persona_scores)
        
        return {
            "aggregated_score": aggregated,
            "persona_scores": {
                p.value: {
                    "score": s.score,
                    "feedback": s.feedback,
                    "focus_areas": self.persona_configs[p].focus_areas,
                }
                for p, s in persona_scores.items()
            },
            "disagreement": disagreement,
            "jury_metadata": {
                "model": self.model,
                "personas": [p.value for p in self.personas],
                "aggregation_method": self.aggregation,
            }
        }
    
    def _evaluate_with_persona(
        self,
        trace: Union[ExecutionTrace, CollaborativeTrace],
        config: PersonaConfig,
        rubric: Optional[CalibratedRubric],
    ) -> EvaluationScore:
        """Evaluate from a specific persona's perspective."""
        
        # Build rubric section if provided
        rubric_section = ""
        if rubric:
            rubric_section = "\n\nEvaluation Rubric:\n"
            for c in rubric.criteria:
                rubric_section += f"- {c.name} (weight {c.weight}): {c.description}\n"
                for score, anchor in sorted(c.score_anchors.items()):
                    rubric_section += f"  Score {score}: {anchor}\n"
        
        # Build trace summary
        if isinstance(trace, ExecutionTrace):
            trace_summary = f"""
Agent Role: {trace.agent_role}
Task: {trace.task_description}
Expected Output: {trace.expected_output or "Not specified"}

Agent's Output:
{trace.final_output[:2000]}{"..." if len(trace.final_output) > 2000 else ""}
"""
        else:  # CollaborativeTrace
            agent_outputs = "\n".join([
                f"- {at.agent_role}: {at.final_output[:300]}..."
                for at in trace.agent_traces
            ])
            trace_summary = f"""
Scenario: {trace.scenario_description}
Pattern: {trace.collaboration_pattern}
Agents: {', '.join(trace.participating_agents)}

Agent Outputs (truncated):
{agent_outputs}

Final Synthesized Output:
{(trace.final_synthesized_output or "Not available")[:1500]}
"""
        
        messages = [
            {
                "role": "system",
                "content": f"""You are an expert evaluator with a specific evaluation persona.

{config.system_prompt_modifier}

Score the output on a scale from 0-10.
{rubric_section}

Return your evaluation as JSON:
{{
    "score": <0-10>,
    "feedback": "detailed feedback from your perspective",
    "key_observations": ["observation 1", "observation 2"],
    "persona_specific_concerns": ["concern relevant to your focus areas"]
}}"""
            },
            {
                "role": "user",
                "content": f"""Evaluate the following from your {config.persona.value.upper()} perspective:

{trace_summary}

Focus on: {', '.join(config.focus_areas)}"""
            }
        ]
        
        try:
            response = self._call_llm(messages)
            data = extract_json_from_response(response)
            
            score_value = float(data.get("score", 5.0))
            return EvaluationScore(
                score=score_value,
                feedback=data.get("feedback", response),
                subcategory_scores={
                    area: score_value for area in config.focus_areas
                },
                raw_response=response
            )
        except Exception as e:
            return EvaluationScore(
                score=None,
                feedback=f"Persona {config.persona.value} evaluation failed: {e}",
                raw_response=str(e)
            )
    
    def _aggregate_scores(
        self,
        persona_scores: dict[JudgePersona, EvaluationScore],
    ) -> EvaluationScore:
        """Aggregate scores from all personas."""
        valid_scores = [
            (p, s) for p, s in persona_scores.items() 
            if s.score is not None
        ]
        
        if not valid_scores:
            return EvaluationScore(
                score=None,
                feedback="All persona evaluations failed",
            )
        
        scores = [s.score for _, s in valid_scores]
        
        if self.aggregation == "median":
            final_score = statistics.median(scores)
        elif self.aggregation == "mean":
            final_score = statistics.mean(scores)
        else:  # weighted_mean
            weighted_sum = sum(
                s.score * self.persona_configs[p].weight
                for p, s in valid_scores
            )
            total_weight = sum(
                self.persona_configs[p].weight
                for p, _ in valid_scores
            )
            final_score = weighted_sum / total_weight
        
        # Combine feedback
        combined_feedback = "Jury Evaluation Summary:\n"
        for persona, score in valid_scores:
            combined_feedback += f"\n[{persona.value.upper()}] Score: {score.score}/10\n"
            combined_feedback += f"  {score.feedback[:200]}...\n"
        
        return EvaluationScore(
            score=round(final_score, 2),
            feedback=combined_feedback,
            subcategory_scores={
                p.value: s.score for p, s in valid_scores
            }
        )
    
    def _compute_disagreement(
        self,
        persona_scores: dict[JudgePersona, EvaluationScore],
    ) -> dict[str, Any]:
        """Compute disagreement metrics between personas."""
        scores = [s.score for s in persona_scores.values() if s.score is not None]
        
        if len(scores) < 2:
            return {"stdev": 0, "range": 0, "high_disagreement": False}
        
        stdev = statistics.stdev(scores)
        score_range = max(scores) - min(scores)
        
        # Identify which personas disagree most
        if len(scores) >= 2:
            sorted_scores = sorted(
                [(p, s.score) for p, s in persona_scores.items() if s.score],
                key=lambda x: x[1]
            )
            lowest = sorted_scores[0]
            highest = sorted_scores[-1]
            main_disagreement = f"{lowest[0].value} ({lowest[1]}) vs {highest[0].value} ({highest[1]})"
        else:
            main_disagreement = "N/A"
        
        return {
            "stdev": round(stdev, 2),
            "range": round(score_range, 2),
            "high_disagreement": score_range > 3.0,  # >3 point spread
            "main_disagreement": main_disagreement,
        }


# =============================================================================
# COMBINED: CALIBRATED JURY EVALUATOR
# =============================================================================

class CalibratedJuryEvaluator:
    """
    Combines Calibration Loop + Jury with Personas for maximum evaluation quality.
    
    Workflow:
    1. Calibration Loop establishes agreed-upon rubric
    2. Jury evaluates using rubric with multiple personas
    3. Results include both calibration metadata and persona breakdown
    
    This is the recommended evaluator for high-stakes or ambiguous evaluations.
    """
    
    def __init__(
        self,
        judge_models: list[str],
        base_url: str = "http://localhost:11434",
        personas: Optional[list[JudgePersona]] = None,
        max_calibration_rounds: int = 2,
    ):
        """
        Initialize calibrated jury evaluator.
        
        Args:
            judge_models: Models for calibration (can be same or different from jury model)
            base_url: Ollama API URL
            personas: Jury personas to use
            max_calibration_rounds: Max calibration iterations
        """
        self.calibration_loop = CalibrationLoop(
            judge_models=judge_models,
            base_url=base_url,
            max_calibration_rounds=max_calibration_rounds,
        )
        
        # Use first judge model for jury (same model, different personas)
        self.jury = JuryEvaluator(
            model=judge_models[0],
            base_url=base_url,
            personas=personas,
        )
    
    def evaluate(
        self,
        trace: Union[ExecutionTrace, CollaborativeTrace],
        task_type: str = "general",
        force_recalibrate: bool = False,
    ) -> dict[str, Any]:
        """
        Run full calibrated jury evaluation.
        
        Args:
            trace: Trace to evaluate
            task_type: Task category for rubric caching
            force_recalibrate: If True, don't use cached rubric
            
        Returns:
            Complete evaluation with calibration and jury results
        """
        # Get task description
        if isinstance(trace, ExecutionTrace):
            task_description = trace.task_description
        else:
            task_description = trace.scenario_description
        
        # Step 1: Calibrate rubric
        if force_recalibrate:
            self.calibration_loop._rubric_cache.clear()
        
        rubric = self.calibration_loop.calibrate(
            task_description=task_description,
            task_type=task_type,
        )
        
        # Step 2: Jury evaluation with calibrated rubric
        jury_result = self.jury.evaluate(trace, rubric=rubric)
        
        return {
            **jury_result,
            "calibration": {
                "task_type": rubric.task_type,
                "criteria": [
                    {
                        "name": c.name,
                        "description": c.description,
                        "weight": c.weight,
                    }
                    for c in rubric.criteria
                ],
                "calibration_rounds": rubric.calibration_rounds,
                "agreed_by": rubric.agreed_by,
                "notes": rubric.calibration_notes,
            }
        }


# =============================================================================
# PROGRESSIVE DEEPENING IMPLEMENTATION (Sequential Pattern)
# =============================================================================

class EvaluationDepth(str, Enum):
    """
    Evaluation depth levels for Progressive Deepening.
    
    QUICK: Fast heuristic checks (regex, length, format)
    STANDARD: Normal LLM evaluation (single pass)
    DEEP: Multi-perspective or detailed analysis
    COMPREHENSIVE: Full evaluation with all patterns
    """
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"
    COMPREHENSIVE = "comprehensive"


@dataclass
class DepthConfig:
    """Configuration for each evaluation depth level."""
    depth: EvaluationDepth
    min_score_to_pass: float  # Score needed to stop (pass threshold)
    max_score_to_fail: float  # Score that triggers immediate fail
    timeout_seconds: float
    description: str


# Default depth configurations - fail-fast optimization
DEFAULT_DEPTH_CONFIGS: dict[EvaluationDepth, DepthConfig] = {
    EvaluationDepth.QUICK: DepthConfig(
        depth=EvaluationDepth.QUICK,
        min_score_to_pass=9.0,    # Only obvious wins pass quickly
        max_score_to_fail=2.0,    # Only obvious failures fail fast
        timeout_seconds=1.0,
        description="Heuristic checks: format, length, key terms"
    ),
    EvaluationDepth.STANDARD: DepthConfig(
        depth=EvaluationDepth.STANDARD,
        min_score_to_pass=8.0,    # Good scores pass
        max_score_to_fail=3.0,    # Poor scores fail
        timeout_seconds=30.0,
        description="Single LLM evaluation pass"
    ),
    EvaluationDepth.DEEP: DepthConfig(
        depth=EvaluationDepth.DEEP,
        min_score_to_pass=7.0,    # Moderate threshold
        max_score_to_fail=4.0,    # Higher fail threshold
        timeout_seconds=60.0,
        description="Multi-persona evaluation (2 perspectives)"
    ),
    EvaluationDepth.COMPREHENSIVE: DepthConfig(
        depth=EvaluationDepth.COMPREHENSIVE,
        min_score_to_pass=0.0,    # Always returns final score
        max_score_to_fail=0.0,    # No early termination
        timeout_seconds=120.0,
        description="Full jury evaluation with calibration"
    ),
}


@dataclass
class ProgressiveResult:
    """Result from Progressive Deepening evaluation."""
    final_score: EvaluationScore
    depth_reached: EvaluationDepth
    early_termination: bool
    termination_reason: str  # "pass", "fail", "completed"
    depth_scores: dict[str, float]  # Score at each depth level
    total_time_seconds: float
    tokens_saved_estimate: int  # Estimated tokens saved by early termination


class ProgressiveDeepening:
    """
    Implements Progressive Deepening evaluation pattern.
    
    This sequential pattern optimizes evaluation cost by:
    1. Running quick heuristic checks first
    2. Escalating to LLM evaluation only if uncertain
    3. Escalating to multi-perspective only if still uncertain
    4. Full comprehensive evaluation as last resort
    
    Benefits:
    - Saves 60-80% tokens on obvious pass/fail cases
    - Maintains evaluation quality for edge cases
    - Configurable thresholds per use case
    
    Algorithm:
    1. Quick Check (heuristics): If score >= 9 → PASS, if score <= 2 → FAIL
    2. Standard (single LLM): If score >= 8 → PASS, if score <= 3 → FAIL  
    3. Deep (2 personas): If score >= 7 → PASS, if score <= 4 → FAIL
    4. Comprehensive (full jury): Return final score (no early termination)
    """
    
    def __init__(
        self,
        model: str = "qwen2.5-coder:7b",
        base_url: str = "http://localhost:11434",
        depth_configs: Optional[dict[EvaluationDepth, DepthConfig]] = None,
        max_depth: EvaluationDepth = EvaluationDepth.COMPREHENSIVE,
    ):
        """
        Initialize progressive deepening evaluator.
        
        Args:
            model: LLM model for evaluation
            base_url: Ollama API URL
            depth_configs: Custom depth configurations (or use defaults)
            max_depth: Maximum depth to reach (can limit for speed)
        """
        self.model = model
        self.base_url = base_url
        self.depth_configs = depth_configs or DEFAULT_DEPTH_CONFIGS
        self.max_depth = max_depth
        
        # Depth order for sequential evaluation
        self._depth_order = [
            EvaluationDepth.QUICK,
            EvaluationDepth.STANDARD,
            EvaluationDepth.DEEP,
            EvaluationDepth.COMPREHENSIVE,
        ]
        
        # Personas for deep evaluation
        self._deep_personas = [JudgePersona.SKEPTIC, JudgePersona.PRAGMATIST]
        
        # Full jury for comprehensive
        self._jury = JuryEvaluator(
            model=model,
            base_url=base_url,
            personas=list(JudgePersona),
        )
    
    def _call_llm(self, messages: list[dict], timeout: float = 30.0) -> str:
        """Call LLM with timeout."""
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.2}
                }
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
    
    def evaluate(
        self,
        trace: Union[ExecutionTrace, CollaborativeTrace],
    ) -> ProgressiveResult:
        """
        Evaluate with progressive deepening.
        
        Runs evaluation at increasing depth levels until:
        - Score clearly passes (>= pass threshold)
        - Score clearly fails (<= fail threshold)  
        - Maximum depth reached
        
        Args:
            trace: Execution or collaborative trace to evaluate
            
        Returns:
            ProgressiveResult with final score and depth information
        """
        import time
        start_time = time.time()
        
        depth_scores: dict[str, float] = {}
        last_score: Optional[EvaluationScore] = None
        tokens_used = 0
        
        for depth in self._depth_order:
            # Check if we should stop at this depth
            if self._depth_index(depth) > self._depth_index(self.max_depth):
                break
            
            config = self.depth_configs[depth]
            
            # Run evaluation at this depth
            score = self._evaluate_at_depth(trace, depth, config)
            if score is None:
                continue
                
            last_score = score
            depth_scores[depth.value] = score.score if score.score else 0.0
            
            # Estimate tokens used (rough approximation)
            tokens_used += self._estimate_tokens(depth)
            
            # Check for early termination
            if score.score is not None:
                if score.score >= config.min_score_to_pass:
                    # Clear pass - stop evaluation
                    total_time = time.time() - start_time
                    tokens_saved = self._estimate_tokens_saved(depth)
                    
                    return ProgressiveResult(
                        final_score=score,
                        depth_reached=depth,
                        early_termination=True,
                        termination_reason="pass",
                        depth_scores=depth_scores,
                        total_time_seconds=total_time,
                        tokens_saved_estimate=tokens_saved,
                    )
                
                if score.score <= config.max_score_to_fail:
                    # Clear fail - stop evaluation
                    total_time = time.time() - start_time
                    tokens_saved = self._estimate_tokens_saved(depth)
                    
                    return ProgressiveResult(
                        final_score=score,
                        depth_reached=depth,
                        early_termination=True,
                        termination_reason="fail",
                        depth_scores=depth_scores,
                        total_time_seconds=total_time,
                        tokens_saved_estimate=tokens_saved,
                    )
        
        # Reached max depth without early termination
        total_time = time.time() - start_time
        
        return ProgressiveResult(
            final_score=last_score or self._fallback_score(),
            depth_reached=self.max_depth,
            early_termination=False,
            termination_reason="completed",
            depth_scores=depth_scores,
            total_time_seconds=total_time,
            tokens_saved_estimate=0,
        )
    
    def _depth_index(self, depth: EvaluationDepth) -> int:
        """Get numeric index of depth level."""
        return self._depth_order.index(depth)
    
    def _evaluate_at_depth(
        self,
        trace: Union[ExecutionTrace, CollaborativeTrace],
        depth: EvaluationDepth,
        config: DepthConfig,
    ) -> Optional[EvaluationScore]:
        """
        Run evaluation at a specific depth level.
        
        Args:
            trace: Trace to evaluate
            depth: Depth level
            config: Configuration for this depth
            
        Returns:
            EvaluationScore or None on error
        """
        try:
            if depth == EvaluationDepth.QUICK:
                return self._quick_evaluation(trace)
            elif depth == EvaluationDepth.STANDARD:
                return self._standard_evaluation(trace, config.timeout_seconds)
            elif depth == EvaluationDepth.DEEP:
                return self._deep_evaluation(trace, config.timeout_seconds)
            elif depth == EvaluationDepth.COMPREHENSIVE:
                return self._comprehensive_evaluation(trace)
        except Exception as e:
            # On error, return None to continue to next depth
            return None
        
        return None
    
    def _quick_evaluation(
        self,
        trace: Union[ExecutionTrace, CollaborativeTrace],
    ) -> EvaluationScore:
        """
        Quick heuristic evaluation (no LLM calls).
        
        Checks:
        - Output exists and has content
        - Minimum length requirements
        - Contains expected keywords
        - Basic format checks
        """
        # Get output and expected
        if isinstance(trace, ExecutionTrace):
            output = trace.final_output or ""
            expected = trace.expected_output or ""
            task = trace.task_description or ""
        else:
            output = trace.final_synthesized_output or ""
            expected = ""  # CollaborativeTrace doesn't have expected_synthesis
            task = trace.scenario_description or ""
        
        score = 5.0  # Start neutral
        reasons = []
        
        # Check 1: Output exists
        if not output or len(output.strip()) < 10:
            return EvaluationScore(
                score=1.0,
                feedback="[heuristic] Output is empty or too short. Quick check: minimal output",
            )
        
        # Check 2: Reasonable length (generous scoring)
        output_len = len(output)
        expected_len = len(expected) if expected else 100
        
        if output_len >= expected_len * 0.5:
            score += 1.0
            reasons.append("adequate length")
        
        if output_len >= expected_len:
            score += 1.0
            reasons.append("meets expected length")
        
        # Bonus for substantial output (>500 chars)
        if output_len > 500:
            score += 0.5
            reasons.append("substantial output")
        
        # Check 3: Contains key terms from task
        task_words = set(task.lower().split())
        important_words = {w for w in task_words if len(w) > 4}
        output_lower = output.lower()
        
        matches = sum(1 for w in important_words if w in output_lower)
        if important_words:
            coverage = matches / len(important_words)
            if coverage > 0.3:
                score += 1.0
                reasons.append(f"keyword coverage: {coverage:.0%}")
            if coverage > 0.5:
                score += 0.5
                reasons.append("good keyword match")
            if coverage > 0.7:
                score += 0.5
                reasons.append("excellent keyword coverage")
        
        # Check 4: Structure indicators (more generous)
        structure_points = 0
        if "\n\n" in output:
            structure_points += 0.3
            reasons.append("has paragraphs")
        if "```" in output:
            structure_points += 0.3
            reasons.append("has code blocks")
        if any(marker in output for marker in ["1.", "2.", "-", "•", "*"]):
            structure_points += 0.3
            reasons.append("has lists")
        if any(marker in output for marker in ["##", "###", "**"]):
            structure_points += 0.3
            reasons.append("has formatting")
        score += min(1.0, structure_points)  # Cap structure bonus at 1.0
        
        # Check 5: Error indicators (negative)
        error_indicators = ["error", "failed", "exception", "cannot", "unable"]
        error_count = sum(1 for e in error_indicators if e in output_lower)
        if error_count > 2:
            score -= 2.0
            reasons.append("multiple error indicators")
        
        # Clamp score
        score = max(0.0, min(10.0, score))
        
        feedback_text = "; ".join(reasons) if reasons else "Basic heuristic evaluation"
        return EvaluationScore(
            score=score,
            feedback=f"[heuristic] {feedback_text}. Quick check: {len(reasons)} positive signals.",
        )
    
    def _standard_evaluation(
        self,
        trace: Union[ExecutionTrace, CollaborativeTrace],
        timeout: float,
    ) -> EvaluationScore:
        """Single LLM evaluation pass."""
        # Format trace info
        if isinstance(trace, ExecutionTrace):
            task = trace.task_description
            expected = trace.expected_output
            actual = trace.final_output
        else:
            task = trace.scenario_description
            expected = "Multi-agent synthesis output"
            actual = trace.final_synthesized_output
        
        messages = [
            {
                "role": "system",
                "content": """You are an evaluation judge. Score the output on a scale of 0-10.

Return JSON: {
    "score": <0-10>,
    "feedback": "brief evaluation feedback",
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"]
}"""
            },
            {
                "role": "user",
                "content": f"""Task: {task}

Expected Output: {expected}

Actual Output: {actual}

Evaluate the actual output against expectations."""
            }
        ]
        
        response = self._call_llm(messages, timeout)
        data = extract_json_from_response(response)
        
        strengths = data.get('strengths', [])
        weaknesses = data.get('weaknesses', [])
        base_feedback = data.get("feedback", "")
        return EvaluationScore(
            score=float(data.get("score", 5.0)),
            feedback=f"[standard] {base_feedback} Strengths: {strengths}. Weaknesses: {weaknesses}",
        )
    
    def _deep_evaluation(
        self,
        trace: Union[ExecutionTrace, CollaborativeTrace],
        timeout: float,
    ) -> EvaluationScore:
        """
        Deep evaluation with 2 contrasting personas (Skeptic + Pragmatist).
        """
        # Use mini-jury with 2 personas
        mini_jury = JuryEvaluator(
            model=self.model,
            base_url=self.base_url,
            personas=self._deep_personas,
            aggregation="mean",
        )
        
        result = mini_jury.evaluate(trace)
        aggregated = result["aggregated_score"]
        
        # Add persona info to feedback
        persona_info = ", ".join(
            f"{p}: {d['score']}" 
            for p, d in result["persona_scores"].items()
            if d["score"] is not None
        )
        
        return EvaluationScore(
            score=aggregated.score,
            feedback=f"[deep] {aggregated.feedback} ({persona_info}). Disagreement: {result['disagreement']['high_disagreement']}",
        )
    
    def _comprehensive_evaluation(
        self,
        trace: Union[ExecutionTrace, CollaborativeTrace],
    ) -> EvaluationScore:
        """Full jury evaluation with all personas."""
        result = self._jury.evaluate(trace)
        aggregated = result["aggregated_score"]
        
        # Enrich with comprehensive info
        persona_summary = ", ".join(
            f"{p}={d['score']:.1f}" 
            for p, d in result["persona_scores"].items()
            if d["score"] is not None
        )
        
        return EvaluationScore(
            score=aggregated.score,
            feedback=f"[comprehensive] {aggregated.feedback} Personas: [{persona_summary}]. Range: {result['disagreement']['range']:.1f}",
            subcategory_scores={
                p: float(d["score"]) 
                for p, d in result["persona_scores"].items()
                if d["score"] is not None
            },
        )
    
    def _estimate_tokens(self, depth: EvaluationDepth) -> int:
        """Estimate tokens used at each depth level."""
        token_estimates = {
            EvaluationDepth.QUICK: 0,       # No LLM
            EvaluationDepth.STANDARD: 500,   # One call
            EvaluationDepth.DEEP: 1000,      # Two personas
            EvaluationDepth.COMPREHENSIVE: 2000,  # Four personas
        }
        return token_estimates.get(depth, 500)
    
    def _estimate_tokens_saved(self, stopped_at: EvaluationDepth) -> int:
        """Estimate tokens saved by early termination."""
        remaining_depths = self._depth_order[self._depth_index(stopped_at) + 1:]
        saved = sum(self._estimate_tokens(d) for d in remaining_depths)
        return saved
    
    def _fallback_score(self) -> EvaluationScore:
        """Return fallback score if all evaluations fail."""
        return EvaluationScore(
            score=5.0,
            feedback="[fallback] Evaluation could not complete, returning neutral score. All evaluation depths failed.",
        )


def create_progressive_evaluator(
    model: str = "qwen2.5-coder:7b",
    base_url: str = "http://localhost:11434",
    max_depth: str = "comprehensive",
    custom_thresholds: Optional[dict[str, tuple[float, float]]] = None,
) -> ProgressiveDeepening:
    """
    Create a progressive deepening evaluator.
    
    This is the recommended evaluator for high-volume evaluation:
    - Saves 60-80% tokens on obvious pass/fail
    - Maintains quality for edge cases
    
    Args:
        model: LLM model to use
        base_url: Ollama API URL
        max_depth: Maximum depth level ("quick", "standard", "deep", "comprehensive")
        custom_thresholds: Optional dict of {depth: (pass_threshold, fail_threshold)}
                          e.g., {"standard": (7.5, 3.5)} to customize thresholds
    
    Returns:
        Configured ProgressiveDeepening evaluator
    """
    # Parse max depth
    depth_map = {
        "quick": EvaluationDepth.QUICK,
        "standard": EvaluationDepth.STANDARD,
        "deep": EvaluationDepth.DEEP,
        "comprehensive": EvaluationDepth.COMPREHENSIVE,
    }
    max_depth_enum = depth_map.get(max_depth.lower(), EvaluationDepth.COMPREHENSIVE)
    
    # Apply custom thresholds if provided
    depth_configs = dict(DEFAULT_DEPTH_CONFIGS)
    if custom_thresholds:
        for depth_name, (pass_thresh, fail_thresh) in custom_thresholds.items():
            if depth_name in depth_map:
                depth_enum = depth_map[depth_name]
                old_config = depth_configs[depth_enum]
                depth_configs[depth_enum] = DepthConfig(
                    depth=depth_enum,
                    min_score_to_pass=pass_thresh,
                    max_score_to_fail=fail_thresh,
                    timeout_seconds=old_config.timeout_seconds,
                    description=old_config.description,
                )
    
    return ProgressiveDeepening(
        model=model,
        base_url=base_url,
        depth_configs=depth_configs,
        max_depth=max_depth_enum,
    )


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_jury_evaluator(
    model: str = "qwen2.5-coder:7b",
    base_url: str = "http://localhost:11434",
    personas: Optional[list[str]] = None,
) -> JuryEvaluator:
    """
    Create a jury evaluator with specified personas.
    
    Args:
        model: LLM model to use
        base_url: Ollama API URL
        personas: List of persona names (skeptic, literalist, optimist, pragmatist)
                 or None for all
    """
    persona_list = None
    if personas:
        persona_list = [JudgePersona(p.lower()) for p in personas]
    
    return JuryEvaluator(
        model=model,
        base_url=base_url,
        personas=persona_list,
    )


def create_calibrated_evaluator(
    judge_models: list[str],
    base_url: str = "http://localhost:11434",
    personas: Optional[list[str]] = None,
) -> CalibratedJuryEvaluator:
    """
    Create a full calibrated jury evaluator.
    
    This is the recommended evaluator for production use:
    - Calibration ensures consistent rubric across judges
    - Multiple personas reduce single-perspective bias
    
    Args:
        judge_models: At least 2 models for calibration
        base_url: Ollama API URL  
        personas: Optional persona filter
    """
    persona_list = None
    if personas:
        persona_list = [JudgePersona(p.lower()) for p in personas]
    
    return CalibratedJuryEvaluator(
        judge_models=judge_models,
        base_url=base_url,
        personas=persona_list,
    )
