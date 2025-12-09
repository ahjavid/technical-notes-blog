"""
Advanced quality scoring for APEE.

Phase 2 implementation with multiple scoring strategies:
- Heuristic scoring (fast, no LLM needed)
- LLM-based scoring (accurate, requires Ollama)
- Composite scoring (combines multiple methods)
"""

import re
from abc import ABC, abstractmethod
from typing import Optional
from pydantic import BaseModel, Field

from apee.models import Task, AgentResult


class QualityScore(BaseModel):
    """Detailed quality score with breakdown."""
    overall: float = Field(ge=0.0, le=1.0)
    relevance: float = Field(ge=0.0, le=1.0, default=0.0)
    completeness: float = Field(ge=0.0, le=1.0, default=0.0)
    structure: float = Field(ge=0.0, le=1.0, default=0.0)
    accuracy: float = Field(ge=0.0, le=1.0, default=0.0)
    clarity: float = Field(ge=0.0, le=1.0, default=0.0)
    reasoning: str = ""


class QualityScorer(ABC):
    """Abstract base class for quality scorers."""
    
    @abstractmethod
    async def score(self, result: AgentResult, task: Task) -> QualityScore:
        """Score the quality of an agent's result."""
        pass
    
    @abstractmethod
    def score_sync(self, result: AgentResult, task: Task) -> QualityScore:
        """Synchronous scoring for simpler use cases."""
        pass


class HeuristicScorer(QualityScorer):
    """
    Fast heuristic-based quality scorer.
    
    Uses text analysis patterns without LLM calls.
    Good for quick evaluations and baseline comparisons.
    """
    
    def __init__(
        self,
        min_words: int = 20,
        max_words: int = 500,
        structure_weight: float = 0.2,
        relevance_weight: float = 0.3,
        completeness_weight: float = 0.3,
        clarity_weight: float = 0.2
    ):
        self.min_words = min_words
        self.max_words = max_words
        self.structure_weight = structure_weight
        self.relevance_weight = relevance_weight
        self.completeness_weight = completeness_weight
        self.clarity_weight = clarity_weight
    
    async def score(self, result: AgentResult, task: Task) -> QualityScore:
        """Async scoring (delegates to sync for heuristics)."""
        return self.score_sync(result, task)
    
    def score_sync(self, result: AgentResult, task: Task) -> QualityScore:
        """Compute heuristic quality score."""
        output = result.output
        
        if not output or not output.strip():
            return QualityScore(
                overall=0.0,
                reasoning="Empty or whitespace-only output"
            )
        
        # Calculate component scores
        relevance = self._score_relevance(output, task)
        completeness = self._score_completeness(output, task)
        structure = self._score_structure(output)
        clarity = self._score_clarity(output)
        
        # Weighted average
        overall = (
            relevance * self.relevance_weight +
            completeness * self.completeness_weight +
            structure * self.structure_weight +
            clarity * self.clarity_weight
        )
        
        return QualityScore(
            overall=overall,
            relevance=relevance,
            completeness=completeness,
            structure=structure,
            accuracy=0.0,  # Cannot assess without ground truth
            clarity=clarity,
            reasoning=self._generate_reasoning(relevance, completeness, structure, clarity)
        )
    
    def _score_relevance(self, output: str, task: Task) -> float:
        """Score based on keyword overlap with task."""
        # Extract meaningful words from task
        task_text = task.description.lower()
        task_words = set(re.findall(r'\b[a-z]{3,}\b', task_text))
        
        # Remove common stopwords
        stopwords = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can',
            'had', 'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been',
            'this', 'that', 'with', 'they', 'from', 'what', 'which', 'their',
            'will', 'would', 'there', 'could', 'should', 'into', 'also'
        }
        task_words -= stopwords
        
        if not task_words:
            return 0.5  # Neutral if no keywords
        
        # Check output for task keywords
        output_lower = output.lower()
        found = sum(1 for word in task_words if word in output_lower)
        
        return min(1.0, found / len(task_words) * 1.5)  # Scale up slightly
    
    def _score_completeness(self, output: str, task: Task) -> float:
        """Score based on response length and coverage."""
        words = output.split()
        word_count = len(words)
        
        # Length scoring
        if word_count < self.min_words:
            length_score = word_count / self.min_words * 0.5
        elif word_count > self.max_words:
            # Penalize overly verbose responses
            length_score = max(0.5, 1.0 - (word_count - self.max_words) / self.max_words * 0.3)
        else:
            # Optimal range
            length_score = 0.7 + 0.3 * (word_count - self.min_words) / (self.max_words - self.min_words)
        
        # Check for incomplete sentences
        incomplete_markers = ['...', '…', 'etc', 'and so on']
        has_incomplete = any(m in output.lower() for m in incomplete_markers)
        
        if has_incomplete:
            length_score *= 0.9
        
        return min(1.0, length_score)
    
    def _score_structure(self, output: str) -> float:
        """Score based on formatting and organization."""
        score = 0.4  # Base score
        
        # Check for bullet points or numbered lists
        if re.search(r'^[\s]*[-*•]\s', output, re.MULTILINE):
            score += 0.2
        if re.search(r'^[\s]*\d+[.)]\s', output, re.MULTILINE):
            score += 0.15
        
        # Check for code blocks
        if '```' in output:
            score += 0.15
        
        # Check for headers/sections
        if re.search(r'^#+\s|^[A-Z][^.!?]*:$', output, re.MULTILINE):
            score += 0.1
        
        # Check for paragraphs (multiple line breaks)
        paragraphs = output.split('\n\n')
        if len(paragraphs) > 1:
            score += 0.1
        
        # Penalize wall of text
        if len(output) > 500 and '\n' not in output:
            score -= 0.2
        
        return min(1.0, max(0.0, score))
    
    def _score_clarity(self, output: str) -> float:
        """Score based on readability indicators."""
        score = 0.5  # Base score
        
        # Check sentence structure
        sentences = re.split(r'[.!?]+', output)
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if valid_sentences:
            # Average sentence length (prefer 10-25 words)
            avg_words = sum(len(s.split()) for s in valid_sentences) / len(valid_sentences)
            if 10 <= avg_words <= 25:
                score += 0.2
            elif avg_words < 10:
                score += 0.1
            else:
                score -= 0.1
        
        # Check for clear transitions
        transitions = ['first', 'second', 'then', 'next', 'finally', 'however', 
                      'therefore', 'because', 'additionally', 'in conclusion']
        transition_count = sum(1 for t in transitions if t in output.lower())
        score += min(0.2, transition_count * 0.05)
        
        # Penalize excessive special characters
        special_ratio = len(re.findall(r'[^a-zA-Z0-9\s.,!?;:\'\"-]', output)) / max(len(output), 1)
        if special_ratio > 0.1:
            score -= 0.15
        
        return min(1.0, max(0.0, score))
    
    def _generate_reasoning(
        self, 
        relevance: float, 
        completeness: float, 
        structure: float, 
        clarity: float
    ) -> str:
        """Generate human-readable scoring explanation."""
        parts = []
        
        if relevance >= 0.8:
            parts.append("highly relevant to task")
        elif relevance >= 0.5:
            parts.append("moderately relevant")
        else:
            parts.append("low task relevance")
        
        if completeness >= 0.8:
            parts.append("comprehensive coverage")
        elif completeness < 0.5:
            parts.append("incomplete response")
        
        if structure >= 0.7:
            parts.append("well-structured")
        elif structure < 0.4:
            parts.append("poor formatting")
        
        if clarity >= 0.7:
            parts.append("clear writing")
        elif clarity < 0.4:
            parts.append("unclear writing")
        
        return "; ".join(parts) if parts else "average quality"


class LLMQualityScorer(QualityScorer):
    """
    LLM-based quality scorer using Ollama.
    
    More accurate but slower than heuristic scoring.
    Uses a judge LLM to evaluate responses.
    """
    
    SCORING_PROMPT = """You are an expert evaluator. Score the following AI response on a scale of 0-10 for each criterion.

TASK: {task}

RESPONSE TO EVALUATE:
{response}

Score each criterion (0-10):
1. RELEVANCE: How well does the response address the task?
2. COMPLETENESS: Does it cover all aspects of the task?
3. STRUCTURE: Is it well-organized and formatted?
4. ACCURACY: Is the information correct and precise?
5. CLARITY: Is it easy to understand?

Respond in this exact format:
RELEVANCE: [score]
COMPLETENESS: [score]
STRUCTURE: [score]
ACCURACY: [score]
CLARITY: [score]
REASONING: [one sentence explanation]"""

    def __init__(
        self,
        model: str = "qwen2.5-coder:3b",
        base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.base_url = base_url
        self._client: Optional['httpx.AsyncClient'] = None
    
    async def _get_client(self):
        """Get or create HTTP client."""
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client
    
    async def score(self, result: AgentResult, task: Task) -> QualityScore:
        """Score using LLM judge."""
        if not result.output.strip():
            return QualityScore(overall=0.0, reasoning="Empty response")
        
        prompt = self.SCORING_PROMPT.format(
            task=task.description,
            response=result.output[:2000]  # Truncate long responses
        )
        
        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "system": "You are an objective evaluator. Be strict but fair.",
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 200}
                }
            )
            response.raise_for_status()
            
            llm_response = response.json().get("response", "")
            return self._parse_scores(llm_response)
            
        except Exception as e:
            # Fallback to heuristic on error
            fallback = HeuristicScorer()
            score = fallback.score_sync(result, task)
            score.reasoning = f"LLM scoring failed ({str(e)[:50]}), used heuristic fallback"
            return score
    
    def score_sync(self, result: AgentResult, task: Task) -> QualityScore:
        """Sync scoring - falls back to heuristic."""
        fallback = HeuristicScorer()
        return fallback.score_sync(result, task)
    
    def _parse_scores(self, llm_response: str) -> QualityScore:
        """Parse LLM response into quality scores."""
        scores = {
            "relevance": 0.5,
            "completeness": 0.5,
            "structure": 0.5,
            "accuracy": 0.5,
            "clarity": 0.5
        }
        reasoning = ""
        
        for line in llm_response.split('\n'):
            line = line.strip().upper()
            for key in scores.keys():
                if line.startswith(key.upper()):
                    try:
                        # Extract number after colon
                        value = re.search(r':\s*(\d+(?:\.\d+)?)', line)
                        if value:
                            scores[key] = float(value.group(1)) / 10.0
                    except ValueError:
                        pass
            
            if line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()
        
        # Calculate overall as weighted average
        overall = (
            scores["relevance"] * 0.25 +
            scores["completeness"] * 0.25 +
            scores["structure"] * 0.15 +
            scores["accuracy"] * 0.20 +
            scores["clarity"] * 0.15
        )
        
        return QualityScore(
            overall=min(1.0, max(0.0, overall)),
            relevance=scores["relevance"],
            completeness=scores["completeness"],
            structure=scores["structure"],
            accuracy=scores["accuracy"],
            clarity=scores["clarity"],
            reasoning=reasoning
        )
    
    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class CodeQualityScorer(QualityScorer):
    """
    Specialized scorer for code-related outputs.
    
    Phase 2: Analyzes code structure, syntax, and best practices.
    """
    
    CODE_PATTERNS = {
        "has_function": r"\bdef\s+\w+\s*\(",
        "has_class": r"\bclass\s+\w+",
        "has_type_hints": r":\s*(int|str|float|bool|list|dict|Optional|Any)\b|-> \w+",
        "has_docstring": r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'',
        "has_error_handling": r"\btry\s*:|except\s+\w*:|raise\s+\w+",
        "has_imports": r"^(import|from)\s+\w+",
        "has_async": r"\basync\s+(def|for|with)",
    }
    
    def __init__(self):
        self.base_scorer = HeuristicScorer()
    
    async def score(self, result: AgentResult, task: Task) -> QualityScore:
        """Score code quality."""
        return self.score_sync(result, task)
    
    def score_sync(self, result: AgentResult, task: Task) -> QualityScore:
        """Analyze code-specific quality metrics."""
        output = result.output
        
        if not output.strip():
            return QualityScore(overall=0.0, reasoning="Empty output")
        
        # Check if output contains code
        has_code_block = "```" in output
        
        # Extract code from markdown blocks if present
        code_blocks = re.findall(r"```(?:\w+)?\n([\s\S]*?)```", output)
        code_content = "\n".join(code_blocks) if code_blocks else output
        
        # Calculate code-specific scores
        code_score = self._analyze_code_patterns(code_content)
        style_score = self._analyze_code_style(code_content)
        
        # Base heuristic scores for text quality
        base_score = self.base_scorer.score_sync(result, task)
        
        # Weighted combination favoring code metrics
        if has_code_block or self._looks_like_code(output):
            overall = (
                code_score * 0.4 +
                style_score * 0.2 +
                base_score.relevance * 0.2 +
                base_score.completeness * 0.2
            )
        else:
            overall = base_score.overall
        
        return QualityScore(
            overall=overall,
            relevance=base_score.relevance,
            completeness=base_score.completeness,
            structure=code_score,
            accuracy=style_score,
            clarity=base_score.clarity,
            reasoning=self._generate_code_reasoning(code_score, style_score, has_code_block)
        )
    
    def _looks_like_code(self, text: str) -> bool:
        """Check if text looks like code even without markdown."""
        code_indicators = [
            r"\bdef\s+\w+",
            r"\bclass\s+\w+",
            r"\bimport\s+\w+",
            r"\breturn\s+",
            r"if\s+.+:",
            r"for\s+\w+\s+in\s+",
        ]
        return any(re.search(p, text) for p in code_indicators)
    
    def _analyze_code_patterns(self, code: str) -> float:
        """Score based on code structure patterns."""
        if not code.strip():
            return 0.0
        
        score = 0.3  # Base score for having code
        
        for pattern_name, pattern in self.CODE_PATTERNS.items():
            if re.search(pattern, code, re.MULTILINE):
                score += 0.1
        
        return min(1.0, score)
    
    def _analyze_code_style(self, code: str) -> float:
        """Analyze code style and best practices."""
        score = 0.5  # Base score
        
        # Good practices
        if re.search(r":\s*\w+\s*(=|,|\))", code):  # Type hints
            score += 0.15
        if re.search(r'""".*"""', code, re.DOTALL):  # Docstrings
            score += 0.15
        if re.search(r"#\s*\w+", code):  # Comments
            score += 0.1
        
        # Bad practices
        lines = code.split("\n")
        long_lines = sum(1 for line in lines if len(line) > 100)
        if long_lines > 3:
            score -= 0.1
        
        # Magic numbers/strings
        magic_count = len(re.findall(r"\b\d{3,}\b", code))
        if magic_count > 2:
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _generate_code_reasoning(
        self, code_score: float, style_score: float, has_block: bool
    ) -> str:
        """Generate reasoning for code quality."""
        parts = []
        
        if has_block:
            parts.append("includes code block")
        
        if code_score >= 0.7:
            parts.append("good code structure")
        elif code_score < 0.4:
            parts.append("minimal code structure")
        
        if style_score >= 0.7:
            parts.append("follows best practices")
        elif style_score < 0.4:
            parts.append("style improvements needed")
        
        return "; ".join(parts) if parts else "code analysis complete"


class ComparativeScorer(QualityScorer):
    """
    Phase 2: Compares multiple responses to determine relative quality.
    
    Useful for model comparison and ranking.
    """
    
    def __init__(self):
        self.base_scorer = HeuristicScorer()
        self.response_cache: list[tuple[AgentResult, QualityScore]] = []
    
    async def score(self, result: AgentResult, task: Task) -> QualityScore:
        """Score relative to other cached responses."""
        return self.score_sync(result, task)
    
    def score_sync(self, result: AgentResult, task: Task) -> QualityScore:
        """Compare against cached responses."""
        base_score = self.base_scorer.score_sync(result, task)
        
        if not self.response_cache:
            # First response - use base score
            self.response_cache.append((result, base_score))
            return base_score
        
        # Calculate relative metrics
        all_scores = [s for _, s in self.response_cache]
        all_scores.append(base_score)
        
        avg_overall = sum(s.overall for s in all_scores) / len(all_scores)
        
        # Percentile ranking
        better_than = sum(1 for s in all_scores if base_score.overall > s.overall)
        percentile = better_than / len(all_scores)
        
        # Adjusted score based on relative performance
        relative_bonus = (base_score.overall - avg_overall) * 0.2
        adjusted_overall = min(1.0, max(0.0, base_score.overall + relative_bonus))
        
        self.response_cache.append((result, base_score))
        
        return QualityScore(
            overall=adjusted_overall,
            relevance=base_score.relevance,
            completeness=base_score.completeness,
            structure=base_score.structure,
            accuracy=percentile,  # Use accuracy field for percentile
            clarity=base_score.clarity,
            reasoning=f"Rank: {percentile:.0%} percentile, {len(self.response_cache)} responses"
        )
    
    def clear_cache(self):
        """Clear response cache for new comparison."""
        self.response_cache.clear()


class MultiJudgeScorer(QualityScorer):
    """
    Phase 2: Uses multiple LLM judges for more robust scoring.
    
    Aggregates scores from different models to reduce bias.
    """
    
    def __init__(
        self,
        judge_models: Optional[list[str]] = None,
        base_url: str = "http://localhost:11434"
    ):
        self.judge_models = judge_models or ["qwen3:4b", "gemma3:4b"]
        self.base_url = base_url
        self.judges: list[LLMQualityScorer] = [
            LLMQualityScorer(model=m, base_url=base_url) for m in self.judge_models
        ]
    
    async def score(self, result: AgentResult, task: Task) -> QualityScore:
        """Get consensus score from multiple judges."""
        scores = []
        
        for judge in self.judges:
            try:
                score = await judge.score(result, task)
                scores.append(score)
            except Exception:
                continue  # Skip failed judges
        
        if not scores:
            # Fallback to heuristic
            return HeuristicScorer().score_sync(result, task)
        
        # Aggregate scores (mean)
        n = len(scores)
        return QualityScore(
            overall=sum(s.overall for s in scores) / n,
            relevance=sum(s.relevance for s in scores) / n,
            completeness=sum(s.completeness for s in scores) / n,
            structure=sum(s.structure for s in scores) / n,
            accuracy=sum(s.accuracy for s in scores) / n,
            clarity=sum(s.clarity for s in scores) / n,
            reasoning=f"Multi-judge consensus ({n} judges)"
        )
    
    def score_sync(self, result: AgentResult, task: Task) -> QualityScore:
        """Sync fallback to heuristic."""
        return HeuristicScorer().score_sync(result, task)
    
    async def close(self):
        """Clean up all judges."""
        for judge in self.judges:
            await judge.close()


class CompositeScorer(QualityScorer):
    """
    Combines multiple scorers with configurable weights.
    
    Useful for balancing speed (heuristic) with accuracy (LLM).
    """
    
    def __init__(
        self,
        heuristic_weight: float = 0.4,
        llm_weight: float = 0.6,
        use_llm: bool = True,
        llm_model: str = "qwen2.5-coder:3b"
    ):
        self.heuristic_weight = heuristic_weight
        self.llm_weight = llm_weight if use_llm else 0.0
        self.use_llm = use_llm
        
        self.heuristic_scorer = HeuristicScorer()
        self.llm_scorer = LLMQualityScorer(model=llm_model) if use_llm else None
        
        # Normalize weights
        total = self.heuristic_weight + self.llm_weight
        self.heuristic_weight /= total
        self.llm_weight /= total
    
    async def score(self, result: AgentResult, task: Task) -> QualityScore:
        """Compute composite score from multiple scorers."""
        heuristic_score = self.heuristic_scorer.score_sync(result, task)
        
        if self.use_llm and self.llm_scorer:
            llm_score = await self.llm_scorer.score(result, task)
            
            # Weighted combination
            return QualityScore(
                overall=(
                    heuristic_score.overall * self.heuristic_weight +
                    llm_score.overall * self.llm_weight
                ),
                relevance=(
                    heuristic_score.relevance * self.heuristic_weight +
                    llm_score.relevance * self.llm_weight
                ),
                completeness=(
                    heuristic_score.completeness * self.heuristic_weight +
                    llm_score.completeness * self.llm_weight
                ),
                structure=(
                    heuristic_score.structure * self.heuristic_weight +
                    llm_score.structure * self.llm_weight
                ),
                accuracy=llm_score.accuracy,  # Only LLM can assess accuracy
                clarity=(
                    heuristic_score.clarity * self.heuristic_weight +
                    llm_score.clarity * self.llm_weight
                ),
                reasoning=f"Heuristic: {heuristic_score.reasoning} | LLM: {llm_score.reasoning}"
            )
        
        return heuristic_score
    
    def score_sync(self, result: AgentResult, task: Task) -> QualityScore:
        """Sync scoring uses only heuristic."""
        return self.heuristic_scorer.score_sync(result, task)
    
    async def close(self):
        """Clean up resources."""
        if self.llm_scorer:
            await self.llm_scorer.close()


class AdaptiveScorer(QualityScorer):
    """
    Phase 2: Automatically selects the best scoring strategy based on task type.
    """
    
    def __init__(self, use_llm: bool = True):
        self.heuristic = HeuristicScorer()
        self.code_scorer = CodeQualityScorer()
        self.llm_scorer = LLMQualityScorer() if use_llm else None
        self.use_llm = use_llm
    
    async def score(self, result: AgentResult, task: Task) -> QualityScore:
        """Adaptively select and apply scorer."""
        task_type = self._detect_task_type(task)
        
        if task_type == "code":
            base_score = self.code_scorer.score_sync(result, task)
        else:
            base_score = self.heuristic.score_sync(result, task)
        
        # Optionally enhance with LLM for complex tasks
        if self.use_llm and self.llm_scorer and task.complexity > 0.6:
            try:
                llm_score = await self.llm_scorer.score(result, task)
                # Blend scores
                return QualityScore(
                    overall=(base_score.overall * 0.4 + llm_score.overall * 0.6),
                    relevance=(base_score.relevance + llm_score.relevance) / 2,
                    completeness=(base_score.completeness + llm_score.completeness) / 2,
                    structure=base_score.structure,
                    accuracy=llm_score.accuracy,
                    clarity=(base_score.clarity + llm_score.clarity) / 2,
                    reasoning=f"Adaptive ({task_type}): {llm_score.reasoning}"
                )
            except Exception:
                pass
        
        return QualityScore(
            overall=base_score.overall,
            relevance=base_score.relevance,
            completeness=base_score.completeness,
            structure=base_score.structure,
            accuracy=base_score.accuracy,
            clarity=base_score.clarity,
            reasoning=f"Adaptive ({task_type}): {base_score.reasoning}"
        )
    
    def score_sync(self, result: AgentResult, task: Task) -> QualityScore:
        """Sync scoring with task-type detection."""
        task_type = self._detect_task_type(task)
        
        if task_type == "code":
            return self.code_scorer.score_sync(result, task)
        return self.heuristic.score_sync(result, task)
    
    def _detect_task_type(self, task: Task) -> str:
        """Detect task type from description."""
        desc_lower = task.description.lower()
        
        code_keywords = ["code", "function", "implement", "write", "program", "script", "class"]
        if any(kw in desc_lower for kw in code_keywords):
            return "code"
        
        analysis_keywords = ["analyze", "compare", "explain", "describe", "review"]
        if any(kw in desc_lower for kw in analysis_keywords):
            return "analysis"
        
        return "general"
    
    async def close(self):
        """Clean up resources."""
        if self.llm_scorer:
            await self.llm_scorer.close()
