"""
Advanced quality scoring for APEE.

Phase 2 implementation with multiple scoring strategies:
- Heuristic scoring (fast, no LLM needed)
- LLM-based scoring (accurate, requires Ollama)
- Composite scoring (combines multiple methods)

Phase 3 heuristic enhancements:
- ROUGE scoring for reference-based evaluation
- Exact keyword matching from expected_keywords
- Constraint validation from constraints field
- Code execution (pass@k) for code generation tasks
- JSON/structure validation
"""

import re
import json
import subprocess
import tempfile
import textwrap
from abc import ABC, abstractmethod
from collections import Counter
from typing import Optional, Any
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
    
    Phase 3 Enhancements:
    - ROUGE scoring when reference_answer exists
    - Exact keyword matching from expected_keywords
    - Constraint validation from constraints field
    - Code execution (pass@k) for code tasks (optional)
    - JSON/structure validation
    """
    
    def __init__(
        self,
        min_words: int = 20,
        max_words: int = 500,
        structure_weight: float = 0.15,
        relevance_weight: float = 0.25,
        completeness_weight: float = 0.25,
        clarity_weight: float = 0.15,
        accuracy_weight: float = 0.20,  # New: for reference-based scoring
        enable_code_execution: bool = False,  # Safety: disabled by default
        code_execution_timeout: float = 5.0,
    ):
        self.min_words = min_words
        self.max_words = max_words
        self.structure_weight = structure_weight
        self.relevance_weight = relevance_weight
        self.completeness_weight = completeness_weight
        self.clarity_weight = clarity_weight
        self.accuracy_weight = accuracy_weight
        self.enable_code_execution = enable_code_execution
        self.code_execution_timeout = code_execution_timeout
    
    async def score(self, result: AgentResult, task: Task) -> QualityScore:
        """Async scoring (delegates to sync for heuristics)."""
        return self.score_sync(result, task)
    
    def score_sync(self, result: AgentResult, task: Task) -> QualityScore:
        """Compute heuristic quality score with enhanced metrics."""
        output = result.output
        
        if not output or not output.strip():
            return QualityScore(
                overall=0.0,
                reasoning="Empty or whitespace-only output"
            )
        
        # Calculate component scores
        relevance = self._score_relevance(output, task)
        completeness = self._score_completeness(output, task)
        structure = self._score_structure(output, task)  # Enhanced with expected_structure
        clarity = self._score_clarity(output)
        
        # Phase 3: Enhanced accuracy scoring
        accuracy = self._score_accuracy(output, task)
        
        # Phase 3: Keyword exact match bonus
        keyword_score = self._score_expected_keywords(output, task)
        
        # Phase 3: Constraint validation
        constraint_score = self._score_constraints(output, task)
        
        # Blend keyword and constraint scores into relevance
        enhanced_relevance = (
            relevance * 0.5 + 
            keyword_score * 0.3 + 
            constraint_score * 0.2
        )
        
        # Weighted average with accuracy
        if accuracy > 0:
            # Reference-based evaluation available
            overall = (
                enhanced_relevance * self.relevance_weight +
                completeness * self.completeness_weight +
                structure * self.structure_weight +
                clarity * self.clarity_weight +
                accuracy * self.accuracy_weight
            )
        else:
            # No reference, redistribute accuracy weight
            total_weight = (
                self.relevance_weight + self.completeness_weight + 
                self.structure_weight + self.clarity_weight
            )
            overall = (
                enhanced_relevance * (self.relevance_weight / total_weight) +
                completeness * (self.completeness_weight / total_weight) +
                structure * (self.structure_weight / total_weight) +
                clarity * (self.clarity_weight / total_weight)
            )
        
        return QualityScore(
            overall=min(1.0, overall),
            relevance=enhanced_relevance,
            completeness=completeness,
            structure=structure,
            accuracy=accuracy,
            clarity=clarity,
            reasoning=self._generate_reasoning(
                enhanced_relevance, completeness, structure, clarity, accuracy,
                keyword_score, constraint_score
            )
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
    
    def _score_expected_keywords(self, output: str, task: Task) -> float:
        """
        Phase 3: Score based on exact expected keyword matching.
        
        Uses expected_keywords from task context for precise evaluation.
        """
        expected_keywords = task.context.get("expected_keywords", [])
        if not expected_keywords:
            return 0.5  # Neutral if no expected keywords
        
        output_lower = output.lower()
        found = sum(1 for kw in expected_keywords if kw.lower() in output_lower)
        
        return found / len(expected_keywords)
    
    def _score_constraints(self, output: str, task: Task) -> float:
        """
        Phase 3: Score based on constraint satisfaction.
        
        Validates that output meets specified constraints.
        """
        constraints = task.context.get("constraints", [])
        if not constraints:
            return 0.5  # Neutral if no constraints
        
        output_lower = output.lower()
        satisfied = 0
        
        for constraint in constraints:
            constraint_lower = constraint.lower()
            
            # Extract key terms from constraint
            key_terms = re.findall(r'\b[a-z]{3,}\b', constraint_lower)
            key_terms = [t for t in key_terms if t not in {
                'must', 'should', 'use', 'have', 'with', 'the', 'and', 'for'
            }]
            
            if key_terms:
                # Check if majority of key terms are present
                found = sum(1 for term in key_terms if term in output_lower)
                if found >= len(key_terms) * 0.5:
                    satisfied += 1
            else:
                satisfied += 0.5  # Partial credit if can't parse constraint
        
        return satisfied / len(constraints)
    
    def _score_accuracy(self, output: str, task: Task) -> float:
        """
        Phase 3: Score accuracy using reference answer when available.
        
        Implements:
        - ROUGE-L scoring for text overlap
        - Code execution (pass@k) for code tasks (if enabled)
        - JSON validation for structured outputs
        """
        reference = task.context.get("reference_answer")
        if not reference:
            return 0.0  # No reference available
        
        # Detect task type
        is_code_task = self._is_code_task(task)
        
        if is_code_task and self.enable_code_execution:
            # Try code execution first
            exec_score = self._execute_code(output, task)
            if exec_score is not None:
                return exec_score
        
        # Check for JSON output
        json_score = self._validate_json(output, reference)
        if json_score is not None:
            return json_score
        
        # Default to ROUGE-L scoring
        return self._rouge_l_score(output, reference)
    
    def _rouge_l_score(self, candidate: str, reference: str) -> float:
        """
        Compute ROUGE-L (Longest Common Subsequence) score.
        
        Measures the longest matching sequence of words.
        """
        # Tokenize
        cand_tokens = self._tokenize(candidate)
        ref_tokens = self._tokenize(reference)
        
        if not cand_tokens or not ref_tokens:
            return 0.0
        
        # Compute LCS length
        lcs_length = self._lcs_length(cand_tokens, ref_tokens)
        
        # Compute precision and recall
        precision = lcs_length / len(cand_tokens) if cand_tokens else 0
        recall = lcs_length / len(ref_tokens) if ref_tokens else 0
        
        # F1 score
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text for ROUGE scoring."""
        # Remove code block markers for fair comparison
        text = re.sub(r'```[a-z]*\n?', '', text)
        # Lowercase and extract words/tokens
        return re.findall(r'\b[a-z0-9_]+\b', text.lower())
    
    def _lcs_length(self, x: list[str], y: list[str]) -> int:
        """Compute length of Longest Common Subsequence."""
        m, n = len(x), len(y)
        
        # Optimize for memory: only keep two rows
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    curr[j] = prev[j-1] + 1
                else:
                    curr[j] = max(prev[j], curr[j-1])
            prev, curr = curr, prev
        
        return prev[n]
    
    def _rouge_1_score(self, candidate: str, reference: str) -> float:
        """
        Compute ROUGE-1 (unigram overlap) F1 score.
        """
        cand_tokens = self._tokenize(candidate)
        ref_tokens = self._tokenize(reference)
        
        if not cand_tokens or not ref_tokens:
            return 0.0
        
        cand_counter = Counter(cand_tokens)
        ref_counter = Counter(ref_tokens)
        
        # Count overlapping tokens
        overlap = sum((cand_counter & ref_counter).values())
        
        precision = overlap / len(cand_tokens) if cand_tokens else 0
        recall = overlap / len(ref_tokens) if ref_tokens else 0
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def _is_code_task(self, task: Task) -> bool:
        """Detect if task is a code generation task."""
        category = task.context.get("category", "")
        if category == "code_generation":
            return True
        
        code_keywords = ["implement", "write a function", "code", "def ", "class "]
        return any(kw in task.description.lower() for kw in code_keywords)
    
    def _execute_code(self, output: str, task: Task) -> Optional[float]:
        """
        Phase 3: Execute code and validate against test cases (pass@k).
        
        Safety: Runs in isolated subprocess with timeout.
        Returns None if execution is not applicable.
        """
        if not self.enable_code_execution:
            return None
        
        # Extract code block from output
        code = self._extract_code(output)
        if not code:
            return None
        
        # Get test cases from task context
        test_cases = task.context.get("test_cases", [])
        reference = task.context.get("reference_answer", "")
        
        # Build test script
        test_script = self._build_test_script(code, test_cases, reference)
        if not test_script:
            return None
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=True) as f:
                f.write(test_script)
                f.flush()
                
                result = subprocess.run(
                    ['python', f.name],
                    capture_output=True,
                    text=True,
                    timeout=self.code_execution_timeout
                )
                
                if result.returncode == 0:
                    # Parse test results from output
                    return self._parse_test_results(result.stdout)
                else:
                    return 0.0  # Execution failed
                    
        except subprocess.TimeoutExpired:
            return 0.0
        except Exception:
            return None  # Fall back to text comparison
    
    def _extract_code(self, output: str) -> Optional[str]:
        """Extract code block from output."""
        # Try to find fenced code block
        match = re.search(r'```(?:python)?\n?(.*?)```', output, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Try to find indented code block
        lines = output.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if line.startswith('def ') or line.startswith('class '):
                in_code = True
            if in_code:
                code_lines.append(line)
                if line and not line[0].isspace() and not line.startswith('def ') and not line.startswith('class '):
                    if code_lines:
                        break
        
        if code_lines:
            return '\n'.join(code_lines)
        
        return None
    
    def _build_test_script(self, code: str, test_cases: list, reference: str) -> Optional[str]:
        """Build a test script for code execution."""
        if not test_cases and not reference:
            return None
        
        script_parts = [
            "import sys",
            "passed = 0",
            "total = 0",
            "",
            "# Code under test",
            code,
            "",
            "# Tests",
        ]
        
        for i, test in enumerate(test_cases):
            if isinstance(test, dict):
                test_code = f"""
total += 1
try:
    result = {test.get('call', 'None')}
    expected = {test.get('expected', 'None')}
    if result == expected:
        passed += 1
except Exception:
    pass
"""
                script_parts.append(test_code)
        
        script_parts.append("""
if total > 0:
    print(f"PASS_RATE:{passed}/{total}")
else:
    print("NO_TESTS")
""")
        
        return '\n'.join(script_parts)
    
    def _parse_test_results(self, stdout: str) -> float:
        """Parse test results from execution output."""
        match = re.search(r'PASS_RATE:(\d+)/(\d+)', stdout)
        if match:
            passed = int(match.group(1))
            total = int(match.group(2))
            return passed / total if total > 0 else 0.0
        return 0.5  # Unknown result
    
    def _validate_json(self, output: str, reference: str) -> Optional[float]:
        """
        Validate JSON structure and content.
        
        Returns None if neither output nor reference is valid JSON.
        """
        try:
            out_json = self._extract_json(output)
            ref_json = self._extract_json(reference)
            
            if out_json is None or ref_json is None:
                return None
            
            return self._compare_json(out_json, ref_json)
            
        except Exception:
            return None
    
    def _extract_json(self, text: str) -> Optional[Any]:
        """Extract and parse JSON from text."""
        # Try to find JSON block
        match = re.search(r'```json\n?(.*?)```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find raw JSON
        for pattern in [r'\{.*\}', r'\[.*\]']:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def _compare_json(self, output: Any, reference: Any) -> float:
        """Compare JSON structures recursively."""
        if type(output) != type(reference):
            return 0.0
        
        if isinstance(reference, dict):
            if not reference:
                return 1.0 if not output else 0.0
            
            scores = []
            for key in reference:
                if key in output:
                    scores.append(self._compare_json(output[key], reference[key]))
                else:
                    scores.append(0.0)
            
            # Penalize extra keys slightly
            extra_keys = len(set(output.keys()) - set(reference.keys()))
            penalty = extra_keys * 0.05
            
            return max(0.0, sum(scores) / len(scores) - penalty) if scores else 0.0
            
        elif isinstance(reference, list):
            if not reference:
                return 1.0 if not output else 0.0
            
            if len(output) != len(reference):
                return 0.5  # Partial credit for wrong length
            
            scores = [self._compare_json(o, r) for o, r in zip(output, reference)]
            return sum(scores) / len(scores) if scores else 0.0
        
        else:
            # Primitive comparison
            return 1.0 if output == reference else 0.0
    
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
    
    def _score_structure(self, output: str, task: Optional[Task] = None) -> float:
        """
        Score based on formatting and organization.
        
        Phase 3: Enhanced with expected_structure validation.
        """
        score = 0.4  # Base score
        
        # Check for bullet points or numbered lists
        has_bullets = bool(re.search(r'^[\s]*[-*•]\s', output, re.MULTILINE))
        has_numbered = bool(re.search(r'^[\s]*\d+[.)]\s', output, re.MULTILINE))
        has_code = '```' in output or bool(re.search(r'^    [a-z]', output, re.MULTILINE))
        has_headers = bool(re.search(r'^#+\s|^[A-Z][^.!?]*:$', output, re.MULTILINE))
        
        if has_bullets:
            score += 0.2
        if has_numbered:
            score += 0.15
        if has_code:
            score += 0.15
        if has_headers:
            score += 0.1
        
        # Check for paragraphs (multiple line breaks)
        paragraphs = output.split('\n\n')
        if len(paragraphs) > 1:
            score += 0.1
        
        # Penalize wall of text
        if len(output) > 500 and '\n' not in output:
            score -= 0.2
        
        # Phase 3: Validate expected structure
        if task:
            expected_structure = task.context.get("expected_structure")
            if expected_structure:
                structure_score = self._validate_expected_structure(
                    output, expected_structure,
                    has_code=has_code,
                    has_bullets=has_bullets,
                    has_numbered=has_numbered,
                    has_headers=has_headers
                )
                # Blend with base structure score
                score = score * 0.5 + structure_score * 0.5
        
        return min(1.0, max(0.0, score))
    
    def _validate_expected_structure(
        self, 
        output: str, 
        expected: str,
        has_code: bool = False,
        has_bullets: bool = False,
        has_numbered: bool = False,
        has_headers: bool = False
    ) -> float:
        """
        Phase 3: Validate output matches expected structure type.
        """
        expected_lower = expected.lower()
        
        structure_checks = {
            "code_block": has_code,
            "code": has_code,
            "list": has_bullets or has_numbered,
            "bullet": has_bullets,
            "numbered": has_numbered,
            "steps": has_numbered,
            "headers": has_headers,
            "sections": has_headers,
        }
        
        for struct_type, has_struct in structure_checks.items():
            if struct_type in expected_lower:
                return 1.0 if has_struct else 0.3
        
        return 0.5  # Unknown structure type
    
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
        clarity: float,
        accuracy: float = 0.0,
        keyword_score: float = 0.0,
        constraint_score: float = 0.0
    ) -> str:
        """
        Generate human-readable scoring explanation.
        
        Phase 3: Enhanced with accuracy, keyword, and constraint feedback.
        """
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
        
        # Phase 3: Enhanced feedback
        if accuracy > 0:
            if accuracy >= 0.8:
                parts.append("high accuracy (ROUGE)")
            elif accuracy >= 0.5:
                parts.append("moderate accuracy")
            else:
                parts.append("low accuracy vs reference")
        
        if keyword_score > 0 and keyword_score != 0.5:
            if keyword_score >= 0.8:
                parts.append(f"keywords: {keyword_score:.0%} matched")
            elif keyword_score < 0.5:
                parts.append(f"keywords: {keyword_score:.0%} missed")
        
        if constraint_score > 0 and constraint_score != 0.5:
            if constraint_score >= 0.8:
                parts.append("constraints satisfied")
            elif constraint_score < 0.5:
                parts.append("constraints violated")
        
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
    Uses large models (12-14B+) as judges for better evaluation quality.
    """
    
    def __init__(
        self,
        judge_models: Optional[list[str]] = None,
        base_url: str = "http://localhost:11434"
    ):
        # Default to large judges from different families
        self.judge_models = judge_models or ["qwen3:14b", "gemma3:12b"]
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
