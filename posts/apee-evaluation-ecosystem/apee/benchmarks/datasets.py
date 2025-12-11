"""
Benchmark datasets for APEE evaluation.

Provides standardized evaluation scenarios across multiple task categories
following LLM evaluation best practices.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

from apee.models import Task


class TaskCategory(str, Enum):
    """Categories of evaluation tasks."""
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    CODE_EXPLANATION = "code_explanation"
    CODE_DEBUG = "code_debug"
    CODE_REFACTOR = "code_refactor"
    REASONING = "reasoning"
    ANALYSIS = "analysis"
    SUMMARIZATION = "summarization"
    QA_FACTUAL = "qa_factual"
    QA_REASONING = "qa_reasoning"
    CREATIVE = "creative"
    MATH = "math"
    INSTRUCTION_FOLLOWING = "instruction_following"


class Complexity(str, Enum):
    """Task complexity levels."""
    TRIVIAL = "trivial"      # 1-line answers
    EASY = "easy"            # Simple, straightforward
    MEDIUM = "medium"        # Requires some thought
    HARD = "hard"            # Complex, multi-step
    EXPERT = "expert"        # Domain expertise needed


class EvaluationScenario(BaseModel):
    """A single evaluation scenario with ground truth."""
    id: str
    category: TaskCategory
    complexity: Complexity
    prompt: str
    system_prompt: Optional[str] = None
    expected_keywords: list[str] = Field(default_factory=list)
    expected_structure: Optional[str] = None  # e.g., "code_block", "list", "steps"
    reference_answer: Optional[str] = None
    constraints: list[str] = Field(default_factory=list)
    rubric: dict[str, float] = Field(default_factory=dict)  # scoring rubric
    
    def to_task(self) -> Task:
        """Convert to APEE Task."""
        return Task(
            task_id=self.id,
            description=self.prompt,
            complexity=self._complexity_to_float(),
            context={
                "category": self.category.value,
                "expected_keywords": self.expected_keywords,
                "expected_structure": self.expected_structure,
                "constraints": self.constraints,
                "rubric": self.rubric,
            }
        )
    
    def _complexity_to_float(self) -> float:
        """Convert complexity enum to float."""
        mapping = {
            Complexity.TRIVIAL: 0.1,
            Complexity.EASY: 0.3,
            Complexity.MEDIUM: 0.5,
            Complexity.HARD: 0.7,
            Complexity.EXPERT: 0.9,
        }
        return mapping[self.complexity]


class BenchmarkDataset:
    """
    Comprehensive benchmark dataset for model evaluation.
    
    Based on best practices from:
    - lm-evaluation-harness
    - DeepEval
    - HumanEval
    - MMLU
    """
    
    def __init__(self):
        self._scenarios: list[EvaluationScenario] = []
        self._load_default_scenarios()
    
    def _load_default_scenarios(self):
        """Load the default evaluation scenarios."""
        self._scenarios = [
            # ============ CODE GENERATION ============
            EvaluationScenario(
                id="cg_001_fibonacci",
                category=TaskCategory.CODE_GENERATION,
                complexity=Complexity.EASY,
                prompt="Write a Python function that returns the nth Fibonacci number using recursion with memoization.",
                expected_keywords=["def", "fibonacci", "memo", "return", "if"],
                expected_structure="code_block",
                # Note: Using None default avoids mutable default argument anti-pattern
                reference_answer="""def fibonacci(n, memo=None):
    if memo is None:
        memo = {}
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci(n-1, memo) + fibonacci(n-2, memo)
    return memo[n]""",
                constraints=["must use memoization", "must be recursive"],
                rubric={"correctness": 0.4, "memoization": 0.3, "style": 0.15, "docstring": 0.15},
            ),
            EvaluationScenario(
                id="cg_002_async_fetch",
                category=TaskCategory.CODE_GENERATION,
                complexity=Complexity.MEDIUM,
                prompt="Write a Python async function that fetches multiple URLs concurrently using aiohttp and returns a dict mapping URL to response status code.",
                expected_keywords=["async", "await", "aiohttp", "gather", "return"],
                expected_structure="code_block",
                constraints=["must use asyncio.gather", "must handle errors gracefully"],
                rubric={"correctness": 0.35, "concurrency": 0.25, "error_handling": 0.25, "typing": 0.15},
            ),
            EvaluationScenario(
                id="cg_003_binary_search",
                category=TaskCategory.CODE_GENERATION,
                complexity=Complexity.EASY,
                prompt="Implement binary search in Python. The function should return the index of the target element or -1 if not found.",
                expected_keywords=["def", "binary", "search", "mid", "return", "while"],
                expected_structure="code_block",
                # Note: mid = left + (right - left) // 2 is overflow-safe (matters in other languages)
                reference_answer="""def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2  # overflow-safe
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1""",
                rubric={"correctness": 0.5, "efficiency": 0.3, "edge_cases": 0.2},
            ),
            EvaluationScenario(
                id="cg_004_lru_cache",
                category=TaskCategory.CODE_GENERATION,
                complexity=Complexity.HARD,
                prompt="Implement an LRU Cache class in Python with O(1) get and put operations. Include a capacity limit.",
                expected_keywords=["class", "LRUCache", "get", "put", "OrderedDict"],
                expected_structure="code_block",
                constraints=["O(1) time complexity for get/put", "must respect capacity"],
                rubric={"correctness": 0.35, "time_complexity": 0.3, "space_efficiency": 0.2, "api_design": 0.15},
            ),
            EvaluationScenario(
                id="cg_005_decorator",
                category=TaskCategory.CODE_GENERATION,
                complexity=Complexity.MEDIUM,
                prompt="Write a Python decorator that measures and logs the execution time of a function. It should work with both sync and async functions.",
                expected_keywords=["def", "decorator", "time", "async", "wraps"],
                expected_structure="code_block",
                constraints=["must use functools.wraps", "must handle async functions"],
                rubric={"correctness": 0.4, "async_support": 0.3, "proper_wrapping": 0.2, "logging": 0.1},
            ),
            
            # ============ CODE REVIEW ============
            EvaluationScenario(
                id="cr_001_security",
                category=TaskCategory.CODE_REVIEW,
                complexity=Complexity.MEDIUM,
                prompt="""Review this code for security issues and bugs:
```python
def login(username, password):
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    result = db.execute(query)
    return result.fetchone() is not None
```""",
                expected_keywords=["SQL injection", "parameterized", "hash", "password"],
                expected_structure="list",
                constraints=["must identify SQL injection", "must suggest parameterized queries"],
                rubric={"sql_injection_found": 0.35, "password_hashing": 0.25, "fixes_suggested": 0.25, "clarity": 0.15},
            ),
            EvaluationScenario(
                id="cr_002_performance",
                category=TaskCategory.CODE_REVIEW,
                complexity=Complexity.MEDIUM,
                prompt="""Review this code for performance issues:
```python
def find_duplicates(lst):
    duplicates = []
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if lst[i] == lst[j] and lst[i] not in duplicates:
                duplicates.append(lst[i])
    return duplicates
```""",
                expected_keywords=["O(n²)", "O(n)", "set", "Counter", "performance"],
                expected_structure="list",
                rubric={"complexity_identified": 0.3, "better_solution": 0.35, "explanation": 0.2, "code_provided": 0.15},
            ),
            
            # ============ CODE EXPLANATION ============
            EvaluationScenario(
                id="ce_001_closure",
                category=TaskCategory.CODE_EXPLANATION,
                complexity=Complexity.MEDIUM,
                prompt="""Explain what this code does and why it works:
```python
def make_counter():
    count = 0
    def counter():
        nonlocal count
        count += 1
        return count
    return counter
```""",
                expected_keywords=["closure", "nonlocal", "state", "encapsulated", "factory"],
                expected_structure="paragraph",
                rubric={"closure_explained": 0.35, "nonlocal_explained": 0.25, "use_case": 0.2, "clarity": 0.2},
            ),
            
            # ============ CODE DEBUG ============
            EvaluationScenario(
                id="cd_001_off_by_one",
                category=TaskCategory.CODE_DEBUG,
                complexity=Complexity.EASY,
                prompt="""Find and fix the bug in this code:
```python
def reverse_string(s):
    result = []
    for i in range(len(s), 0, -1):
        result.append(s[i])
    return ''.join(result)
```""",
                expected_keywords=["index", "range", "IndexError", "off-by-one", "-1"],
                expected_structure="code_block",
                # Bug: range(len(s), 0, -1) produces indices len(s)..1, but s[len(s)] is out of bounds
                # Fix: range(len(s)-1, -1, -1) produces indices len(s)-1..0
                reference_answer="""def reverse_string(s):
    result = []
    for i in range(len(s) - 1, -1, -1):
        result.append(s[i])
    return ''.join(result)
# Alternative: return s[::-1]""",
                rubric={"bug_identified": 0.4, "correct_fix": 0.4, "explanation": 0.2},
            ),
            
            # ============ REASONING ============
            EvaluationScenario(
                id="re_001_architecture",
                category=TaskCategory.REASONING,
                complexity=Complexity.MEDIUM,
                prompt="Compare microservices vs monolithic architecture. Provide 3 specific pros and 3 cons for each approach.",
                expected_keywords=["microservices", "monolithic", "scalability", "complexity", "deployment"],
                expected_structure="list",
                constraints=["exactly 3 pros and 3 cons for each", "be specific"],
                rubric={"completeness": 0.3, "accuracy": 0.3, "balance": 0.2, "specificity": 0.2},
            ),
            EvaluationScenario(
                id="re_002_tradeoffs",
                category=TaskCategory.REASONING,
                complexity=Complexity.HARD,
                prompt="Explain the CAP theorem and how it applies to distributed databases. Give a specific example of how a real database (e.g., MongoDB, Cassandra) handles CAP trade-offs.",
                expected_keywords=["consistency", "availability", "partition", "tolerance", "trade-off"],
                expected_structure="paragraph",
                rubric={"cap_explained": 0.3, "trade_offs": 0.25, "real_example": 0.25, "accuracy": 0.2},
            ),
            
            # ============ ANALYSIS ============
            EvaluationScenario(
                id="an_001_complexity",
                category=TaskCategory.ANALYSIS,
                complexity=Complexity.MEDIUM,
                prompt="""Analyze the time and space complexity of this algorithm:
```python
def find_pairs(arr, target):
    seen = set()
    pairs = []
    for num in arr:
        complement = target - num
        if complement in seen:
            pairs.append((complement, num))
        seen.add(num)
    return pairs
```""",
                expected_keywords=["O(n)", "time", "space", "set", "linear", "hash"],
                reference_answer="Time complexity: O(n) - single pass through array. Space complexity: O(n) - set stores up to n elements.",
                rubric={"time_correct": 0.35, "space_correct": 0.35, "explanation": 0.3},
            ),
            
            # ============ SUMMARIZATION ============
            EvaluationScenario(
                id="su_001_technical",
                category=TaskCategory.SUMMARIZATION,
                complexity=Complexity.EASY,
                prompt="""Summarize the key points of this technical document in 3 bullet points:

REST (Representational State Transfer) is an architectural style for designing networked applications. It relies on stateless, client-server communication, typically over HTTP. RESTful systems use standard HTTP methods like GET, POST, PUT, and DELETE to perform CRUD operations. Resources are identified by URIs, and data is typically exchanged in JSON or XML format. REST emphasizes scalability, simplicity, and the uniform interface constraint. Unlike SOAP, REST is not a protocol but a set of guidelines, making it more flexible but potentially less standardized across implementations.""",
                expected_keywords=["HTTP", "stateless", "CRUD", "URI", "JSON"],
                expected_structure="list",
                constraints=["exactly 3 bullet points", "concise"],
                rubric={"accuracy": 0.35, "conciseness": 0.25, "completeness": 0.25, "format": 0.15},
            ),
            
            # ============ QA FACTUAL ============
            EvaluationScenario(
                id="qa_001_python",
                category=TaskCategory.QA_FACTUAL,
                complexity=Complexity.EASY,
                prompt="What is the difference between a list and a tuple in Python?",
                expected_keywords=["mutable", "immutable", "list", "tuple", "[]", "()"],
                reference_answer="Lists are mutable (can be modified), use [] brackets. Tuples are immutable (cannot be modified), use () parentheses.",
                rubric={"mutability": 0.4, "syntax": 0.3, "use_cases": 0.3},
            ),
            EvaluationScenario(
                id="qa_002_git",
                category=TaskCategory.QA_FACTUAL,
                complexity=Complexity.EASY,
                prompt="What is the difference between git merge and git rebase?",
                expected_keywords=["merge", "rebase", "commit", "history", "linear"],
                rubric={"merge_explained": 0.35, "rebase_explained": 0.35, "when_to_use": 0.3},
            ),
            
            # ============ QA REASONING ============
            EvaluationScenario(
                id="qr_001_design",
                category=TaskCategory.QA_REASONING,
                complexity=Complexity.HARD,
                prompt="Design a rate limiter for an API. What data structures would you use and why? Consider both fixed window and sliding window approaches.",
                expected_keywords=["token bucket", "sliding window", "fixed window", "Redis", "counter"],
                expected_structure="paragraph",
                rubric={"algorithms_explained": 0.3, "data_structures": 0.25, "trade_offs": 0.25, "implementation": 0.2},
            ),
            
            # ============ MATH ============
            EvaluationScenario(
                id="ma_001_probability",
                category=TaskCategory.MATH,
                complexity=Complexity.MEDIUM,
                prompt="A bag contains 5 red balls and 3 blue balls. If you draw 2 balls without replacement, what is the probability that both are red?",
                # P(both red) = P(1st red) × P(2nd red | 1st red) = (5/8) × (4/7) = 20/56 = 5/14
                expected_keywords=["5/8", "4/7", "5/14", "without replacement", "conditional"],
                reference_answer="P(both red) = (5/8) × (4/7) = 20/56 = 5/14 ≈ 0.357 or about 35.7%",
                rubric={"correct_answer": 0.5, "methodology": 0.3, "explanation": 0.2},
            ),
            
            # ============ INSTRUCTION FOLLOWING ============
            EvaluationScenario(
                id="if_001_format",
                category=TaskCategory.INSTRUCTION_FOLLOWING,
                complexity=Complexity.EASY,
                prompt="List exactly 5 popular Python web frameworks. Format each as a numbered list item with the framework name followed by a colon and a one-sentence description.",
                expected_keywords=["Django", "Flask", "FastAPI"],
                expected_structure="list",
                constraints=["exactly 5 items", "numbered list", "colon format"],
                rubric={"correct_count": 0.3, "format_followed": 0.3, "accuracy": 0.25, "descriptions": 0.15},
            ),
            EvaluationScenario(
                id="if_002_constraint",
                category=TaskCategory.INSTRUCTION_FOLLOWING,
                complexity=Complexity.MEDIUM,
                prompt="Explain what a Python decorator does in exactly 3 sentences. The first sentence must start with 'A decorator'. The last sentence must give a practical example.",
                constraints=["exactly 3 sentences", "first starts with 'A decorator'", "last has example"],
                rubric={"sentence_count": 0.25, "first_sentence": 0.25, "example_included": 0.25, "accuracy": 0.25},
            ),
        ]
    
    @property
    def scenarios(self) -> list[EvaluationScenario]:
        """Get all scenarios."""
        return self._scenarios.copy()
    
    def get_by_category(self, category: TaskCategory) -> list[EvaluationScenario]:
        """Get scenarios by category."""
        return [s for s in self._scenarios if s.category == category]
    
    def get_by_complexity(self, complexity: Complexity) -> list[EvaluationScenario]:
        """Get scenarios by complexity."""
        return [s for s in self._scenarios if s.complexity == complexity]
    
    def get_by_id(self, scenario_id: str) -> Optional[EvaluationScenario]:
        """Get a specific scenario by ID."""
        for s in self._scenarios:
            if s.id == scenario_id:
                return s
        return None
    
    def get_subset(
        self,
        categories: Optional[list[TaskCategory]] = None,
        complexities: Optional[list[Complexity]] = None,
        limit: Optional[int] = None,
    ) -> list[EvaluationScenario]:
        """Get a filtered subset of scenarios."""
        result = self._scenarios
        
        if categories:
            result = [s for s in result if s.category in categories]
        
        if complexities:
            result = [s for s in result if s.complexity in complexities]
        
        if limit:
            result = result[:limit]
        
        return result
    
    def add_scenario(self, scenario: EvaluationScenario):
        """Add a custom scenario."""
        self._scenarios.append(scenario)
    
    def summary(self) -> dict:
        """Get a summary of the dataset."""
        by_category = {}
        by_complexity = {}
        
        for s in self._scenarios:
            by_category[s.category.value] = by_category.get(s.category.value, 0) + 1
            by_complexity[s.complexity.value] = by_complexity.get(s.complexity.value, 0) + 1
        
        return {
            "total_scenarios": len(self._scenarios),
            "by_category": by_category,
            "by_complexity": by_complexity,
        }
