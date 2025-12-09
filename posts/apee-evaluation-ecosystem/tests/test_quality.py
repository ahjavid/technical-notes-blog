"""Tests for quality scoring (Phase 2)."""

import pytest
from apee.models import Task, AgentResult
from apee.evaluation.quality import HeuristicScorer, QualityScore


@pytest.fixture
def sample_task():
    """Create a sample task for testing."""
    return Task(
        task_id="test_task",
        description="Write a Python function to calculate factorial",
        complexity=0.5,
    )


@pytest.fixture
def successful_result():
    """Create a successful result."""
    return AgentResult(
        agent_id="agent_1",
        task_id="test_task",
        agent_role="executor",
        output="""Here's a Python function to calculate factorial:

```python
def factorial(n: int) -> int:
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```

This function uses recursion to calculate the factorial of n.
First, it checks if n is 0 or 1, returning 1 as the base case.
Otherwise, it multiplies n by factorial(n-1).""",
        latency_ms=150.0,
        success=True,
        quality_score=0.8,
        tokens_used=80,
    )


@pytest.fixture
def failed_result():
    """Create a failed result."""
    return AgentResult(
        agent_id="agent_1",
        task_id="test_task",
        agent_role="executor",
        output="",
        latency_ms=10.0,
        success=False,
        quality_score=0.0,
        error="Timeout",
    )


@pytest.fixture
def minimal_result():
    """Create a minimal/short result."""
    return AgentResult(
        agent_id="agent_1",
        task_id="test_task",
        agent_role="executor",
        output="def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
        latency_ms=50.0,
        success=True,
        quality_score=0.5,
        tokens_used=15,
    )


class TestHeuristicScorer:
    """Test heuristic-based scoring."""
    
    def test_score_successful_result(self, sample_task, successful_result):
        """Test scoring a successful result."""
        scorer = HeuristicScorer()
        score = scorer.score_sync(successful_result, sample_task)
        
        assert isinstance(score, QualityScore)
        assert 0.0 <= score.overall <= 1.0
        assert score.relevance > 0.0
        assert score.completeness > 0.0
        assert score.structure > 0.0
        assert score.clarity > 0.0
    
    def test_score_failed_result(self, sample_task, failed_result):
        """Test scoring a failed result with empty output."""
        scorer = HeuristicScorer()
        score = scorer.score_sync(failed_result, sample_task)
        
        # Empty output should have zero score
        assert score.overall == 0.0
        assert "empty" in score.reasoning.lower()
    
    def test_relevance_scoring(self, sample_task, successful_result):
        """Test that relevance is scored based on task keywords."""
        scorer = HeuristicScorer()
        score = scorer.score_sync(successful_result, sample_task)
        
        # Result contains "factorial", "python", "function" from task
        assert score.relevance > 0.5
    
    def test_structure_scoring_code_block(self, sample_task, successful_result):
        """Test that code blocks improve structure score."""
        scorer = HeuristicScorer()
        score = scorer.score_sync(successful_result, sample_task)
        
        # Result has code block formatting
        assert score.structure > 0.4
    
    def test_minimal_result_lower_completeness(self, sample_task, minimal_result):
        """Test that short responses have lower completeness."""
        scorer = HeuristicScorer()
        
        full_score = scorer.score_sync(
            AgentResult(
                agent_id="a", task_id="t", agent_role="r",
                output="This is a comprehensive response with many words explaining the factorial function in detail with examples and use cases and more explanation " * 5,
                latency_ms=100, success=True, quality_score=0.8, tokens_used=100
            ),
            sample_task
        )
        
        minimal_score = scorer.score_sync(minimal_result, sample_task)
        
        # Longer response should have better completeness (up to a point)
        assert full_score.completeness >= minimal_score.completeness
    
    def test_reasoning_generated(self, sample_task, successful_result):
        """Test that reasoning is generated."""
        scorer = HeuristicScorer()
        score = scorer.score_sync(successful_result, sample_task)
        
        assert score.reasoning is not None
        assert len(score.reasoning) > 0


class TestQualityScore:
    """Test QualityScore model."""
    
    def test_quality_score_creation(self):
        """Test creating a quality score."""
        score = QualityScore(
            overall=0.85,
            relevance=0.9,
            completeness=0.8,
            structure=0.7,
            accuracy=0.0,
            clarity=0.75,
            reasoning="Good response",
        )
        
        assert score.overall == 0.85
        assert score.relevance == 0.9
        assert score.reasoning == "Good response"
    
    def test_quality_score_defaults(self):
        """Test default values."""
        score = QualityScore(overall=0.5)
        
        assert score.overall == 0.5
        assert score.relevance == 0.0
        assert score.reasoning == ""
    
    def test_quality_score_bounds(self):
        """Test that scores are bounded 0-1."""
        # Valid scores
        score = QualityScore(overall=0.0)
        assert score.overall == 0.0
        
        score = QualityScore(overall=1.0)
        assert score.overall == 1.0
        
        # Invalid scores should raise validation error
        with pytest.raises(ValueError):
            QualityScore(overall=1.5)
        
        with pytest.raises(ValueError):
            QualityScore(overall=-0.1)


class TestScorerEdgeCases:
    """Test edge cases for scoring."""
    
    def test_empty_task_description(self):
        """Test scoring with empty task description."""
        scorer = HeuristicScorer()
        task = Task(task_id="t", description="", complexity=0.5)
        result = AgentResult(
            agent_id="a", task_id="t", agent_role="r",
            output="Some output",
            latency_ms=100, success=True, quality_score=0.5, tokens_used=10
        )
        
        score = scorer.score_sync(result, task)
        # Should still produce a valid score
        assert 0.0 <= score.overall <= 1.0
    
    def test_special_characters_in_output(self):
        """Test scoring with special characters."""
        scorer = HeuristicScorer()
        task = Task(task_id="t", description="Test task", complexity=0.5)
        result = AgentResult(
            agent_id="a", task_id="t", agent_role="r",
            output="!!!@@@###$$$%%%^^^&&&***((())) weird output",
            latency_ms=100, success=True, quality_score=0.5, tokens_used=10
        )
        
        score = scorer.score_sync(result, task)
        # Should handle special characters and potentially lower clarity
        assert 0.0 <= score.overall <= 1.0
    
    def test_very_long_output(self):
        """Test scoring with very long output."""
        scorer = HeuristicScorer()
        task = Task(task_id="t", description="Test task", complexity=0.5)
        result = AgentResult(
            agent_id="a", task_id="t", agent_role="r",
            output="word " * 1000,  # Very long output
            latency_ms=100, success=True, quality_score=0.5, tokens_used=1000
        )
        
        score = scorer.score_sync(result, task)
        # Should handle long output and potentially lower completeness due to verbosity
        assert 0.0 <= score.overall <= 1.0
