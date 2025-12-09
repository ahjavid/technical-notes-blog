"""Tests for APEE models."""

import pytest
from apee.models import Task, AgentResult, AgentRole


class TestTask:
    """Test Task model."""
    
    def test_task_creation(self):
        """Test basic task creation."""
        task = Task(
            task_id="test_1",
            description="Test description",
            complexity=0.5,
        )
        assert task.task_id == "test_1"
        assert task.description == "Test description"
        assert task.complexity == 0.5
    
    def test_task_with_context(self):
        """Test task with context."""
        task = Task(
            task_id="test_2",
            description="Write a function",
            context={"language": "python"},
            complexity=0.7,
        )
        assert task.context == {"language": "python"}
    
    def test_task_complexity_bounds(self):
        """Test that complexity is properly bounded."""
        task = Task(task_id="t", description="d", complexity=0.0)
        assert task.complexity == 0.0
        
        task = Task(task_id="t", description="d", complexity=1.0)
        assert task.complexity == 1.0


class TestAgentResult:
    """Test AgentResult model."""
    
    def test_result_success(self):
        """Test successful result."""
        result = AgentResult(
            agent_id="agent_1",
            task_id="task_1",
            agent_role="executor",
            output="This is a response",
            latency_ms=150.5,
            success=True,
            quality_score=0.85,
            tokens_used=50,
        )
        assert result.success is True
        assert result.latency_ms == 150.5
        assert result.error is None
    
    def test_result_failure(self):
        """Test failed result."""
        result = AgentResult(
            agent_id="agent_1",
            task_id="task_1",
            agent_role="executor",
            output="",
            latency_ms=10.0,
            success=False,
            quality_score=0.0,
            error="Connection timeout",
        )
        assert result.success is False
        assert result.error == "Connection timeout"


class TestAgentRole:
    """Test AgentRole enum."""
    
    def test_all_roles_exist(self):
        """Ensure all expected roles exist."""
        roles = [r.value for r in AgentRole]
        assert "analyzer" in roles
        assert "executor" in roles
        assert "synthesizer" in roles
        assert "planner" in roles
