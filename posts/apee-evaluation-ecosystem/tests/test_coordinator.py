"""Tests for APEE coordinator patterns."""

import pytest
import asyncio
from apee.coordination.coordinator import Coordinator
from apee.agents.base import MockAgent
from apee.models import Task, AgentRole


@pytest.fixture
def mock_agents():
    """Create mock agents for testing."""
    return [
        MockAgent("agent_a", AgentRole.ANALYZER, response_delay=0.01),
        MockAgent("agent_b", AgentRole.CODER, response_delay=0.01),
        MockAgent("agent_c", AgentRole.REVIEWER, response_delay=0.01),
    ]


@pytest.fixture
def coordinator(mock_agents):
    """Create coordinator with mock agents."""
    return Coordinator(mock_agents)


@pytest.fixture
def task():
    """Create a sample task."""
    return Task(
        task_id="test_task",
        description="Test task for coordination patterns",
        complexity=0.5,
    )


class TestCoordinatorInit:
    """Tests for Coordinator initialization."""
    
    def test_coordinator_init_with_agents(self, mock_agents):
        """Test coordinator initializes with agents."""
        coordinator = Coordinator(mock_agents)
        assert len(coordinator.agents) == 3
        assert "agent_a" in coordinator.agents
        assert "agent_b" in coordinator.agents
        assert "agent_c" in coordinator.agents
    
    def test_coordinator_init_with_truncation_limits(self, mock_agents):
        """Test coordinator initializes with custom truncation limits."""
        coordinator = Coordinator(
            mock_agents,
            context_truncation_limit=300,
            output_truncation_limit=600,
        )
        assert coordinator.context_truncation_limit == 300
        assert coordinator.output_truncation_limit == 600
    
    def test_coordinator_default_truncation_limits(self, mock_agents):
        """Test coordinator has default truncation limits."""
        coordinator = Coordinator(mock_agents)
        assert coordinator.context_truncation_limit == 500
        assert coordinator.output_truncation_limit == 1000


class TestRunParallel:
    """Tests for run_parallel pattern."""
    
    @pytest.mark.asyncio
    async def test_parallel_runs_all_agents(self, coordinator, task):
        """Test parallel runs all agents."""
        results = await coordinator.run_parallel(task)
        assert len(results) == 3
    
    @pytest.mark.asyncio
    async def test_parallel_results_from_different_agents(self, coordinator, task):
        """Test parallel returns results from all agents."""
        results = await coordinator.run_parallel(task)
        agent_ids = {r.agent_id for r in results}
        assert agent_ids == {"agent_a", "agent_b", "agent_c"}
    
    @pytest.mark.asyncio
    async def test_parallel_all_success(self, coordinator, task):
        """Test parallel results are all successful."""
        results = await coordinator.run_parallel(task)
        assert all(r.success for r in results)
    
    @pytest.mark.asyncio
    async def test_parallel_logs_execution(self, coordinator, task):
        """Test parallel logs execution history."""
        await coordinator.run_parallel(task)
        assert len(coordinator.execution_history) == 1
        assert coordinator.execution_history[0]["pattern"] == "parallel"


class TestRunPipeline:
    """Tests for run_pipeline pattern."""
    
    @pytest.mark.asyncio
    async def test_pipeline_runs_in_order(self, coordinator, task):
        """Test pipeline runs agents in specified order."""
        results = await coordinator.run_pipeline(
            task, agent_order=["agent_a", "agent_b", "agent_c"]
        )
        assert len(results) == 3
        assert results[0].agent_id == "agent_a"
        assert results[1].agent_id == "agent_b"
        assert results[2].agent_id == "agent_c"
    
    @pytest.mark.asyncio
    async def test_pipeline_skips_missing_agents(self, coordinator, task):
        """Test pipeline skips non-existent agents."""
        results = await coordinator.run_pipeline(
            task, agent_order=["agent_a", "nonexistent", "agent_c"]
        )
        assert len(results) == 2
        assert results[0].agent_id == "agent_a"
        assert results[1].agent_id == "agent_c"
    
    @pytest.mark.asyncio
    async def test_pipeline_logs_execution(self, coordinator, task):
        """Test pipeline logs execution history."""
        await coordinator.run_pipeline(task, agent_order=["agent_a", "agent_b"])
        assert len(coordinator.execution_history) == 1
        assert coordinator.execution_history[0]["pattern"] == "pipeline"


class TestRunDebate:
    """Tests for run_debate pattern."""
    
    @pytest.mark.asyncio
    async def test_debate_runs_multiple_rounds(self, coordinator, task):
        """Test debate runs specified number of rounds."""
        results = await coordinator.run_debate(task, rounds=2)
        # 3 agents × 2 rounds = 6 results
        assert len(results) == 6
    
    @pytest.mark.asyncio
    async def test_debate_with_specific_agents(self, coordinator, task):
        """Test debate with specific agent subset."""
        results = await coordinator.run_debate(
            task, rounds=2, agent_ids=["agent_a", "agent_b"]
        )
        # 2 agents × 2 rounds = 4 results
        assert len(results) == 4
    
    @pytest.mark.asyncio
    async def test_debate_logs_execution(self, coordinator, task):
        """Test debate logs execution history."""
        await coordinator.run_debate(task, rounds=1)
        assert len(coordinator.execution_history) == 1
        assert coordinator.execution_history[0]["pattern"] == "debate"


class TestRunHierarchical:
    """Tests for run_hierarchical pattern."""
    
    @pytest.mark.asyncio
    async def test_hierarchical_runs_three_phases(self, coordinator, task):
        """Test hierarchical runs plan, execute, synthesize."""
        results = await coordinator.run_hierarchical(task, leader_id="agent_a")
        # 1 plan + 2 workers + 1 synthesis = 4 results
        assert len(results) == 4
    
    @pytest.mark.asyncio
    async def test_hierarchical_leader_first_and_last(self, coordinator, task):
        """Test hierarchical has leader first (plan) and last (synthesis)."""
        results = await coordinator.run_hierarchical(task, leader_id="agent_a")
        assert results[0].agent_id == "agent_a"  # Planning
        assert results[-1].agent_id == "agent_a"  # Synthesis
    
    @pytest.mark.asyncio
    async def test_hierarchical_workers_in_middle(self, coordinator, task):
        """Test hierarchical has workers in middle."""
        results = await coordinator.run_hierarchical(task, leader_id="agent_a")
        worker_ids = {results[1].agent_id, results[2].agent_id}
        assert worker_ids == {"agent_b", "agent_c"}
    
    @pytest.mark.asyncio
    async def test_hierarchical_with_specific_workers(self, coordinator, task):
        """Test hierarchical with specific worker list."""
        results = await coordinator.run_hierarchical(
            task, leader_id="agent_a", worker_ids=["agent_b"]
        )
        # 1 plan + 1 worker + 1 synthesis = 3 results
        assert len(results) == 3
    
    @pytest.mark.asyncio
    async def test_hierarchical_raises_for_missing_leader(self, coordinator, task):
        """Test hierarchical raises error for non-existent leader."""
        with pytest.raises(ValueError, match="Leader agent.*not found"):
            await coordinator.run_hierarchical(task, leader_id="nonexistent")
    
    @pytest.mark.asyncio
    async def test_hierarchical_raises_for_no_workers(self, mock_agents, task):
        """Test hierarchical raises error when no workers available."""
        # Create coordinator with only one agent
        coordinator = Coordinator([mock_agents[0]])
        with pytest.raises(ValueError, match="requires at least 1 worker"):
            await coordinator.run_hierarchical(task, leader_id="agent_a")
    
    @pytest.mark.asyncio
    async def test_hierarchical_logs_execution(self, coordinator, task):
        """Test hierarchical logs execution history."""
        await coordinator.run_hierarchical(task, leader_id="agent_a")
        assert len(coordinator.execution_history) == 1
        assert coordinator.execution_history[0]["pattern"] == "hierarchical"


class TestRunConsensus:
    """Tests for run_consensus pattern."""
    
    @pytest.mark.asyncio
    async def test_consensus_runs_up_to_max_rounds(self, coordinator, task):
        """Test consensus runs up to max_rounds."""
        results = await coordinator.run_consensus(task, max_rounds=2)
        # Could be 3 agents × 1-2 rounds
        assert len(results) >= 3
        assert len(results) <= 6
    
    @pytest.mark.asyncio
    async def test_consensus_with_specific_agents(self, coordinator, task):
        """Test consensus with specific agent subset."""
        results = await coordinator.run_consensus(
            task, max_rounds=1, agent_ids=["agent_a", "agent_b"]
        )
        # 2 agents × 1 round = 2 results
        assert len(results) == 2
    
    @pytest.mark.asyncio
    async def test_consensus_requires_two_agents(self, mock_agents, task):
        """Test consensus raises error with fewer than 2 agents."""
        coordinator = Coordinator([mock_agents[0]])
        with pytest.raises(ValueError, match="requires at least 2 agents"):
            await coordinator.run_consensus(task)
    
    @pytest.mark.asyncio
    async def test_consensus_logs_execution(self, coordinator, task):
        """Test consensus logs execution history."""
        await coordinator.run_consensus(task, max_rounds=1)
        assert len(coordinator.execution_history) == 1
        assert "consensus" in coordinator.execution_history[0]["pattern"]


class TestRunPeerReview:
    """Tests for run_peer_review pattern."""
    
    @pytest.mark.asyncio
    async def test_peer_review_runs_three_phases(self, coordinator, task):
        """Test peer review runs initial, review, and revision phases."""
        results = await coordinator.run_peer_review(task)
        # 3 agents × 3 phases = 9 results
        assert len(results) == 9
    
    @pytest.mark.asyncio
    async def test_peer_review_with_specific_agents(self, coordinator, task):
        """Test peer review with specific agent subset."""
        results = await coordinator.run_peer_review(
            task, agent_ids=["agent_a", "agent_b"]
        )
        # 2 agents × 3 phases = 6 results
        assert len(results) == 6
    
    @pytest.mark.asyncio
    async def test_peer_review_requires_two_agents(self, mock_agents, task):
        """Test peer review raises error with fewer than 2 agents."""
        coordinator = Coordinator([mock_agents[0]])
        with pytest.raises(ValueError, match="requires at least 2 agents"):
            await coordinator.run_peer_review(task)
    
    @pytest.mark.asyncio
    async def test_peer_review_logs_execution(self, coordinator, task):
        """Test peer review logs execution history."""
        await coordinator.run_peer_review(task)
        assert len(coordinator.execution_history) == 1
        assert coordinator.execution_history[0]["pattern"] == "peer_review"


class TestConsensusDetection:
    """Tests for consensus detection helper methods."""
    
    def test_analyze_agreement_sentiment_positive(self, coordinator):
        """Test sentiment analysis detects positive agreement."""
        score = coordinator._analyze_agreement_sentiment("i agree with the conclusion")
        assert score > 0.5
    
    def test_analyze_agreement_sentiment_negative(self, coordinator):
        """Test sentiment analysis detects disagreement."""
        score = coordinator._analyze_agreement_sentiment("i disagree with this approach")
        assert score < 0.5
    
    def test_analyze_agreement_sentiment_negated_positive(self, coordinator):
        """Test sentiment analysis handles negated positive (don't agree)."""
        score = coordinator._analyze_agreement_sentiment("i don't agree with this")
        assert score < 0.5
    
    def test_analyze_agreement_sentiment_neutral(self, coordinator):
        """Test sentiment analysis returns 0.5 for neutral text."""
        score = coordinator._analyze_agreement_sentiment("the weather is nice today")
        assert score == 0.5


class TestCoordinatorStatistics:
    """Tests for coordinator statistics."""
    
    @pytest.mark.asyncio
    async def test_get_statistics_after_execution(self, coordinator, task):
        """Test statistics are collected after execution."""
        await coordinator.run_parallel(task)
        stats = coordinator.get_statistics()
        
        assert stats["total_executions"] == 1
        assert stats["total_results"] == 3
        assert stats["success_rate"] == 1.0
        assert "agents" in stats
    
    def test_get_statistics_empty(self, coordinator):
        """Test statistics for empty coordinator."""
        stats = coordinator.get_statistics()
        assert stats["total_tasks"] == 0
    
    @pytest.mark.asyncio
    async def test_clear_history(self, coordinator, task):
        """Test clearing execution history."""
        await coordinator.run_parallel(task)
        assert len(coordinator.results) > 0
        
        coordinator.clear_history()
        
        assert len(coordinator.results) == 0
        assert len(coordinator.execution_history) == 0
        assert len(coordinator.message_log) == 0
