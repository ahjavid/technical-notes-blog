"""
APEE Collaborative Scenarios.

Phase 3+: Multi-agent evaluation scenarios following the APEE framework.
Tests collaboration, coordination, and emergent behaviors.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

from apee.models import Task, AgentRole


class CollaborationPattern(str, Enum):
    """Types of multi-agent collaboration patterns."""
    PARALLEL = "parallel"           # Agents work independently
    SEQUENTIAL = "sequential"       # Pipeline: output flows to next agent
    DEBATE = "debate"               # Agents argue/critique each other
    CONSENSUS = "consensus"         # Agents must agree on output
    HIERARCHICAL = "hierarchical"   # Leader delegates to workers
    PEER_REVIEW = "peer_review"     # Each agent reviews others' work


class CollaborativeScenario(BaseModel):
    """A multi-agent evaluation scenario."""
    id: str
    name: str
    description: str
    pattern: CollaborationPattern
    
    # Agent configuration
    agent_roles: list[AgentRole]
    min_agents: int = 2
    max_agents: int = 4
    
    # Task definition
    task_description: str
    subtasks: list[str] = Field(default_factory=list)
    
    # Expected outcomes
    expected_interactions: int = Field(default=0, description="Min expected messages")
    expected_handoffs: int = Field(default=0, description="Expected task handoffs")
    success_criteria: list[str] = Field(default_factory=list)
    
    # Metrics focus
    primary_metrics: list[str] = Field(default_factory=list)


class MultiAgentDataset:
    """
    Dataset of multi-agent collaboration scenarios.
    
    Following APEE framework Section 5.1:
    - Scenario 1: Collaborative Code Review
    - Scenario 2: Research Synthesis
    - Scenario 3: Problem Solving Under Constraints
    """
    
    def __init__(self):
        self._scenarios = self._load_scenarios()
    
    def _load_scenarios(self) -> list[CollaborativeScenario]:
        """Load APEE-defined multi-agent scenarios."""
        return [
            # ============ SCENARIO 1: Collaborative Code Review ============
            CollaborativeScenario(
                id="collab_code_review",
                name="Collaborative Code Review",
                description="Multiple agents review code changes and provide recommendations",
                pattern=CollaborationPattern.PEER_REVIEW,
                agent_roles=[AgentRole.ANALYZER, AgentRole.REVIEWER, AgentRole.SYNTHESIZER],
                min_agents=3,
                max_agents=4,
                task_description="""
Review the following Python code for issues and improvements:

```python
def process_data(data):
    result = []
    for i in range(len(data)):
        if data[i] > 0:
            result.append(data[i] * 2)
    return result

class DataProcessor:
    def __init__(self):
        self.cache = {}
    
    def fetch(self, url):
        import requests
        resp = requests.get(url)
        return resp.json()
```

Each agent should:
1. ANALYZER: Identify code smells and anti-patterns
2. CRITIC: Evaluate security and performance issues  
3. SYNTHESIZER: Combine insights into actionable recommendations
""",
                subtasks=[
                    "Identify code quality issues",
                    "Analyze security vulnerabilities",
                    "Suggest performance improvements",
                    "Synthesize final recommendations",
                ],
                expected_interactions=6,
                expected_handoffs=2,
                success_criteria=[
                    "Identifies non-Pythonic loop",
                    "Notes missing error handling",
                    "Suggests list comprehension",
                    "Identifies security issue with requests",
                ],
                primary_metrics=["coordination_efficiency", "synergy_score", "conflict_rate"],
            ),
            
            # ============ SCENARIO 2: Research Synthesis ============
            CollaborativeScenario(
                id="research_synthesis",
                name="Research Synthesis",
                description="Agents collaborate to produce comprehensive research summary",
                pattern=CollaborationPattern.SEQUENTIAL,
                agent_roles=[AgentRole.ANALYZER, AgentRole.EXECUTOR, AgentRole.REVIEWER, AgentRole.SYNTHESIZER],
                min_agents=3,
                max_agents=4,
                task_description="""
Research and synthesize information about "Retrieval-Augmented Generation (RAG)":

Pipeline:
1. ANALYZER (Searcher): Identify key concepts and components
2. EXECUTOR (Analyzer): Explain how RAG works technically
3. CRITIC (Fact-Checker): Verify accuracy and completeness
4. SYNTHESIZER (Writer): Produce cohesive summary

Final output should be 200-300 words covering:
- What RAG is
- Key components
- Benefits and limitations
- Use cases
""",
                subtasks=[
                    "Identify RAG components",
                    "Explain technical architecture",
                    "Verify factual accuracy",
                    "Write cohesive summary",
                ],
                expected_interactions=8,
                expected_handoffs=3,
                success_criteria=[
                    "Mentions retrieval component",
                    "Mentions generation component",
                    "Explains vector embeddings",
                    "Discusses hallucination reduction",
                ],
                primary_metrics=["handoff_accuracy", "information_flow", "synergy_score"],
            ),
            
            # ============ SCENARIO 3: Problem Solving Under Constraints ============
            CollaborativeScenario(
                id="constrained_problem",
                name="Problem Solving Under Constraints",
                description="Agents solve optimization problem with resource constraints",
                pattern=CollaborationPattern.DEBATE,
                agent_roles=[AgentRole.ANALYZER, AgentRole.EXECUTOR, AgentRole.REVIEWER],
                min_agents=3,
                max_agents=3,
                task_description="""
Design an API rate limiter with these constraints:

Requirements:
- Handle 1000 requests/second per user
- Memory limit: 100MB total
- Latency: < 1ms per check
- Must handle distributed deployment

Agents should debate approaches:
1. ANALYZER (Planner): Propose solution architecture
2. EXECUTOR: Evaluate implementation feasibility
3. CRITIC (Monitor): Challenge assumptions and edge cases

Reach consensus on best approach with justification.
""",
                subtasks=[
                    "Propose algorithm options",
                    "Analyze trade-offs",
                    "Challenge assumptions",
                    "Reach consensus",
                ],
                expected_interactions=10,
                expected_handoffs=0,  # Debate has no linear handoffs
                success_criteria=[
                    "Considers token bucket algorithm",
                    "Addresses distributed state",
                    "Discusses Redis or similar",
                    "Mentions sliding window",
                ],
                primary_metrics=["conflict_resolution_rate", "consensus_time", "adaptability"],
            ),
            
            # ============ SCENARIO 4: Emergent Behavior Test ============
            CollaborativeScenario(
                id="emergent_behavior",
                name="Emergent Behavior Detection",
                description="Test for novel agent interaction patterns",
                pattern=CollaborationPattern.PARALLEL,
                agent_roles=[AgentRole.EXECUTOR, AgentRole.EXECUTOR, AgentRole.SYNTHESIZER],
                min_agents=3,
                max_agents=5,
                task_description="""
Solve this ambiguous problem - each agent independently:

"Design a system that is both fast AND reliable AND cheap.
You can only optimize for 2 of these 3 properties."

Each EXECUTOR agent should:
1. Choose which 2 properties to optimize
2. Justify the choice
3. Describe the trade-off

SYNTHESIZER should:
1. Compare the different approaches
2. Identify any novel combinations
3. Note if agents converged or diverged

This tests emergent consensus/divergence patterns.
""",
                subtasks=[
                    "Agent 1: Make optimization choice",
                    "Agent 2: Make optimization choice",
                    "Agent 3: Make optimization choice",
                    "Synthesize patterns",
                ],
                expected_interactions=4,
                expected_handoffs=0,
                success_criteria=[
                    "Identifies trade-off",
                    "Makes justified choice",
                    "Compares approaches",
                    "Notes convergence/divergence",
                ],
                primary_metrics=["emergent_pattern_score", "diversity_index", "consensus_rate"],
            ),
            
            # ============ SCENARIO 5: Scalability Test ============
            CollaborativeScenario(
                id="scalability_test",
                name="Agent Scalability Test",
                description="Test performance scaling with agent count",
                pattern=CollaborationPattern.HIERARCHICAL,
                agent_roles=[AgentRole.ANALYZER, AgentRole.EXECUTOR, AgentRole.EXECUTOR, AgentRole.EXECUTOR],
                min_agents=2,
                max_agents=8,
                task_description="""
Implement a modular calculator with these operations:
- Addition
- Subtraction  
- Multiplication
- Division

ANALYZER (Leader) should:
1. Break down the task into modules
2. Assign modules to EXECUTOR agents
3. Coordinate integration

EXECUTOR agents should:
1. Implement assigned module
2. Report completion to leader
3. Handle integration requests

Test will be run with 2, 4, and 8 agents to measure scaling.
""",
                subtasks=[
                    "Task decomposition",
                    "Module assignment",
                    "Parallel implementation",
                    "Integration coordination",
                ],
                expected_interactions=12,
                expected_handoffs=4,
                success_criteria=[
                    "All operations implemented",
                    "Modules integrate correctly",
                    "Work distributed evenly",
                ],
                primary_metrics=["scalability_factor", "coordination_overhead", "parallel_efficiency"],
            ),
            
            # ============ SCENARIO 6: Conflict Resolution ============
            CollaborativeScenario(
                id="conflict_resolution",
                name="Conflict Resolution Test",
                description="Test how agents handle disagreements",
                pattern=CollaborationPattern.CONSENSUS,
                agent_roles=[AgentRole.ANALYZER, AgentRole.ANALYZER, AgentRole.SYNTHESIZER],
                min_agents=3,
                max_agents=3,
                task_description="""
Two analysts must agree on the best database for this use case:

Requirements:
- 10TB of time-series sensor data
- 100K writes/second
- Complex analytical queries
- 99.9% availability requirement

ANALYZER 1: Advocate for PostgreSQL with TimescaleDB
ANALYZER 2: Advocate for InfluxDB

SYNTHESIZER: 
1. Evaluate both arguments
2. Identify points of agreement
3. Propose compromise or final recommendation

Goal: Reach consensus despite initial disagreement.
""",
                subtasks=[
                    "Advocate position 1",
                    "Advocate position 2",
                    "Identify conflicts",
                    "Resolve to consensus",
                ],
                expected_interactions=8,
                expected_handoffs=2,
                success_criteria=[
                    "Both positions argued",
                    "Conflicts identified",
                    "Compromise reached",
                    "Final recommendation given",
                ],
                primary_metrics=["conflict_resolution_rate", "time_to_consensus", "argument_quality"],
            ),
        ]
    
    @property
    def scenarios(self) -> list[CollaborativeScenario]:
        """Get all scenarios."""
        return self._scenarios.copy()
    
    def get_by_pattern(self, pattern: CollaborationPattern) -> list[CollaborativeScenario]:
        """Get scenarios by collaboration pattern."""
        return [s for s in self._scenarios if s.pattern == pattern]
    
    def get_by_id(self, scenario_id: str) -> Optional[CollaborativeScenario]:
        """Get a specific scenario by ID."""
        for s in self._scenarios:
            if s.id == scenario_id:
                return s
        return None
    
    def summary(self) -> dict:
        """Get dataset summary."""
        by_pattern = {}
        for s in self._scenarios:
            by_pattern[s.pattern.value] = by_pattern.get(s.pattern.value, 0) + 1
        
        return {
            "total_scenarios": len(self._scenarios),
            "by_pattern": by_pattern,
            "patterns": [p.value for p in CollaborationPattern],
        }
