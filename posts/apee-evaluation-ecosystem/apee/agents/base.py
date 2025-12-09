"""Base agent class and role definitions."""

import time
from abc import ABC, abstractmethod
from typing import Optional

from apee.models import AgentRole, AgentMetrics, Task, AgentResult, Message


# System prompts for each agent role
ROLE_SYSTEM_PROMPTS: dict[AgentRole, str] = {
    AgentRole.ANALYZER: """You are an expert analyst agent. Your role is to:
- Break down complex problems into components
- Identify patterns, insights, and key factors
- Provide structured analysis with clear reasoning
Be concise and use bullet points for clarity.""",

    AgentRole.CODER: """You are an expert coding agent. Your role is to:
- Write clean, efficient, well-documented code
- Follow best practices and design patterns
- Consider edge cases and error handling
Keep code minimal and focused on the task.""",

    AgentRole.REVIEWER: """You are an expert code reviewer agent. Your role is to:
- Review code for bugs, security issues, and inefficiencies
- Suggest specific, actionable improvements
- Evaluate code quality and maintainability
Be constructive and prioritize critical issues.""",

    AgentRole.SYNTHESIZER: """You are a synthesis agent. Your role is to:
- Combine insights from multiple sources coherently
- Create clear, comprehensive summaries
- Highlight key takeaways and recommendations
Keep summaries structured and actionable.""",

    AgentRole.PLANNER: """You are a planning agent. Your role is to:
- Break down complex tasks into steps
- Identify dependencies and priorities
- Create actionable execution plans
Focus on feasibility and clear milestones.""",

    AgentRole.EXECUTOR: """You are an execution agent. Your role is to:
- Execute tasks according to plans
- Handle edge cases and errors gracefully
- Report progress and blockers clearly
Focus on completing tasks efficiently.""",

    AgentRole.CUSTOM: """You are an AI assistant agent. 
Follow the instructions provided in the task carefully.""",
}


class Agent(ABC):
    """
    Abstract base class for APEE agents.
    
    All agents must implement the execute() method to process tasks.
    Metrics are automatically tracked for evaluation.
    """
    
    def __init__(
        self, 
        agent_id: str, 
        role: AgentRole,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.agent_id = agent_id
        self.role = role
        self.system_prompt = system_prompt or ROLE_SYSTEM_PROMPTS.get(role, "")
        self.model = model  # Optional model identifier
        self.metrics = AgentMetrics()
        self.message_inbox: list[Message] = []
    
    @abstractmethod
    async def execute(self, task: Task) -> AgentResult:
        """
        Execute a task and return a result.
        
        This method must be implemented by all agent subclasses.
        """
        pass
    
    def receive_message(self, message: Message) -> None:
        """Receive a message from another agent."""
        self.message_inbox.append(message)
        self.metrics.messages_received += 1
    
    def get_recent_messages(self, count: int = 5) -> list[Message]:
        """Get the most recent messages from the inbox."""
        return self.message_inbox[-count:]
    
    def clear_inbox(self) -> None:
        """Clear all messages from the inbox."""
        self.message_inbox.clear()
    
    def __repr__(self) -> str:
        return f"Agent(id={self.agent_id}, role={self.role.value})"


class MockAgent(Agent):
    """
    Mock agent for testing and simulation.
    
    Returns predefined or random responses without LLM calls.
    """
    
    def __init__(
        self,
        agent_id: str,
        role: AgentRole,
        response_delay: float = 0.1,
        quality_range: tuple[float, float] = (0.7, 0.95)
    ):
        super().__init__(agent_id, role)
        self.response_delay = response_delay
        self.quality_range = quality_range
    
    async def execute(self, task: Task) -> AgentResult:
        """Execute task with simulated response."""
        import asyncio
        import random
        
        start_time = time.time()
        
        # Simulate processing delay based on complexity
        delay = self.response_delay * (1 + task.complexity)
        await asyncio.sleep(delay)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Generate mock response
        output = f"[{self.role.value}] Analysis of: {task.description[:100]}"
        
        # Calculate quality score
        base_quality = random.uniform(*self.quality_range)
        complexity_penalty = task.complexity * 0.1
        quality = max(0.0, min(1.0, base_quality - complexity_penalty))
        
        # Record metrics
        self.metrics.record_success(latency_ms, quality, tokens=0)
        
        return AgentResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            agent_role=self.role.value,
            output=output,
            quality_score=quality,
            latency_ms=latency_ms,
            tokens_used=0,
            success=True
        )
