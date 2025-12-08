"""
APEE Demo - Adaptive Poly-Agentic Evaluation Ecosystem

A demonstration of the APEE framework concepts with a simple multi-agent system.
This demo shows how to:
1. Define agents with specific roles
2. Coordinate agent interactions
3. Collect and compute evaluation metrics
4. Apply adaptive evaluation criteria

Requirements:
    pip install asyncio dataclasses

Usage:
    python demo_apee.py
"""

import asyncio
import time
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
import json


# ============================================================================
# Core Data Structures
# ============================================================================

class AgentRole(Enum):
    ANALYZER = "analyzer"
    EXECUTOR = "executor"
    REVIEWER = "reviewer"
    SYNTHESIZER = "synthesizer"


@dataclass
class Task:
    """Represents a task to be processed by agents."""
    task_id: str
    description: str
    complexity: float  # 0.0 to 1.0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Result:
    """Result produced by an agent."""
    task_id: str
    agent_id: str
    output: str
    quality_score: float  # 0.0 to 1.0
    latency_ms: float
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """Inter-agent communication message."""
    message_id: str
    sender_id: str
    receiver_id: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AgentMetrics:
    """Metrics collected for an individual agent."""
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_latency_ms: float = 0.0
    total_quality: float = 0.0
    messages_sent: int = 0
    messages_received: int = 0
    
    @property
    def completion_rate(self) -> float:
        total = self.tasks_completed + self.tasks_failed
        return self.tasks_completed / total if total > 0 else 0.0
    
    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.tasks_completed if self.tasks_completed > 0 else 0.0
    
    @property
    def avg_quality(self) -> float:
        return self.total_quality / self.tasks_completed if self.tasks_completed > 0 else 0.0


# ============================================================================
# Agent Implementation
# ============================================================================

class APEEAgent:
    """
    Base agent class for APEE framework.
    
    Each agent has a specific role and processes tasks accordingly.
    Metrics are automatically collected during execution.
    """
    
    def __init__(self, agent_id: str, role: AgentRole):
        self.agent_id = agent_id
        self.role = role
        self.metrics = AgentMetrics()
        self._processing_delay = random.uniform(0.1, 0.3)  # Simulated processing time
    
    async def execute(self, task: Task) -> Result:
        """Execute a task and return result with metrics."""
        start_time = time.time()
        
        try:
            # Simulate processing with role-specific behavior
            output = await self._process(task)
            latency_ms = (time.time() - start_time) * 1000
            
            # Simulate quality based on complexity and role suitability
            quality = self._calculate_quality(task)
            
            # Update metrics
            self.metrics.tasks_completed += 1
            self.metrics.total_latency_ms += latency_ms
            self.metrics.total_quality += quality
            
            return Result(
                task_id=task.task_id,
                agent_id=self.agent_id,
                output=output,
                quality_score=quality,
                latency_ms=latency_ms,
                success=True
            )
            
        except Exception as e:
            self.metrics.tasks_failed += 1
            return Result(
                task_id=task.task_id,
                agent_id=self.agent_id,
                output=str(e),
                quality_score=0.0,
                latency_ms=(time.time() - start_time) * 1000,
                success=False
            )
    
    async def _process(self, task: Task) -> str:
        """Simulate task processing."""
        # Add some variance to processing time based on complexity
        delay = self._processing_delay * (1 + task.complexity)
        await asyncio.sleep(delay)
        
        return f"[{self.role.value}] Processed: {task.description[:50]}..."
    
    def _calculate_quality(self, task: Task) -> float:
        """Calculate quality score based on task-role fit."""
        base_quality = random.uniform(0.7, 0.95)
        
        # Role-specific quality adjustments
        role_bonuses = {
            AgentRole.ANALYZER: 0.05 if "analyz" in task.description.lower() else 0,
            AgentRole.EXECUTOR: 0.05 if "execut" in task.description.lower() else 0,
            AgentRole.REVIEWER: 0.05 if "review" in task.description.lower() else 0,
            AgentRole.SYNTHESIZER: 0.05 if "synth" in task.description.lower() else 0,
        }
        
        bonus = role_bonuses.get(self.role, 0)
        complexity_penalty = task.complexity * 0.1
        
        return min(1.0, max(0.0, base_quality + bonus - complexity_penalty))
    
    def send_message(self, receiver_id: str, content: str) -> Message:
        """Send a message to another agent."""
        self.metrics.messages_sent += 1
        return Message(
            message_id=f"msg_{self.agent_id}_{time.time()}",
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            content=content
        )
    
    def receive_message(self, message: Message):
        """Receive a message from another agent."""
        self.metrics.messages_received += 1


# ============================================================================
# Coordinator Implementation
# ============================================================================

class APEECoordinator:
    """
    Coordinates multi-agent interactions and task distribution.
    
    The coordinator manages:
    - Task assignment to appropriate agents
    - Message routing between agents
    - Conflict resolution
    - State management
    """
    
    def __init__(self, agents: List[APEEAgent]):
        self.agents = {a.agent_id: a for a in agents}
        self.message_log: List[Message] = []
        self.task_history: List[Task] = []
        self.result_history: List[Result] = []
    
    async def distribute_task(self, task: Task) -> List[Result]:
        """Distribute a task to all agents and collect results."""
        self.task_history.append(task)
        
        # Run all agents in parallel
        results = await asyncio.gather(*[
            agent.execute(task) for agent in self.agents.values()
        ])
        
        self.result_history.extend(results)
        return list(results)
    
    async def sequential_pipeline(self, task: Task, agent_order: List[str]) -> Result:
        """Process task through agents sequentially."""
        self.task_history.append(task)
        current_output = task.description
        final_result = None
        
        for agent_id in agent_order:
            agent = self.agents.get(agent_id)
            if not agent:
                continue
            
            # Create modified task with previous output
            pipeline_task = Task(
                task_id=f"{task.task_id}_{agent_id}",
                description=current_output,
                complexity=task.complexity,
                metadata={"original_task": task.task_id}
            )
            
            result = await agent.execute(pipeline_task)
            self.result_history.append(result)
            current_output = result.output
            final_result = result
        
        return final_result
    
    def route_message(self, message: Message):
        """Route a message to the target agent."""
        self.message_log.append(message)
        
        receiver = self.agents.get(message.receiver_id)
        if receiver:
            receiver.receive_message(message)
    
    def broadcast_message(self, sender: APEEAgent, content: str):
        """Broadcast a message from one agent to all others."""
        for agent_id, agent in self.agents.items():
            if agent_id != sender.agent_id:
                message = sender.send_message(agent_id, content)
                self.route_message(message)


# ============================================================================
# Evaluation Implementation
# ============================================================================

@dataclass
class IndividualReport:
    """Report for individual agent metrics."""
    agent_id: str
    role: str
    completion_rate: float
    avg_latency_ms: float
    avg_quality: float
    tasks_completed: int
    messages_sent: int
    messages_received: int


@dataclass
class CollaborativeReport:
    """Report for collaborative metrics."""
    total_messages: int
    avg_messages_per_task: float
    communication_efficiency: float
    synergy_score: float


@dataclass
class EcosystemReport:
    """Report for ecosystem-level metrics."""
    total_tasks: int
    overall_success_rate: float
    avg_system_latency_ms: float
    agent_utilization: float


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    individual: List[IndividualReport]
    collaborative: CollaborativeReport
    ecosystem: EcosystemReport
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "individual": [vars(i) for i in self.individual],
            "collaborative": vars(self.collaborative),
            "ecosystem": vars(self.ecosystem)
        }
    
    def print_summary(self):
        """Print a formatted summary of the report."""
        print("\n" + "="*60)
        print("         APEE EVALUATION REPORT")
        print("="*60)
        
        print("\n📊 INDIVIDUAL AGENT METRICS")
        print("-"*60)
        for ir in self.individual:
            print(f"\n  Agent: {ir.agent_id} ({ir.role})")
            print(f"    • Completion Rate: {ir.completion_rate:.1%}")
            print(f"    • Avg Quality:     {ir.avg_quality:.2f}")
            print(f"    • Avg Latency:     {ir.avg_latency_ms:.1f}ms")
            print(f"    • Tasks Completed: {ir.tasks_completed}")
            print(f"    • Messages:        {ir.messages_sent}↑ / {ir.messages_received}↓")
        
        print("\n\n🤝 COLLABORATIVE METRICS")
        print("-"*60)
        print(f"  • Total Messages:          {self.collaborative.total_messages}")
        print(f"  • Avg Messages/Task:       {self.collaborative.avg_messages_per_task:.2f}")
        print(f"  • Communication Efficiency: {self.collaborative.communication_efficiency:.1%}")
        print(f"  • Synergy Score:           {self.collaborative.synergy_score:.2f}x")
        
        print("\n\n🌐 ECOSYSTEM METRICS")
        print("-"*60)
        print(f"  • Total Tasks:         {self.ecosystem.total_tasks}")
        print(f"  • Overall Success Rate: {self.ecosystem.overall_success_rate:.1%}")
        print(f"  • Avg System Latency:  {self.ecosystem.avg_system_latency_ms:.1f}ms")
        print(f"  • Agent Utilization:   {self.ecosystem.agent_utilization:.1%}")
        
        print("\n" + "="*60)


class APEEEvaluator:
    """
    Evaluates poly-agent system performance.
    
    Computes metrics at three levels:
    1. Individual agent performance
    2. Collaborative effectiveness
    3. Ecosystem health
    """
    
    def __init__(self, coordinator: APEECoordinator):
        self.coordinator = coordinator
    
    def evaluate(self) -> EvaluationReport:
        """Generate comprehensive evaluation report."""
        return EvaluationReport(
            individual=self._compute_individual_metrics(),
            collaborative=self._compute_collaborative_metrics(),
            ecosystem=self._compute_ecosystem_metrics()
        )
    
    def _compute_individual_metrics(self) -> List[IndividualReport]:
        """Compute metrics for each agent."""
        reports = []
        
        for agent_id, agent in self.coordinator.agents.items():
            reports.append(IndividualReport(
                agent_id=agent_id,
                role=agent.role.value,
                completion_rate=agent.metrics.completion_rate,
                avg_latency_ms=agent.metrics.avg_latency_ms,
                avg_quality=agent.metrics.avg_quality,
                tasks_completed=agent.metrics.tasks_completed,
                messages_sent=agent.metrics.messages_sent,
                messages_received=agent.metrics.messages_received
            ))
        
        return reports
    
    def _compute_collaborative_metrics(self) -> CollaborativeReport:
        """Compute collaboration metrics."""
        total_messages = len(self.coordinator.message_log)
        total_tasks = len(self.coordinator.task_history)
        
        avg_messages = total_messages / total_tasks if total_tasks > 0 else 0
        
        # Communication efficiency: ratio of meaningful exchanges
        efficiency = min(1.0, 0.8 + random.uniform(-0.1, 0.1))  # Simulated
        
        # Synergy: combined performance vs sum of individual
        individual_sum = sum(
            a.metrics.avg_quality for a in self.coordinator.agents.values()
        )
        num_agents = len(self.coordinator.agents)
        
        if individual_sum > 0 and self.coordinator.result_history:
            combined_quality = sum(r.quality_score for r in self.coordinator.result_history)
            synergy = (combined_quality / len(self.coordinator.result_history)) / (individual_sum / num_agents)
        else:
            synergy = 1.0
        
        return CollaborativeReport(
            total_messages=total_messages,
            avg_messages_per_task=avg_messages,
            communication_efficiency=efficiency,
            synergy_score=synergy
        )
    
    def _compute_ecosystem_metrics(self) -> EcosystemReport:
        """Compute ecosystem-level metrics."""
        total_tasks = len(self.coordinator.task_history)
        
        successful = sum(1 for r in self.coordinator.result_history if r.success)
        total_results = len(self.coordinator.result_history)
        success_rate = successful / total_results if total_results > 0 else 0
        
        avg_latency = (
            sum(r.latency_ms for r in self.coordinator.result_history) / total_results
            if total_results > 0 else 0
        )
        
        # Agent utilization: ratio of active time to total time
        utilization = min(1.0, 0.75 + random.uniform(-0.1, 0.15))  # Simulated
        
        return EcosystemReport(
            total_tasks=total_tasks,
            overall_success_rate=success_rate,
            avg_system_latency_ms=avg_latency,
            agent_utilization=utilization
        )


# ============================================================================
# Demo Execution
# ============================================================================

async def run_demo():
    """Run the APEE demonstration."""
    print("\n🚀 Starting APEE Demo")
    print("="*60)
    
    # Create agents
    print("\n📦 Creating agents...")
    agents = [
        APEEAgent("agent_analyzer", AgentRole.ANALYZER),
        APEEAgent("agent_executor", AgentRole.EXECUTOR),
        APEEAgent("agent_reviewer", AgentRole.REVIEWER),
        APEEAgent("agent_synthesizer", AgentRole.SYNTHESIZER),
    ]
    
    for agent in agents:
        print(f"   ✓ {agent.agent_id} ({agent.role.value})")
    
    # Create coordinator
    print("\n🎯 Initializing coordinator...")
    coordinator = APEECoordinator(agents)
    
    # Create sample tasks
    print("\n📝 Creating sample tasks...")
    tasks = [
        Task("task_1", "Analyze the performance data and identify bottlenecks", 0.3),
        Task("task_2", "Execute the optimization pipeline with new parameters", 0.5),
        Task("task_3", "Review the code changes for security vulnerabilities", 0.4),
        Task("task_4", "Synthesize findings from multiple analysis reports", 0.6),
        Task("task_5", "Complex multi-step analysis requiring all agent types", 0.8),
    ]
    
    for task in tasks:
        print(f"   • {task.task_id}: {task.description[:40]}... (complexity: {task.complexity})")
    
    # Run parallel processing
    print("\n⚡ Running parallel task processing...")
    for task in tasks:
        results = await coordinator.distribute_task(task)
        print(f"   ✓ {task.task_id} completed by {len(results)} agents")
    
    # Simulate inter-agent communication
    print("\n💬 Simulating agent communication...")
    for i, agent in enumerate(agents):
        if i < len(agents) - 1:
            coordinator.broadcast_message(agent, f"Status update from {agent.role.value}")
    print(f"   ✓ {len(coordinator.message_log)} messages exchanged")
    
    # Run sequential pipeline
    print("\n🔄 Running sequential pipeline...")
    pipeline_task = Task("task_pipeline", "End-to-end document processing", 0.7)
    pipeline_order = ["agent_analyzer", "agent_executor", "agent_reviewer", "agent_synthesizer"]
    await coordinator.sequential_pipeline(pipeline_task, pipeline_order)
    print("   ✓ Pipeline completed")
    
    # Generate evaluation report
    print("\n📊 Generating evaluation report...")
    evaluator = APEEEvaluator(coordinator)
    report = evaluator.evaluate()
    
    # Print report
    report.print_summary()
    
    # Save report to JSON
    report_path = "demo_report.json"
    with open(report_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"\n💾 Report saved to: {report_path}")
    
    print("\n✅ APEE Demo completed successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(run_demo())
