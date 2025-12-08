"""
APEE Demo with Ollama - Adaptive Poly-Agentic Evaluation Ecosystem

Real multi-agent system using local Ollama LLMs.
Each agent has a specialized role and processes tasks using actual LLM inference.

Requirements:
    pip install httpx

Usage:
    # Make sure Ollama is running
    ollama serve
    
    # Run the demo
    python demo_apee_ollama.py
"""

import asyncio
import time
import json
import httpx
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod


# ============================================================================
# Configuration
# ============================================================================

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5-coder:3b"  # Fast model for demo
TIMEOUT = 60.0  # seconds


# ============================================================================
# Core Data Structures
# ============================================================================

class AgentRole(Enum):
    ANALYZER = "analyzer"
    CODER = "coder"
    REVIEWER = "reviewer"
    SYNTHESIZER = "synthesizer"


ROLE_SYSTEM_PROMPTS = {
    AgentRole.ANALYZER: """You are an expert analyst agent. Your role is to:
- Break down complex problems into components
- Identify patterns and insights
- Provide structured analysis
Keep responses concise and focused. Use bullet points.""",

    AgentRole.CODER: """You are an expert coding agent. Your role is to:
- Write clean, efficient code
- Implement solutions to problems
- Follow best practices
Keep code minimal and well-commented.""",

    AgentRole.REVIEWER: """You are an expert code reviewer agent. Your role is to:
- Review code for bugs and issues
- Suggest improvements
- Check for security concerns
Be constructive and specific in feedback.""",

    AgentRole.SYNTHESIZER: """You are a synthesis agent. Your role is to:
- Combine insights from multiple sources
- Create coherent summaries
- Highlight key takeaways
Keep summaries brief but comprehensive."""
}


@dataclass
class Task:
    """Represents a task to be processed by agents."""
    task_id: str
    description: str
    complexity: float = 0.5  # 0.0 to 1.0
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AgentResult:
    """Result produced by an agent."""
    task_id: str
    agent_id: str
    agent_role: str
    output: str
    quality_score: float
    latency_ms: float
    tokens_used: int
    success: bool
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Message:
    """Inter-agent communication message."""
    sender_id: str
    receiver_id: str
    content: str
    message_type: str = "info"  # info, request, response
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AgentMetrics:
    """Metrics collected for an individual agent."""
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    quality_scores: List[float] = field(default_factory=list)
    
    @property
    def completion_rate(self) -> float:
        total = self.tasks_completed + self.tasks_failed
        return self.tasks_completed / total if total > 0 else 0.0
    
    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.tasks_completed if self.tasks_completed > 0 else 0.0
    
    @property
    def avg_quality(self) -> float:
        return sum(self.quality_scores) / len(self.quality_scores) if self.quality_scores else 0.0
    
    @property
    def avg_tokens(self) -> float:
        return self.total_tokens / self.tasks_completed if self.tasks_completed > 0 else 0.0


# ============================================================================
# Ollama Client
# ============================================================================

class OllamaClient:
    """Async client for Ollama API."""
    
    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = DEFAULT_MODEL):
        self.base_url = base_url
        self.model = model
        self.client = httpx.AsyncClient(timeout=TIMEOUT)
    
    async def generate(self, prompt: str, system: str = "") -> Dict[str, Any]:
        """Generate a response from Ollama."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 500,  # Limit tokens for faster responses
            }
        }
        
        response = await self.client.post(
            f"{self.base_url}/api/generate",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def check_health(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


# ============================================================================
# Agent Implementation
# ============================================================================

class OllamaAgent:
    """
    LLM-powered agent using Ollama.
    
    Each agent has a specialized role and processes tasks using actual LLM inference.
    """
    
    def __init__(self, agent_id: str, role: AgentRole, model: str = DEFAULT_MODEL):
        self.agent_id = agent_id
        self.role = role
        self.model = model
        self.client = OllamaClient(model=model)
        self.system_prompt = ROLE_SYSTEM_PROMPTS[role]
        self.metrics = AgentMetrics()
        self.message_inbox: List[Message] = []
    
    async def execute(self, task: Task) -> AgentResult:
        """Execute a task using LLM and return result with metrics."""
        start_time = time.time()
        
        try:
            # Build prompt with context
            prompt = self._build_prompt(task)
            
            # Call Ollama
            response = await self.client.generate(prompt, self.system_prompt)
            
            latency_ms = (time.time() - start_time) * 1000
            output = response.get("response", "")
            tokens = response.get("eval_count", 0) + response.get("prompt_eval_count", 0)
            
            # Calculate quality (based on response characteristics)
            quality = self._evaluate_response_quality(output, task)
            
            # Update metrics
            self.metrics.tasks_completed += 1
            self.metrics.total_latency_ms += latency_ms
            self.metrics.total_tokens += tokens
            self.metrics.quality_scores.append(quality)
            
            return AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                agent_role=self.role.value,
                output=output,
                quality_score=quality,
                latency_ms=latency_ms,
                tokens_used=tokens,
                success=True
            )
            
        except Exception as e:
            self.metrics.tasks_failed += 1
            return AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                agent_role=self.role.value,
                output="",
                quality_score=0.0,
                latency_ms=(time.time() - start_time) * 1000,
                tokens_used=0,
                success=False,
                error=str(e)
            )
    
    def _build_prompt(self, task: Task) -> str:
        """Build prompt from task and context."""
        prompt = f"Task: {task.description}\n"
        
        if task.context:
            prompt += "\nContext:\n"
            for key, value in task.context.items():
                prompt += f"- {key}: {value}\n"
        
        # Include relevant messages from inbox
        if self.message_inbox:
            prompt += "\nRelevant messages from other agents:\n"
            for msg in self.message_inbox[-3:]:  # Last 3 messages
                prompt += f"- [{msg.sender_id}]: {msg.content[:200]}\n"
        
        prompt += "\nProvide your response:"
        return prompt
    
    def _evaluate_response_quality(self, output: str, task: Task) -> float:
        """Evaluate response quality based on heuristics."""
        if not output:
            return 0.0
        
        score = 0.5  # Base score
        
        # Length appropriateness (not too short, not too long)
        word_count = len(output.split())
        if 20 <= word_count <= 300:
            score += 0.2
        elif word_count < 10:
            score -= 0.2
        
        # Structure (has bullet points, code blocks, etc.)
        if any(marker in output for marker in ['- ', '* ', '1.', '```']):
            score += 0.15
        
        # Relevance (contains task keywords)
        task_words = set(task.description.lower().split())
        output_words = set(output.lower().split())
        overlap = len(task_words & output_words) / max(len(task_words), 1)
        score += overlap * 0.15
        
        return min(1.0, max(0.0, score))
    
    def receive_message(self, message: Message):
        """Receive a message from another agent."""
        self.message_inbox.append(message)
    
    async def close(self):
        """Clean up resources."""
        await self.client.close()


# ============================================================================
# Coordinator Implementation
# ============================================================================

class APEECoordinator:
    """
    Coordinates multi-agent interactions.
    
    Manages task distribution, message routing, and result aggregation.
    """
    
    def __init__(self, agents: List[OllamaAgent]):
        self.agents = {a.agent_id: a for a in agents}
        self.message_log: List[Message] = []
        self.results: List[AgentResult] = []
    
    async def run_parallel(self, task: Task) -> List[AgentResult]:
        """Run all agents on the same task in parallel."""
        results = await asyncio.gather(*[
            agent.execute(task) for agent in self.agents.values()
        ])
        self.results.extend(results)
        return list(results)
    
    async def run_pipeline(self, task: Task, agent_order: List[str]) -> List[AgentResult]:
        """Run agents sequentially, passing output to next agent."""
        results = []
        current_context = task.context.copy()
        
        for agent_id in agent_order:
            agent = self.agents.get(agent_id)
            if not agent:
                continue
            
            # Create task with accumulated context
            pipeline_task = Task(
                task_id=f"{task.task_id}_{agent_id}",
                description=task.description,
                context=current_context
            )
            
            result = await agent.execute(pipeline_task)
            results.append(result)
            self.results.append(result)
            
            # Add output to context for next agent
            current_context[f"{agent.role.value}_output"] = result.output[:500]
        
        return results
    
    async def run_debate(self, task: Task, rounds: int = 2) -> List[AgentResult]:
        """
        Run agents in a debate format where they can see each other's responses.
        """
        all_results = []
        
        for round_num in range(rounds):
            round_results = []
            
            for agent in self.agents.values():
                # Add previous round's outputs to context
                debate_context = task.context.copy()
                if all_results:
                    debate_context["previous_responses"] = {
                        r.agent_id: r.output[:300] 
                        for r in all_results[-len(self.agents):]
                    }
                debate_context["debate_round"] = round_num + 1
                
                debate_task = Task(
                    task_id=f"{task.task_id}_round{round_num}_{agent.agent_id}",
                    description=f"Round {round_num + 1}: {task.description}",
                    context=debate_context
                )
                
                result = await agent.execute(debate_task)
                round_results.append(result)
            
            all_results.extend(round_results)
            self.results.extend(round_results)
        
        return all_results
    
    def send_message(self, sender_id: str, receiver_id: str, content: str):
        """Route a message between agents."""
        message = Message(
            sender_id=sender_id,
            receiver_id=receiver_id,
            content=content
        )
        self.message_log.append(message)
        
        receiver = self.agents.get(receiver_id)
        if receiver:
            receiver.receive_message(message)
    
    def broadcast(self, sender_id: str, content: str):
        """Broadcast a message to all other agents."""
        for agent_id in self.agents:
            if agent_id != sender_id:
                self.send_message(sender_id, agent_id, content)


# ============================================================================
# Evaluation Implementation
# ============================================================================

@dataclass
class EvaluationReport:
    """Complete evaluation report for APEE system."""
    
    # Individual metrics
    individual_metrics: Dict[str, Dict[str, Any]]
    
    # Collaborative metrics
    total_tasks: int
    total_messages: int
    avg_latency_ms: float
    total_tokens: int
    overall_success_rate: float
    
    # Quality metrics
    avg_quality: float
    quality_variance: float
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "individual_metrics": self.individual_metrics,
            "collaborative_metrics": {
                "total_tasks": self.total_tasks,
                "total_messages": self.total_messages,
                "avg_latency_ms": self.avg_latency_ms,
                "total_tokens": self.total_tokens,
                "overall_success_rate": self.overall_success_rate,
            },
            "quality_metrics": {
                "avg_quality": self.avg_quality,
                "quality_variance": self.quality_variance,
            }
        }
    
    def print_summary(self):
        """Print formatted evaluation summary."""
        print("\n" + "=" * 70)
        print("              APEE EVALUATION REPORT (Ollama)")
        print("=" * 70)
        
        print("\n📊 INDIVIDUAL AGENT METRICS")
        print("-" * 70)
        for agent_id, metrics in self.individual_metrics.items():
            print(f"\n  🤖 {agent_id} ({metrics['role']})")
            print(f"     • Tasks: {metrics['tasks_completed']} completed, {metrics['tasks_failed']} failed")
            print(f"     • Avg Latency: {metrics['avg_latency_ms']:.0f}ms")
            print(f"     • Avg Quality: {metrics['avg_quality']:.2f}")
            print(f"     • Avg Tokens: {metrics['avg_tokens']:.0f}")
        
        print("\n\n🌐 SYSTEM METRICS")
        print("-" * 70)
        print(f"  • Total Tasks Processed: {self.total_tasks}")
        print(f"  • Total Messages: {self.total_messages}")
        print(f"  • Avg System Latency: {self.avg_latency_ms:.0f}ms")
        print(f"  • Total Tokens Used: {self.total_tokens:,}")
        print(f"  • Success Rate: {self.overall_success_rate:.1%}")
        
        print("\n\n📈 QUALITY METRICS")
        print("-" * 70)
        print(f"  • Average Quality: {self.avg_quality:.2f}")
        print(f"  • Quality Variance: {self.quality_variance:.4f}")
        
        print("\n" + "=" * 70)


class APEEEvaluator:
    """Evaluates the multi-agent system performance."""
    
    def __init__(self, coordinator: APEECoordinator):
        self.coordinator = coordinator
    
    def evaluate(self) -> EvaluationReport:
        """Generate comprehensive evaluation report."""
        
        # Individual metrics
        individual = {}
        for agent_id, agent in self.coordinator.agents.items():
            individual[agent_id] = {
                "role": agent.role.value,
                "tasks_completed": agent.metrics.tasks_completed,
                "tasks_failed": agent.metrics.tasks_failed,
                "avg_latency_ms": agent.metrics.avg_latency_ms,
                "avg_quality": agent.metrics.avg_quality,
                "avg_tokens": agent.metrics.avg_tokens,
                "completion_rate": agent.metrics.completion_rate,
            }
        
        # Aggregate metrics
        all_qualities = [r.quality_score for r in self.coordinator.results if r.success]
        avg_quality = sum(all_qualities) / len(all_qualities) if all_qualities else 0
        quality_var = (
            sum((q - avg_quality) ** 2 for q in all_qualities) / len(all_qualities)
            if all_qualities else 0
        )
        
        successful = sum(1 for r in self.coordinator.results if r.success)
        total = len(self.coordinator.results)
        
        avg_latency = (
            sum(r.latency_ms for r in self.coordinator.results) / total
            if total > 0 else 0
        )
        
        total_tokens = sum(r.tokens_used for r in self.coordinator.results)
        
        return EvaluationReport(
            individual_metrics=individual,
            total_tasks=total,
            total_messages=len(self.coordinator.message_log),
            avg_latency_ms=avg_latency,
            total_tokens=total_tokens,
            overall_success_rate=successful / total if total > 0 else 0,
            avg_quality=avg_quality,
            quality_variance=quality_var,
        )


# ============================================================================
# Demo Scenarios
# ============================================================================

async def demo_code_review_pipeline(coordinator: APEECoordinator):
    """Demo: Code review pipeline with all agents."""
    print("\n" + "=" * 70)
    print("📝 SCENARIO 1: Code Review Pipeline")
    print("=" * 70)
    
    task = Task(
        task_id="code_review_1",
        description="""Review this Python function and suggest improvements:

def calc(x, y, op):
    if op == '+':
        return x + y
    elif op == '-':
        return x - y
    elif op == '*':
        return x * y
    elif op == '/':
        return x / y
    else:
        return None
""",
        context={"language": "Python", "priority": "high"}
    )
    
    agent_order = ["analyzer", "reviewer", "coder", "synthesizer"]
    results = await coordinator.run_pipeline(task, agent_order)
    
    for result in results:
        print(f"\n🤖 {result.agent_id} ({result.agent_role}):")
        print(f"   Latency: {result.latency_ms:.0f}ms | Quality: {result.quality_score:.2f}")
        print(f"   Output: {result.output[:200]}...")


async def demo_parallel_analysis(coordinator: APEECoordinator):
    """Demo: All agents analyze the same problem in parallel."""
    print("\n" + "=" * 70)
    print("⚡ SCENARIO 2: Parallel Analysis")
    print("=" * 70)
    
    task = Task(
        task_id="parallel_1",
        description="What are the key considerations when designing a REST API for a todo application? Provide 3 main points.",
        context={"format": "brief bullet points"}
    )
    
    results = await coordinator.run_parallel(task)
    
    for result in results:
        print(f"\n🤖 {result.agent_id} ({result.agent_role}):")
        print(f"   Latency: {result.latency_ms:.0f}ms | Quality: {result.quality_score:.2f}")
        print(f"   Output:\n{result.output[:300]}...")


async def demo_debate(coordinator: APEECoordinator):
    """Demo: Agents debate a technical decision."""
    print("\n" + "=" * 70)
    print("💬 SCENARIO 3: Agent Debate (2 rounds)")
    print("=" * 70)
    
    task = Task(
        task_id="debate_1",
        description="Should we use microservices or a monolith for a new startup's MVP? Give your perspective in 2-3 sentences.",
        context={"team_size": "3 developers", "timeline": "3 months"}
    )
    
    results = await coordinator.run_debate(task, rounds=2)
    
    round_num = 0
    for i, result in enumerate(results):
        if i % len(coordinator.agents) == 0:
            round_num += 1
            print(f"\n--- Round {round_num} ---")
        print(f"\n🤖 {result.agent_id}:")
        print(f"   {result.output[:250]}...")


# ============================================================================
# Main Demo
# ============================================================================

async def main():
    """Run the APEE Ollama demo."""
    print("\n" + "🚀" * 35)
    print("\n       APEE Demo with Ollama LLM Agents")
    print("\n" + "🚀" * 35)
    
    # Check Ollama health
    print("\n🔍 Checking Ollama connection...")
    client = OllamaClient()
    if not await client.check_health():
        print("❌ Error: Ollama is not running. Start it with: ollama serve")
        await client.close()
        return
    await client.close()
    print(f"✅ Ollama is running, using model: {DEFAULT_MODEL}")
    
    # Create agents
    print("\n📦 Creating agents...")
    agents = [
        OllamaAgent("analyzer", AgentRole.ANALYZER),
        OllamaAgent("coder", AgentRole.CODER),
        OllamaAgent("reviewer", AgentRole.REVIEWER),
        OllamaAgent("synthesizer", AgentRole.SYNTHESIZER),
    ]
    
    for agent in agents:
        print(f"   ✓ {agent.agent_id} ({agent.role.value}) - {agent.model}")
    
    # Create coordinator
    coordinator = APEECoordinator(agents)
    
    try:
        # Run demo scenarios
        await demo_parallel_analysis(coordinator)
        await demo_code_review_pipeline(coordinator)
        await demo_debate(coordinator)
        
        # Generate evaluation report
        print("\n\n📊 Generating Evaluation Report...")
        evaluator = APEEEvaluator(coordinator)
        report = evaluator.evaluate()
        report.print_summary()
        
        # Save report
        report_path = "ollama_demo_report.json"
        with open(report_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\n💾 Report saved to: {report_path}")
        
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        for agent in agents:
            await agent.close()
    
    print("\n✅ APEE Ollama Demo completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
