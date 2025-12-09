"""Multi-agent coordinator for task distribution and orchestration."""

import asyncio
import time
from typing import Optional
from uuid import uuid4

from apee.models import Task, AgentResult, Message
from apee.agents.base import Agent


class Coordinator:
    """
    Orchestrates multi-agent interactions.
    
    Supports multiple execution patterns:
    - Parallel: All agents process the same task simultaneously
    - Pipeline: Sequential processing where each agent builds on previous output
    - Debate: Multi-round discussion where agents can see each other's responses
    - Selective: Route tasks to specific agents based on criteria
    """
    
    def __init__(self, agents: list[Agent]):
        self.agents: dict[str, Agent] = {a.agent_id: a for a in agents}
        self.message_log: list[Message] = []
        self.results: list[AgentResult] = []
        self.execution_history: list[dict] = []
    
    def add_agent(self, agent: Agent) -> None:
        """Add an agent to the coordinator."""
        self.agents[agent.agent_id] = agent
    
    def remove_agent(self, agent_id: str) -> Optional[Agent]:
        """Remove and return an agent from the coordinator."""
        return self.agents.pop(agent_id, None)
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID."""
        return self.agents.get(agent_id)
    
    async def run_parallel(self, task: Task) -> list[AgentResult]:
        """
        Run all agents on the same task in parallel.
        
        Best for: Getting diverse perspectives on the same problem.
        """
        start_time = time.time()
        
        results = await asyncio.gather(*[
            agent.execute(task) for agent in self.agents.values()
        ], return_exceptions=True)
        
        # Convert exceptions to failed results
        processed_results = []
        for i, (agent_id, result) in enumerate(zip(self.agents.keys(), results)):
            if isinstance(result, Exception):
                processed_results.append(AgentResult(
                    task_id=task.task_id,
                    agent_id=agent_id,
                    agent_role=self.agents[agent_id].role.value,
                    output="",
                    quality_score=0.0,
                    latency_ms=0.0,
                    success=False,
                    error=str(result)
                ))
            else:
                processed_results.append(result)
        
        self.results.extend(processed_results)
        self._log_execution("parallel", task, processed_results, time.time() - start_time)
        
        return processed_results
    
    async def run_pipeline(
        self, 
        task: Task, 
        agent_order: list[str],
        pass_context: bool = True
    ) -> list[AgentResult]:
        """
        Run agents sequentially, optionally passing output to next agent.
        
        Best for: Code review workflows, multi-stage analysis.
        
        Args:
            task: The task to process
            agent_order: List of agent IDs in execution order
            pass_context: Whether to pass each agent's output to the next
        """
        start_time = time.time()
        results = []
        accumulated_context = dict(task.context)
        
        for i, agent_id in enumerate(agent_order):
            agent = self.agents.get(agent_id)
            if not agent:
                continue
            
            # Create task with accumulated context
            pipeline_task = Task(
                task_id=f"{task.task_id}_step{i}_{agent_id}",
                description=task.description,
                complexity=task.complexity,
                context=accumulated_context.copy()
            )
            
            result = await agent.execute(pipeline_task)
            results.append(result)
            self.results.append(result)
            
            # Add output to context for next agent
            if pass_context and result.success:
                accumulated_context[f"{agent.role.value}_output"] = result.output[:1000]
        
        self._log_execution("pipeline", task, results, time.time() - start_time)
        return results
    
    async def run_debate(
        self, 
        task: Task, 
        rounds: int = 2,
        agent_ids: Optional[list[str]] = None
    ) -> list[AgentResult]:
        """
        Run agents in a debate format with multiple rounds.
        
        Each round, agents can see previous responses.
        Best for: Decision-making, exploring trade-offs.
        
        Args:
            task: The task/question to debate
            rounds: Number of debate rounds
            agent_ids: Specific agents to include (default: all)
        """
        start_time = time.time()
        all_results = []
        participating_agents = (
            [self.agents[aid] for aid in agent_ids if aid in self.agents]
            if agent_ids else list(self.agents.values())
        )
        
        for round_num in range(rounds):
            round_context = dict(task.context)
            round_context["debate_round"] = round_num + 1
            round_context["total_rounds"] = rounds
            
            # Add previous round's responses
            if all_results:
                prev_round_start = -len(participating_agents)
                prev_responses = {
                    r.agent_id: r.output[:400]
                    for r in all_results[prev_round_start:]
                }
                round_context["previous_responses"] = prev_responses
            
            # Run all agents for this round
            round_results = []
            for agent in participating_agents:
                debate_task = Task(
                    task_id=f"{task.task_id}_round{round_num}_{agent.agent_id}",
                    description=f"Round {round_num + 1}/{rounds}: {task.description}",
                    complexity=task.complexity,
                    context=round_context
                )
                
                result = await agent.execute(debate_task)
                round_results.append(result)
            
            all_results.extend(round_results)
            self.results.extend(round_results)
        
        self._log_execution("debate", task, all_results, time.time() - start_time)
        return all_results
    
    async def run_selective(
        self, 
        task: Task, 
        selector_fn
    ) -> list[AgentResult]:
        """
        Route task to specific agents based on a selector function.
        
        Args:
            task: The task to process
            selector_fn: Function(task, agents) -> list[agent_ids]
        """
        start_time = time.time()
        
        selected_ids = selector_fn(task, self.agents)
        selected_agents = [
            self.agents[aid] for aid in selected_ids 
            if aid in self.agents
        ]
        
        results = await asyncio.gather(*[
            agent.execute(task) for agent in selected_agents
        ])
        
        self.results.extend(results)
        self._log_execution("selective", task, list(results), time.time() - start_time)
        
        return list(results)
    
    def send_message(
        self, 
        sender_id: str, 
        receiver_id: str, 
        content: str,
        message_type: str = "info"
    ) -> Optional[Message]:
        """Send a message from one agent to another."""
        if receiver_id not in self.agents:
            return None
        
        message = Message(
            message_id=str(uuid4()),
            sender_id=sender_id,
            receiver_id=receiver_id,
            content=content,
            message_type=message_type
        )
        
        self.message_log.append(message)
        self.agents[receiver_id].receive_message(message)
        
        if sender_id in self.agents:
            self.agents[sender_id].metrics.messages_sent += 1
        
        return message
    
    def broadcast(self, sender_id: str, content: str) -> list[Message]:
        """Broadcast a message from one agent to all others."""
        messages = []
        for agent_id in self.agents:
            if agent_id != sender_id:
                msg = self.send_message(sender_id, agent_id, content, "broadcast")
                if msg:
                    messages.append(msg)
        return messages
    
    def _log_execution(
        self, 
        pattern: str, 
        task: Task, 
        results: list[AgentResult],
        total_time_s: float
    ) -> None:
        """Log execution details for analysis."""
        self.execution_history.append({
            "pattern": pattern,
            "task_id": task.task_id,
            "task_description": task.description[:100],
            "agent_count": len(results),
            "success_count": sum(1 for r in results if r.success),
            "total_time_s": total_time_s,
            "avg_quality": sum(r.quality_score for r in results) / len(results) if results else 0
        })
    
    def get_statistics(self) -> dict:
        """Get coordinator statistics."""
        if not self.results:
            return {"total_tasks": 0, "message": "No results yet"}
        
        successful = [r for r in self.results if r.success]
        
        return {
            "total_executions": len(self.execution_history),
            "total_results": len(self.results),
            "success_rate": len(successful) / len(self.results),
            "total_messages": len(self.message_log),
            "avg_latency_ms": sum(r.latency_ms for r in self.results) / len(self.results),
            "avg_quality": sum(r.quality_score for r in successful) / len(successful) if successful else 0,
            "total_tokens": sum(r.tokens_used for r in self.results),
            "agents": list(self.agents.keys())
        }
    
    def clear_history(self) -> None:
        """Clear all execution history and results."""
        self.results.clear()
        self.message_log.clear()
        self.execution_history.clear()
        for agent in self.agents.values():
            agent.clear_inbox()
