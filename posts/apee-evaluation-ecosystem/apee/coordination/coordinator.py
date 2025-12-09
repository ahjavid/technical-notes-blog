"""Multi-agent coordinator for task distribution and orchestration."""

import asyncio
import time
from typing import Optional, Sequence
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
    - Hierarchical: Leader delegates to workers and synthesizes results
    - Consensus: Iterate until agreement is reached
    - Peer Review: Work → review → revise cycle
    """
    
    def __init__(
        self, 
        agents: Sequence[Agent],
        context_truncation_limit: int = 500,
        output_truncation_limit: int = 1000
    ):
        self.agents: dict[str, Agent] = {a.agent_id: a for a in agents}
        self.message_log: list[Message] = []
        self.results: list[AgentResult] = []
        self.execution_history: list[dict] = []
        self.context_truncation_limit = context_truncation_limit
        self.output_truncation_limit = output_truncation_limit
    
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
        pass_context: bool = True,
        stop_on_failure: bool = False
    ) -> list[AgentResult]:
        """
        Run agents sequentially, optionally passing output to next agent.
        
        Best for: Code review workflows, multi-stage analysis.
        
        Args:
            task: The task to process
            agent_order: List of agent IDs in execution order
            pass_context: Whether to pass each agent's output to the next
            stop_on_failure: Whether to stop pipeline if an agent fails
        """
        import logging
        logger = logging.getLogger(__name__)
        
        start_time = time.time()
        results = []
        accumulated_context = dict(task.context)
        
        for i, agent_id in enumerate(agent_order):
            agent = self.agents.get(agent_id)
            if not agent:
                logger.warning(f"Pipeline: Agent '{agent_id}' not found, skipping step {i}")
                continue
            
            # Create task with accumulated context
            pipeline_task = Task(
                task_id=f"{task.task_id}_step{i}_{agent_id}",
                description=task.description,
                complexity=task.complexity,
                context=accumulated_context.copy()
            )
            
            try:
                result = await agent.execute(pipeline_task)
            except Exception as e:
                logger.error(f"Pipeline: Agent '{agent_id}' raised exception: {e}")
                result = AgentResult(
                    task_id=pipeline_task.task_id,
                    agent_id=agent_id,
                    agent_role=agent.role.value,
                    output="",
                    quality_score=0.0,
                    latency_ms=0.0,
                    success=False,
                    error=str(e)
                )
            
            results.append(result)
            self.results.append(result)
            
            # Check for failure if stop_on_failure is enabled
            if stop_on_failure and not result.success:
                logger.warning(f"Pipeline: Stopping due to failure at step {i} (agent '{agent_id}')")
                break
            
            # Add output to context for next agent
            if pass_context and result.success:
                accumulated_context[f"{agent.role.value}_output"] = result.output[:self.output_truncation_limit]
        
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
        
        Each round, agents can see previous responses and respond in parallel.
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
                    r.agent_id: r.output[:self.context_truncation_limit]
                    for r in all_results[prev_round_start:]
                }
                round_context["previous_responses"] = prev_responses
            
            # Prepare all debate tasks for this round
            debate_tasks = []
            for agent in participating_agents:
                debate_task = Task(
                    task_id=f"{task.task_id}_round{round_num}_{agent.agent_id}",
                    description=f"Round {round_num + 1}/{rounds}: {task.description}",
                    complexity=task.complexity,
                    context=round_context
                )
                debate_tasks.append((agent, debate_task))
            
            # Run all agents for this round IN PARALLEL
            round_results_raw = await asyncio.gather(*[
                agent.execute(debate_task) for agent, debate_task in debate_tasks
            ], return_exceptions=True)
            
            # Process results, converting exceptions to failed results
            round_results = []
            for (agent, _), result in zip(debate_tasks, round_results_raw):
                if isinstance(result, Exception):
                    result = AgentResult(
                        task_id=f"{task.task_id}_round{round_num}_{agent.agent_id}",
                        agent_id=agent.agent_id,
                        agent_role=agent.role.value,
                        output="",
                        quality_score=0.0,
                        latency_ms=0.0,
                        success=False,
                        error=str(result)
                    )
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
    
    async def run_hierarchical(
        self,
        task: Task,
        leader_id: str,
        worker_ids: Optional[list[str]] = None
    ) -> list[AgentResult]:
        """
        Run agents in a hierarchical pattern with a leader delegating to workers.
        
        The leader first analyzes the task and breaks it down, then workers
        execute subtasks, and finally the leader synthesizes results.
        
        Best for: Complex task breakdown, large projects.
        
        Args:
            task: The task to process
            leader_id: ID of the leader agent
            worker_ids: IDs of worker agents (default: all except leader)
        """
        start_time = time.time()
        results = []
        
        leader = self.agents.get(leader_id)
        if not leader:
            raise ValueError(f"Leader agent '{leader_id}' not found")
        
        # Default workers: all agents except leader
        if worker_ids is None:
            worker_ids = [aid for aid in self.agents.keys() if aid != leader_id]
        
        workers = [self.agents[wid] for wid in worker_ids if wid in self.agents]
        
        # Phase 1: Leader analyzes and plans
        planning_task = Task(
            task_id=f"{task.task_id}_plan",
            description=f"As team leader, analyze this task and create a plan. Task: {task.description}",
            complexity=task.complexity,
            context={
                **task.context,
                "role": "leader",
                "phase": "planning",
                "worker_count": len(workers),
            }
        )
        leader_plan = await leader.execute(planning_task)
        results.append(leader_plan)
        
        # Phase 2: Workers execute in parallel
        worker_tasks = []
        for i, worker in enumerate(workers):
            worker_task = Task(
                task_id=f"{task.task_id}_worker{i}",
                description=f"Execute your part of the task. Leader's guidance: {leader_plan.output[:self.context_truncation_limit]}\n\nOriginal task: {task.description}",
                complexity=task.complexity,
                context={
                    **task.context,
                    "role": "worker",
                    "phase": "execution",
                    "leader_plan": leader_plan.output[:self.output_truncation_limit],
                }
            )
            worker_tasks.append((worker, worker_task))
        
        worker_results = await asyncio.gather(*[
            w.execute(t) for w, t in worker_tasks
        ], return_exceptions=True)
        
        for i, result in enumerate(worker_results):
            if isinstance(result, Exception):
                result = AgentResult(
                    task_id=f"{task.task_id}_worker{i}",
                    agent_id=workers[i].agent_id,
                    agent_role=workers[i].role.value,
                    output="",
                    quality_score=0.0,
                    latency_ms=0.0,
                    success=False,
                    error=str(result)
                )
            results.append(result)
        
        # Phase 3: Leader synthesizes
        worker_outputs = "\n---\n".join([
            f"Worker {r.agent_id}: {r.output[:self.context_truncation_limit]}"
            for r in results[1:] if r.success
        ])
        
        synthesis_task = Task(
            task_id=f"{task.task_id}_synthesize",
            description=f"Synthesize worker outputs into a final cohesive result.\n\nWorker outputs:\n{worker_outputs}\n\nOriginal task: {task.description}",
            complexity=task.complexity,
            context={
                **task.context,
                "role": "leader",
                "phase": "synthesis",
            }
        )
        synthesis_result = await leader.execute(synthesis_task)
        results.append(synthesis_result)
        
        self.results.extend(results)
        self._log_execution("hierarchical", task, results, time.time() - start_time)
        
        return results
    
    async def run_consensus(
        self,
        task: Task,
        max_rounds: int = 3,
        agreement_threshold: float = 0.8,
        agent_ids: Optional[list[str]] = None
    ) -> list[AgentResult]:
        """
        Run agents until they reach consensus on the output.
        
        Agents iterate, viewing each other's responses, until sufficient
        agreement is reached or max rounds exceeded.
        
        Best for: Critical decisions, validation tasks.
        
        Args:
            task: The task to process
            max_rounds: Maximum number of consensus rounds
            agreement_threshold: Required ratio of agreeing agents (0-1)
            agent_ids: Specific agents to include (default: all)
        """
        start_time = time.time()
        all_results = []
        
        participating_agents = (
            [self.agents[aid] for aid in agent_ids if aid in self.agents]
            if agent_ids else list(self.agents.values())
        )
        
        if len(participating_agents) < 2:
            raise ValueError("Consensus requires at least 2 agents")
        
        consensus_reached = False
        
        for round_num in range(max_rounds):
            round_context = dict(task.context)
            round_context["consensus_round"] = round_num + 1
            round_context["max_rounds"] = max_rounds
            round_context["goal"] = "reach agreement with other agents"
            
            # Add previous round's responses
            if all_results:
                prev_round_start = -len(participating_agents)
                prev_responses = {
                    r.agent_id: r.output[:self.context_truncation_limit]
                    for r in all_results[prev_round_start:]
                }
                round_context["other_agent_responses"] = prev_responses
                round_context["instruction"] = (
                    "Review other agents' responses and adjust your answer to find common ground. "
                    "If you agree with the consensus, clearly state 'I AGREE' at the start. "
                    "If you disagree, clearly state 'I DISAGREE' at the start and explain your reasoning."
                )
            
            # Prepare all consensus tasks for this round
            consensus_tasks = []
            for agent in participating_agents:
                consensus_task = Task(
                    task_id=f"{task.task_id}_consensus_r{round_num}_{agent.agent_id}",
                    description=task.description,
                    complexity=task.complexity,
                    context=round_context
                )
                consensus_tasks.append((agent, consensus_task))
            
            # Run all agents for this round IN PARALLEL
            round_results_raw = await asyncio.gather(*[
                agent.execute(consensus_task) for agent, consensus_task in consensus_tasks
            ], return_exceptions=True)
            
            # Process results, converting exceptions to failed results
            round_results = []
            for (agent, _), result in zip(consensus_tasks, round_results_raw):
                if isinstance(result, Exception):
                    result = AgentResult(
                        task_id=f"{task.task_id}_consensus_r{round_num}_{agent.agent_id}",
                        agent_id=agent.agent_id,
                        agent_role=agent.role.value,
                        output="",
                        quality_score=0.0,
                        latency_ms=0.0,
                        success=False,
                        error=str(result)
                    )
                round_results.append(result)
            
            all_results.extend(round_results)
            
            # Improved consensus detection with semantic analysis
            consensus_reached = self._check_consensus(round_results, agreement_threshold)
            
            if consensus_reached:
                break
        
        self.results.extend(all_results)
        self._log_execution(
            f"consensus{'_reached' if consensus_reached else '_partial'}",
            task, all_results, time.time() - start_time
        )
        
        return all_results
    
    def _check_consensus(
        self, 
        results: list[AgentResult], 
        threshold: float
    ) -> bool:
        """
        Check if agents have reached consensus using improved semantic analysis.
        
        Uses multiple heuristics:
        1. Explicit agreement/disagreement markers
        2. Negation-aware keyword detection
        3. Output similarity comparison
        
        Args:
            results: Results from the current round
            threshold: Required agreement ratio (0-1)
            
        Returns:
            True if consensus is reached
        """
        if not results:
            return False
        
        successful_results = [r for r in results if r.success]
        if len(successful_results) < 2:
            return False
        
        agreement_scores = []
        
        for result in successful_results:
            output_lower = result.output.lower()
            
            # Check for explicit markers (strongest signal)
            if output_lower.startswith("i agree") or "i agree with" in output_lower[:100]:
                agreement_scores.append(1.0)
                continue
            if output_lower.startswith("i disagree") or "i disagree with" in output_lower[:100]:
                agreement_scores.append(0.0)
                continue
            
            # Negation-aware keyword analysis
            score = self._analyze_agreement_sentiment(output_lower)
            agreement_scores.append(score)
        
        # Also check output similarity (if outputs are very similar, likely consensus)
        similarity_bonus = self._calculate_output_similarity(successful_results)
        
        # Combine explicit agreement ratio with similarity
        avg_agreement = sum(agreement_scores) / len(agreement_scores) if agreement_scores else 0
        combined_score = 0.7 * avg_agreement + 0.3 * similarity_bonus
        
        return combined_score >= threshold
    
    def _analyze_agreement_sentiment(self, text: str) -> float:
        """
        Analyze text for agreement/disagreement sentiment with negation awareness.
        
        Returns a score from 0 (strong disagreement) to 1 (strong agreement).
        """
        # Positive indicators (agreement)
        positive_patterns = [
            "agree", "consensus", "concur", "same conclusion", "aligned",
            "correct", "right", "exactly", "precisely", "indeed",
            "support this", "endorse", "confirm"
        ]
        
        # Negative indicators (disagreement)
        negative_patterns = [
            "disagree", "different view", "alternative", "however",
            "but i think", "on the contrary", "instead", "rather",
            "not quite", "partially", "reservations", "concerns"
        ]
        
        # Negation words that flip meaning
        negations = ["not ", "don't ", "doesn't ", "cannot ", "can't ", "won't ", "wouldn't "]
        
        positive_count = 0
        negative_count = 0
        
        for pattern in positive_patterns:
            if pattern in text:
                # Check if negated
                pattern_idx = text.find(pattern)
                context_start = max(0, pattern_idx - 20)
                context = text[context_start:pattern_idx]
                
                if any(neg in context for neg in negations):
                    negative_count += 1  # Negated positive = negative
                else:
                    positive_count += 1
        
        for pattern in negative_patterns:
            if pattern in text:
                # Check if negated
                pattern_idx = text.find(pattern)
                context_start = max(0, pattern_idx - 20)
                context = text[context_start:pattern_idx]
                
                if any(neg in context for neg in negations):
                    positive_count += 1  # Negated negative = positive
                else:
                    negative_count += 1
        
        total = positive_count + negative_count
        if total == 0:
            return 0.5  # Neutral
        
        return positive_count / total
    
    def _calculate_output_similarity(self, results: list[AgentResult]) -> float:
        """
        Calculate average pairwise similarity between outputs.
        
        Uses simple word overlap (Jaccard similarity) as a lightweight metric.
        For production, consider using embeddings.
        """
        if len(results) < 2:
            return 1.0
        
        def tokenize(text: str) -> set:
            # Simple word tokenization
            import re
            words = re.findall(r'\b\w+\b', text.lower())
            return set(words)
        
        def jaccard_similarity(set1: set, set2: set) -> float:
            if not set1 and not set2:
                return 1.0
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            return intersection / union if union > 0 else 0.0
        
        # Calculate pairwise similarities
        similarities = []
        token_sets = [tokenize(r.output) for r in results]
        
        for i in range(len(token_sets)):
            for j in range(i + 1, len(token_sets)):
                sim = jaccard_similarity(token_sets[i], token_sets[j])
                similarities.append(sim)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    async def run_peer_review(
        self,
        task: Task,
        agent_ids: Optional[list[str]] = None
    ) -> list[AgentResult]:
        """
        Run agents where each reviews others' work.
        
        Phase 1: All agents produce initial output (parallel)
        Phase 2: Each agent reviews one other agent's work (parallel)
        Phase 3: Original authors respond to feedback (parallel)
        
        Best for: Code review, document review, quality assurance.
        
        Args:
            task: The task to process
            agent_ids: Specific agents to include (default: all)
        """
        start_time = time.time()
        results = []
        
        participating_agents = (
            [self.agents[aid] for aid in agent_ids if aid in self.agents]
            if agent_ids else list(self.agents.values())
        )
        
        if len(participating_agents) < 2:
            raise ValueError("Peer review requires at least 2 agents")
        
        n_agents = len(participating_agents)
        
        # Phase 1: Initial work (PARALLEL)
        initial_tasks = [
            (agent, Task(
                task_id=f"{task.task_id}_initial_{agent.agent_id}",
                description=task.description,
                complexity=task.complexity,
                context={**task.context, "phase": "initial_work"}
            ))
            for agent in participating_agents
        ]
        
        initial_results_raw = await asyncio.gather(*[
            agent.execute(t) for agent, t in initial_tasks
        ], return_exceptions=True)
        
        initial_results = []
        for (agent, t), result in zip(initial_tasks, initial_results_raw):
            if isinstance(result, Exception):
                result = AgentResult(
                    task_id=t.task_id,
                    agent_id=agent.agent_id,
                    agent_role=agent.role.value,
                    output="",
                    quality_score=0.0,
                    latency_ms=0.0,
                    success=False,
                    error=str(result)
                )
            initial_results.append(result)
        
        results.extend(initial_results)
        
        # Phase 2: Peer review (circular: agent i reviews agent (i+1) % n) (PARALLEL)
        review_tasks = []
        for i, reviewer in enumerate(participating_agents):
            reviewee_idx = (i + 1) % n_agents
            reviewee_work = initial_results[reviewee_idx]
            reviewee_name = participating_agents[reviewee_idx].agent_id
            
            review_task = Task(
                task_id=f"{task.task_id}_review_{reviewer.agent_id}",
                description=f"Review {reviewee_name}'s work and provide constructive feedback.\n\nOriginal task: {task.description}\n\n{reviewee_name}'s work:\n{reviewee_work.output[:self.output_truncation_limit]}",
                complexity=task.complexity,
                context={
                    **task.context,
                    "phase": "peer_review",
                    "reviewing": reviewee_name,
                }
            )
            review_tasks.append((reviewer, review_task))
        
        review_results_raw = await asyncio.gather(*[
            agent.execute(t) for agent, t in review_tasks
        ], return_exceptions=True)
        
        review_results = []
        for (agent, t), result in zip(review_tasks, review_results_raw):
            if isinstance(result, Exception):
                result = AgentResult(
                    task_id=t.task_id,
                    agent_id=agent.agent_id,
                    agent_role=agent.role.value,
                    output="",
                    quality_score=0.0,
                    latency_ms=0.0,
                    success=False,
                    error=str(result)
                )
            review_results.append(result)
        
        results.extend(review_results)
        
        # Phase 3: Authors respond to feedback (PARALLEL)
        # Agent (i-1) % n reviewed agent i, so review_results[(i-1) % n] is the feedback for agent i
        response_tasks = []
        for i, author in enumerate(participating_agents):
            # Find who reviewed this author: agent (i-1) % n reviewed agent i
            reviewer_idx = (i - 1) % n_agents
            feedback = review_results[reviewer_idx]
            reviewer_name = participating_agents[reviewer_idx].agent_id
            
            response_task = Task(
                task_id=f"{task.task_id}_response_{author.agent_id}",
                description=f"Consider {reviewer_name}'s feedback and provide your revised/final response.\n\nOriginal task: {task.description}\n\nYour original work:\n{initial_results[i].output[:self.context_truncation_limit]}\n\nFeedback from {reviewer_name}:\n{feedback.output[:self.output_truncation_limit]}",
                complexity=task.complexity,
                context={
                    **task.context,
                    "phase": "revision",
                    "feedback_from": reviewer_name,
                }
            )
            response_tasks.append((author, response_task))
        
        response_results_raw = await asyncio.gather(*[
            agent.execute(t) for agent, t in response_tasks
        ], return_exceptions=True)
        
        response_results = []
        for (agent, t), result in zip(response_tasks, response_results_raw):
            if isinstance(result, Exception):
                result = AgentResult(
                    task_id=t.task_id,
                    agent_id=agent.agent_id,
                    agent_role=agent.role.value,
                    output="",
                    quality_score=0.0,
                    latency_ms=0.0,
                    success=False,
                    error=str(result)
                )
            response_results.append(result)
        
        results.extend(response_results)
        
        self.results.extend(results)
        self._log_execution("peer_review", task, results, time.time() - start_time)
        
        return results
    
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
