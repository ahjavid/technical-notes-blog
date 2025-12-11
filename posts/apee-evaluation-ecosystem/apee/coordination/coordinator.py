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
        context_truncation_limit: int = 2048,
        output_truncation_limit: int = 3072
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
        stop_on_failure: bool = False,
        require_build_on_previous: bool = True
    ) -> list[AgentResult]:
        """
        Run agents sequentially with explicit handoff and build-on-previous requirements.
        
        Each agent receives the previous agent's output and must explicitly build on it.
        Best for: Code review workflows, multi-stage analysis, knowledge transfer.
        
        Args:
            task: The task to process
            agent_order: List of agent IDs in execution order
            pass_context: Whether to pass each agent's output to the next
            stop_on_failure: Whether to stop pipeline if an agent fails
            require_build_on_previous: If True, prompts explicitly require building on previous work
        """
        import logging
        logger = logging.getLogger(__name__)
        
        start_time = time.time()
        results = []
        accumulated_context = dict(task.context)
        all_previous_outputs = []  # Track ALL previous outputs for synthesis
        
        total_steps = len(agent_order)
        
        for i, agent_id in enumerate(agent_order):
            agent = self.agents.get(agent_id)
            if not agent:
                logger.warning(f"Pipeline: Agent '{agent_id}' not found, skipping step {i}")
                continue
            
            step_num = i + 1
            
            # Build stage-specific instructions
            if i == 0:
                # First agent - initial analysis
                stage_instruction = (
                    f"PIPELINE Stage {step_num}/{total_steps}: Initial Analysis\n\n"
                    f"You are the FIRST agent in a sequential pipeline. "
                    f"Your output will be passed to the next agent who will build on it.\n"
                    f"Provide a thorough foundation that others can extend."
                )
                task_desc = f"{stage_instruction}\n\nTask: {task.description}"
            elif i == total_steps - 1:
                # Last agent - final synthesis
                all_prev_formatted = "\n\n".join([
                    f"=== Stage {j+1} ({prev['agent_id']}, {prev['role']}) ===\n{prev['output']}"
                    for j, prev in enumerate(all_previous_outputs)
                ])
                stage_instruction = (
                    f"PIPELINE Stage {step_num}/{total_steps}: Final Synthesis\n\n"
                    f"You are the FINAL agent. You MUST:\n"
                    f"1. INTEGRATE all previous contributions (don't ignore any)\n"
                    f"2. SYNTHESIZE a cohesive final output\n"
                    f"3. ACKNOWLEDGE key insights from each previous stage\n"
                    f"4. ADD your own perspective/refinements\n\n"
                    f"=== ALL Previous Stage Outputs ===\n{all_prev_formatted}"
                )
                task_desc = f"{stage_instruction}\n\nOriginal Task: {task.description}"
            else:
                # Middle agent - build on previous
                prev_output = all_previous_outputs[-1] if all_previous_outputs else None
                all_prev_formatted = "\n\n".join([
                    f"=== Stage {j+1} ({prev['agent_id']}, {prev['role']}) ===\n{prev['output']}"
                    for j, prev in enumerate(all_previous_outputs)
                ])
                stage_instruction = (
                    f"PIPELINE Stage {step_num}/{total_steps}: Build & Extend\n\n"
                    f"You are in the MIDDLE of a sequential pipeline. You MUST:\n"
                    f"1. BUILD ON the previous agent(s) work - don't start from scratch\n"
                    f"2. EXTEND with your expertise ({agent.role.value})\n"
                    f"3. REFERENCE specific points from previous stages\n"
                    f"4. ADD new value while preserving previous insights\n\n"
                    f"=== Previous Stage Outputs ===\n{all_prev_formatted}"
                )
                task_desc = f"{stage_instruction}\n\nOriginal Task: {task.description}"
            
            # Create task with stage instructions
            pipeline_task = Task(
                task_id=f"{task.task_id}_stage{step_num}_{agent_id}",
                description=task_desc,
                complexity=task.complexity,
                context={
                    **accumulated_context,
                    "pipeline_stage": step_num,
                    "total_stages": total_steps,
                    "previous_outputs": all_previous_outputs.copy(),
                }
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
            
            # Log message handoff to next agent
            if i < total_steps - 1 and result.success:
                next_agent_id = agent_order[i + 1] if i + 1 < len(agent_order) else None
                if next_agent_id and next_agent_id in self.agents:
                    self.send_message(
                        agent_id,
                        next_agent_id,
                        f"[Pipeline Handoff Stage {step_num}→{step_num+1}] {result.output[:1000]}",
                        message_type="handoff"
                    )
            
            # Check for failure if stop_on_failure is enabled
            if stop_on_failure and not result.success:
                logger.warning(f"Pipeline: Stopping due to failure at step {i} (agent '{agent_id}')")
                break
            
            # Track all outputs for later agents
            if result.success:
                all_previous_outputs.append({
                    "agent_id": agent_id,
                    "role": agent.role.value,
                    "output": result.output[:self.output_truncation_limit],
                    "stage": step_num,
                })
                
                # Also add to accumulated context for backward compatibility
                if pass_context:
                    accumulated_context[f"{agent.role.value}_output"] = result.output[:self.output_truncation_limit]
                    accumulated_context[f"stage{step_num}_output"] = result.output[:self.output_truncation_limit]
        
        self._log_execution("pipeline", task, results, time.time() - start_time)
        return results
    
    async def run_debate(
        self, 
        task: Task, 
        rounds: int = 3,
        agent_ids: Optional[list[str]] = None,
        require_response_to_others: bool = True
    ) -> list[AgentResult]:
        """
        Run agents in a debate format with multiple rounds and explicit agent-to-agent responses.
        
        Each round, agents MUST respond to and engage with other agents' previous arguments.
        Best for: Decision-making, exploring trade-offs, adversarial review.
        
        Args:
            task: The task/question to debate
            rounds: Number of debate rounds (default: 3 for meaningful exchange)
            agent_ids: Specific agents to include (default: all)
            require_response_to_others: If True, prompts explicitly require responding to others
        """
        start_time = time.time()
        all_results = []
        participating_agents = (
            [self.agents[aid] for aid in agent_ids if aid in self.agents]
            if agent_ids else list(self.agents.values())
        )
        
        if len(participating_agents) < 2:
            raise ValueError("Debate requires at least 2 agents")
        
        for round_num in range(rounds):
            round_context = dict(task.context)
            round_context["debate_round"] = round_num + 1
            round_context["total_rounds"] = rounds
            
            # Build debate instructions based on round
            if round_num == 0:
                debate_instruction = (
                    "This is a DEBATE. Present your initial position on the task. "
                    "Be clear about your stance and reasoning. Other agents will respond to you."
                )
            else:
                debate_instruction = (
                    f"This is round {round_num + 1} of a DEBATE. "
                    "You MUST engage with other agents' arguments:\n"
                    "1. Acknowledge points you agree with from specific agents\n"
                    "2. Counter arguments you disagree with - cite which agent and why\n"
                    "3. Refine or strengthen your position based on the discussion\n"
                    "4. Try to find synthesis where possible"
                )
            
            round_context["debate_instruction"] = debate_instruction
            
            # Add previous round's responses with explicit formatting for engagement
            if all_results:
                prev_round_start = -len(participating_agents)
                prev_responses = {}
                formatted_responses = []
                
                for r in all_results[prev_round_start:]:
                    prev_responses[r.agent_id] = r.output[:self.context_truncation_limit]
                    formatted_responses.append(
                        f"=== {r.agent_id} ({r.agent_role}) said ===\n{r.output[:self.context_truncation_limit]}\n"
                    )
                
                round_context["previous_responses"] = prev_responses
                round_context["formatted_debate_history"] = "\n".join(formatted_responses)
                
                # Log messages between agents for tracking
                for r in all_results[prev_round_start:]:
                    for other_agent in participating_agents:
                        if other_agent.agent_id != r.agent_id:
                            self.send_message(
                                r.agent_id, 
                                other_agent.agent_id, 
                                f"[Round {round_num}] {r.output[:500]}",
                                message_type="debate"
                            )
            
            # Prepare debate tasks with explicit engagement requirements
            debate_tasks = []
            for agent in participating_agents:
                if round_num == 0:
                    task_desc = f"DEBATE Round {round_num + 1}/{rounds}: {task.description}"
                else:
                    task_desc = (
                        f"DEBATE Round {round_num + 1}/{rounds}: {task.description}\n\n"
                        f"IMPORTANT: You must respond to the other agents' arguments below.\n\n"
                        f"{round_context.get('formatted_debate_history', '')}"
                    )
                
                debate_task = Task(
                    task_id=f"{task.task_id}_debate_r{round_num}_{agent.agent_id}",
                    description=task_desc,
                    complexity=task.complexity,
                    context=round_context
                )
                debate_tasks.append((agent, debate_task))
            
            # Run all agents for this round IN PARALLEL
            round_results_raw = await asyncio.gather(*[
                agent.execute(debate_task) for agent, debate_task in debate_tasks
            ], return_exceptions=True)
            
            # Process results
            round_results = []
            for (agent, _), result in zip(debate_tasks, round_results_raw):
                if isinstance(result, Exception):
                    result = AgentResult(
                        task_id=f"{task.task_id}_debate_r{round_num}_{agent.agent_id}",
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
        
        # Log final round messages (loop only logs previous round's messages)
        if all_results:
            final_round_start = -len(participating_agents)
            for r in all_results[final_round_start:]:
                for other_agent in participating_agents:
                    if other_agent.agent_id != r.agent_id:
                        self.send_message(
                            r.agent_id, 
                            other_agent.agent_id, 
                            f"[Final Round {rounds}] {r.output[:500]}",
                            message_type="debate"
                        )
        
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
        worker_ids: Optional[list[str]] = None,
        include_feedback_round: bool = True
    ) -> list[AgentResult]:
        """
        Run agents in a hierarchical pattern with explicit delegation and feedback.
        
        Phase 1: Leader analyzes the task and creates delegated work assignments
        Phase 2: Workers execute their assignments in parallel
        Phase 3 (optional): Workers report back and leader provides feedback
        Phase 4: Leader synthesizes all results into final output
        
        Best for: Complex task breakdown, large projects, team coordination.
        
        Args:
            task: The task to process
            leader_id: ID of the leader agent
            worker_ids: IDs of worker agents (default: all except leader)
            include_feedback_round: If True, includes worker feedback and leader response
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
        
        if len(workers) < 1:
            raise ValueError("Hierarchical pattern requires at least 1 worker agent")
        
        worker_names = [w.agent_id for w in workers]
        
        # Phase 1: Leader analyzes and creates specific assignments
        planning_task = Task(
            task_id=f"{task.task_id}_plan",
            description=(
                f"HIERARCHICAL COORDINATION - Phase 1: Planning & Delegation\n\n"
                f"You are the TEAM LEADER. Analyze this task and create a plan.\n\n"
                f"Task: {task.description}\n\n"
                f"Your team members: {', '.join(worker_names)}\n\n"
                f"Provide:\n"
                f"1. TASK ANALYSIS: Break down the task into components\n"
                f"2. DELEGATION: Assign specific work to each team member by name\n"
                f"3. COORDINATION NOTES: How should team members' work fit together?\n"
                f"4. SUCCESS CRITERIA: How will you evaluate the final result?"
            ),
            complexity=task.complexity,
            context={
                **task.context,
                "role": "leader",
                "phase": "planning",
                "worker_count": len(workers),
                "worker_names": worker_names,
            }
        )
        leader_plan = await leader.execute(planning_task)
        results.append(leader_plan)
        
        # Send delegation messages to workers
        for worker in workers:
            self.send_message(
                leader_id,
                worker.agent_id,
                f"[Delegation] {leader_plan.output[:1000]}",
                message_type="delegation"
            )
        
        # Phase 2: Workers execute their assigned work
        worker_tasks = []
        for i, worker in enumerate(workers):
            worker_task = Task(
                task_id=f"{task.task_id}_worker_{worker.agent_id}",
                description=(
                    f"HIERARCHICAL COORDINATION - Phase 2: Worker Execution\n\n"
                    f"You are a TEAM MEMBER working under {leader_id}'s direction.\n\n"
                    f"=== Leader's Plan & Your Assignment ===\n"
                    f"{leader_plan.output[:self.context_truncation_limit]}\n\n"
                    f"=== Original Task ===\n"
                    f"{task.description}\n\n"
                    f"Execute YOUR assigned part of the work. Be thorough and specific.\n"
                    f"If your assignment is unclear, use your best judgment based on the overall task."
                ),
                complexity=task.complexity,
                context={
                    **task.context,
                    "role": "worker",
                    "phase": "execution",
                    "leader_id": leader_id,
                    "leader_plan": leader_plan.output[:self.output_truncation_limit],
                }
            )
            worker_tasks.append((worker, worker_task))
        
        worker_results_raw = await asyncio.gather(*[
            w.execute(t) for w, t in worker_tasks
        ], return_exceptions=True)
        
        worker_results = []
        for i, (worker, _) in enumerate(worker_tasks):
            result = worker_results_raw[i]
            if isinstance(result, Exception):
                result = AgentResult(
                    task_id=f"{task.task_id}_worker_{worker.agent_id}",
                    agent_id=worker.agent_id,
                    agent_role=worker.role.value,
                    output="",
                    quality_score=0.0,
                    latency_ms=0.0,
                    success=False,
                    error=str(result)
                )
            else:
                # Workers report back to leader
                self.send_message(
                    worker.agent_id,
                    leader_id,
                    f"[Work Complete] {result.output[:1000]}",
                    message_type="report"
                )
            worker_results.append(result)
            results.append(result)
        
        # Phase 3 (optional): Feedback round
        if include_feedback_round:
            # Leader provides feedback to each worker
            worker_outputs_for_feedback = "\n\n".join([
                f"=== {workers[i].agent_id}'s Work ===\n{r.output[:self.context_truncation_limit]}"
                for i, r in enumerate(worker_results) if r.success
            ])
            
            feedback_task = Task(
                task_id=f"{task.task_id}_feedback",
                description=(
                    f"HIERARCHICAL COORDINATION - Phase 3: Leader Feedback\n\n"
                    f"Review your team members' work and provide feedback.\n\n"
                    f"=== Original Task ===\n{task.description}\n\n"
                    f"=== Team Outputs ===\n{worker_outputs_for_feedback}\n\n"
                    f"Provide SPECIFIC FEEDBACK for each team member:\n"
                    f"1. What they did well\n"
                    f"2. What could be improved\n"
                    f"3. Any corrections or additions needed"
                ),
                complexity=task.complexity,
                context={
                    **task.context,
                    "role": "leader",
                    "phase": "feedback",
                }
            )
            
            feedback_result = await leader.execute(feedback_task)
            results.append(feedback_result)
            
            # Send feedback to workers
            for worker in workers:
                self.send_message(
                    leader_id,
                    worker.agent_id,
                    f"[Feedback] {feedback_result.output[:1000]}",
                    message_type="feedback"
                )
        
        # Phase 4: Leader synthesizes final output
        worker_outputs = "\n---\n".join([
            f"=== {workers[i].agent_id} ({workers[i].role.value}) ===\n{r.output[:self.context_truncation_limit]}"
            for i, r in enumerate(worker_results) if r.success
        ])
        
        synthesis_task = Task(
            task_id=f"{task.task_id}_synthesize",
            description=(
                f"HIERARCHICAL COORDINATION - Phase 4: Final Synthesis\n\n"
                f"As team leader, synthesize all team outputs into a cohesive final result.\n\n"
                f"=== Original Task ===\n{task.description}\n\n"
                f"=== Team Outputs ===\n{worker_outputs}\n\n"
                f"Create a UNIFIED FINAL OUTPUT that:\n"
                f"1. Integrates the best elements from each team member's work\n"
                f"2. Resolves any contradictions or inconsistencies\n"
                f"3. Fills any gaps in the collective output\n"
                f"4. Provides a complete, polished response to the original task"
            ),
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
        max_rounds: int = 4,
        agreement_threshold: float = 0.8,
        agent_ids: Optional[list[str]] = None,
        require_explicit_engagement: bool = True
    ) -> list[AgentResult]:
        """
        Run agents until they reach consensus on the output with explicit agent-to-agent engagement.
        
        Agents iterate, explicitly responding to each other's viewpoints, until sufficient
        agreement is reached or max rounds exceeded.
        
        Best for: Critical decisions, validation tasks, conflict resolution.
        
        Args:
            task: The task to process
            max_rounds: Maximum number of consensus rounds (default: 4 for thorough discussion)
            agreement_threshold: Required ratio of agreeing agents (0-1)
            agent_ids: Specific agents to include (default: all)
            require_explicit_engagement: If True, agents must explicitly respond to others
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
            
            # Build round-specific instructions
            if round_num == 0:
                round_context["instruction"] = (
                    "CONSENSUS BUILDING: Present your initial analysis and position on this task. "
                    "Be clear about your reasoning. In subsequent rounds, you will need to find "
                    "common ground with other agents."
                )
            else:
                # Add previous round's responses with explicit formatting
                prev_round_start = -len(participating_agents)
                prev_responses = {}
                formatted_positions = []
                
                for r in all_results[prev_round_start:]:
                    prev_responses[r.agent_id] = r.output[:self.context_truncation_limit]
                    # Detect agreement/disagreement stance
                    stance = "unclear"
                    if r.output.lower().startswith("i agree"):
                        stance = "AGREES"
                    elif r.output.lower().startswith("i disagree"):
                        stance = "DISAGREES"
                    formatted_positions.append(
                        f"=== {r.agent_id} ({r.agent_role}) [{stance}] ===\n{r.output[:self.context_truncation_limit]}\n"
                    )
                    
                    # Log messages for tracking
                    for other_agent in participating_agents:
                        if other_agent.agent_id != r.agent_id:
                            self.send_message(
                                r.agent_id,
                                other_agent.agent_id,
                                f"[Consensus Round {round_num}] {r.output[:500]}",
                                message_type="consensus"
                            )
                
                round_context["other_agent_responses"] = prev_responses
                round_context["formatted_positions"] = "\n".join(formatted_positions)
                
                round_context["instruction"] = (
                    f"CONSENSUS BUILDING - Round {round_num + 1}/{max_rounds}\n\n"
                    "You MUST engage with other agents' positions:\n"
                    "1. If you AGREE with the emerging consensus, start with 'I AGREE' and explain why\n"
                    "2. If you DISAGREE, start with 'I DISAGREE' and cite specific points from other agents\n"
                    "3. Look for COMMON GROUND - what do most agents agree on?\n"
                    "4. Propose COMPROMISES where there are differences\n"
                    "5. Ask clarifying questions if needed\n\n"
                    f"Other agents' positions:\n{round_context['formatted_positions']}"
                )
            
            # Prepare consensus tasks
            consensus_tasks = []
            for agent in participating_agents:
                if round_num == 0:
                    task_desc = f"CONSENSUS Round {round_num + 1}/{max_rounds}: {task.description}"
                else:
                    task_desc = (
                        f"CONSENSUS Round {round_num + 1}/{max_rounds}: {task.description}\n\n"
                        f"{round_context['instruction']}"
                    )
                
                consensus_task = Task(
                    task_id=f"{task.task_id}_consensus_r{round_num}_{agent.agent_id}",
                    description=task_desc,
                    complexity=task.complexity,
                    context=round_context
                )
                consensus_tasks.append((agent, consensus_task))
            
            # Run all agents for this round IN PARALLEL
            round_results_raw = await asyncio.gather(*[
                agent.execute(consensus_task) for agent, consensus_task in consensus_tasks
            ], return_exceptions=True)
            
            # Process results
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
            
            # Check consensus with improved detection
            consensus_reached = self._check_consensus(round_results, agreement_threshold)
            
            if consensus_reached:
                break
        
        # Log final round messages (loop only logs previous round's messages)
        if all_results:
            final_round_start = -len(participating_agents)
            for r in all_results[final_round_start:]:
                for other_agent in participating_agents:
                    if other_agent.agent_id != r.agent_id:
                        self.send_message(
                            r.agent_id, 
                            other_agent.agent_id, 
                            f"[Final Consensus Round] {r.output[:500]}",
                            message_type="consensus"
                        )
        
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
        import re
        
        # Positive indicators (agreement) - use word boundaries
        positive_patterns = [
            r"\bagree\b", r"\bconsensus\b", r"\bconcur\b", r"\bsame conclusion\b", r"\baligned\b",
            r"\bcorrect\b", r"\bright\b", r"\bexactly\b", r"\bprecisely\b", r"\bindeed\b",
            r"\bsupport this\b", r"\bendorse\b", r"\bconfirm\b"
        ]
        
        # Negative indicators (disagreement) - use word boundaries
        negative_patterns = [
            r"\bdisagree\b", r"\bdifferent view\b", r"\balternative\b", r"\bhowever\b",
            r"\bbut i think\b", r"\bon the contrary\b", r"\binstead\b", r"\brather\b",
            r"\bnot quite\b", r"\bpartially\b", r"\breservations\b", r"\bconcerns\b"
        ]
        
        # Negation words that flip meaning
        negations = ["not ", "don't ", "doesn't ", "cannot ", "can't ", "won't ", "wouldn't "]
        
        positive_count = 0
        negative_count = 0
        
        for pattern in positive_patterns:
            match = re.search(pattern, text)
            if match:
                # Check if negated
                pattern_idx = match.start()
                context_start = max(0, pattern_idx - 20)
                context = text[context_start:pattern_idx]
                
                if any(neg in context for neg in negations):
                    negative_count += 1  # Negated positive = negative
                else:
                    positive_count += 1
        
        for pattern in negative_patterns:
            match = re.search(pattern, text)
            if match:
                # Check if negated
                pattern_idx = match.start()
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
        agent_ids: Optional[list[str]] = None,
        include_meta_review: bool = True
    ) -> list[AgentResult]:
        """
        Run agents where each reviews others' work with explicit feedback exchange.
        
        Phase 1: All agents produce initial output (parallel)
        Phase 2: Each agent reviews one other agent's work with specific feedback (parallel)
        Phase 3: Original authors respond to feedback and revise (parallel)
        Phase 4 (optional): Meta-review synthesizing all perspectives
        
        Best for: Code review, document review, quality assurance.
        
        Args:
            task: The task to process
            agent_ids: Specific agents to include (default: all)
            include_meta_review: If True, adds a final synthesis phase
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
                description=(
                    f"PEER REVIEW - Phase 1: Initial Work\n\n"
                    f"Task: {task.description}\n\n"
                    f"Produce your best work on this task. Your output will be reviewed "
                    f"by another agent, and you will have a chance to revise based on feedback."
                ),
                complexity=task.complexity,
                context={**task.context, "phase": "initial_work", "review_phase": 1}
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
        
        # Phase 2: Peer review with explicit feedback structure (circular assignment)
        review_tasks = []
        for i, reviewer in enumerate(participating_agents):
            reviewee_idx = (i + 1) % n_agents
            reviewee_work = initial_results[reviewee_idx]
            reviewee_agent = participating_agents[reviewee_idx]
            reviewee_name = reviewee_agent.agent_id
            
            # Log the review assignment
            self.send_message(
                "coordinator",
                reviewer.agent_id,
                f"You are assigned to review {reviewee_name}'s work",
                message_type="assignment"
            )
            
            review_task = Task(
                task_id=f"{task.task_id}_review_{reviewer.agent_id}",
                description=(
                    f"PEER REVIEW - Phase 2: Review {reviewee_name}'s Work\n\n"
                    f"Original task: {task.description}\n\n"
                    f"=== {reviewee_name}'s Work ===\n"
                    f"{reviewee_work.output[:self.output_truncation_limit]}\n\n"
                    f"Provide a STRUCTURED REVIEW with:\n"
                    f"1. STRENGTHS: What did {reviewee_name} do well? (be specific)\n"
                    f"2. WEAKNESSES: What needs improvement? (be specific)\n"
                    f"3. SUGGESTIONS: Concrete recommendations for improvement\n"
                    f"4. QUESTIONS: Any clarifications needed?\n"
                    f"5. OVERALL ASSESSMENT: Summary evaluation"
                ),
                complexity=task.complexity,
                context={
                    **task.context,
                    "phase": "peer_review",
                    "review_phase": 2,
                    "reviewing": reviewee_name,
                    "reviewee_role": reviewee_agent.role.value,
                }
            )
            review_tasks.append((reviewer, review_task, reviewee_name))
        
        review_results_raw = await asyncio.gather(*[
            agent.execute(t) for agent, t, _ in review_tasks
        ], return_exceptions=True)
        
        review_results = []
        for (agent, t, reviewee_name), result in zip(review_tasks, review_results_raw):
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
            else:
                # Log the review feedback as a message
                self.send_message(
                    agent.agent_id,
                    reviewee_name,
                    f"[Review Feedback] {result.output[:1000]}",
                    message_type="review"
                )
            review_results.append(result)
        
        results.extend(review_results)
        
        # Phase 3: Authors respond to feedback and revise (PARALLEL)
        response_tasks = []
        for i, author in enumerate(participating_agents):
            # Find who reviewed this author: agent (i-1) % n reviewed agent i
            reviewer_idx = (i - 1) % n_agents
            feedback = review_results[reviewer_idx]
            reviewer_agent = participating_agents[reviewer_idx]
            reviewer_name = reviewer_agent.agent_id
            
            response_task = Task(
                task_id=f"{task.task_id}_response_{author.agent_id}",
                description=(
                    f"PEER REVIEW - Phase 3: Respond to Feedback and Revise\n\n"
                    f"Original task: {task.description}\n\n"
                    f"=== Your Original Work ===\n"
                    f"{initial_results[i].output[:self.context_truncation_limit]}\n\n"
                    f"=== Feedback from {reviewer_name} ===\n"
                    f"{feedback.output[:self.output_truncation_limit]}\n\n"
                    f"Provide:\n"
                    f"1. RESPONSE TO FEEDBACK: Address {reviewer_name}'s points (agree/disagree with reasons)\n"
                    f"2. REVISED WORK: Your improved output incorporating valid feedback\n"
                    f"3. CHANGES MADE: List specific changes you made based on the review"
                ),
                complexity=task.complexity,
                context={
                    **task.context,
                    "phase": "revision",
                    "review_phase": 3,
                    "feedback_from": reviewer_name,
                    "original_work": initial_results[i].output[:self.context_truncation_limit],
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
        
        # Phase 4 (optional): Meta-review synthesis
        if include_meta_review and participating_agents:
            # Use the first agent (or a designated synthesizer) for meta-review
            synthesizer = participating_agents[0]
            
            all_revisions = "\n\n".join([
                f"=== {participating_agents[i].agent_id}'s Revised Work ===\n{r.output[:self.context_truncation_limit]}"
                for i, r in enumerate(response_results) if r.success
            ])
            
            meta_task = Task(
                task_id=f"{task.task_id}_meta_review",
                description=(
                    f"PEER REVIEW - Phase 4: Meta-Review Synthesis\n\n"
                    f"Original task: {task.description}\n\n"
                    f"All revised works after peer review:\n{all_revisions}\n\n"
                    f"Synthesize the best elements from all revisions into a final, cohesive output. "
                    f"Identify common themes, resolve contradictions, and produce the best possible result."
                ),
                complexity=task.complexity,
                context={
                    **task.context,
                    "phase": "meta_review",
                    "review_phase": 4,
                }
            )
            
            meta_result = await synthesizer.execute(meta_task)
            results.append(meta_result)
        
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
        """Send a message from one agent (or coordinator) to another."""
        # Allow messages from coordinator even if not an agent
        if receiver_id not in self.agents and receiver_id != "coordinator":
            return None
        
        message = Message(
            message_id=str(uuid4()),
            sender_id=sender_id,
            receiver_id=receiver_id,
            content=content,
            message_type=message_type
        )
        
        self.message_log.append(message)
        
        # Only deliver to actual agents
        if receiver_id in self.agents:
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
