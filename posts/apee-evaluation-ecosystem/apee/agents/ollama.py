"""Ollama-powered LLM agent implementation."""

import time
from typing import Optional

import httpx

from apee.models import AgentRole, Task, AgentResult
from apee.agents.base import Agent


OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5-coder:3b"
DEFAULT_TIMEOUT = 60.0


class OllamaClient:
    """Async HTTP client for Ollama API."""
    
    def __init__(
        self, 
        base_url: str = OLLAMA_BASE_URL, 
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    
    async def generate(
        self, 
        prompt: str, 
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> dict:
        """Generate a response from Ollama."""
        client = await self._get_client()
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        response = await client.post(
            f"{self.base_url}/api/generate",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def check_health(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False
    
    async def list_models(self) -> list[str]:
        """List available Ollama models."""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/api/tags")
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class OllamaAgent(Agent):
    """
    LLM-powered agent using local Ollama inference.
    
    Supports various open-source models like Qwen, Llama, Mistral, etc.
    """
    
    def __init__(
        self,
        agent_id: str,
        role: AgentRole,
        model: str = DEFAULT_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        temperature: float = 0.7,
        max_tokens: int = 500,
        system_prompt: Optional[str] = None
    ):
        super().__init__(agent_id, role, system_prompt)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OllamaClient(base_url=base_url, model=model)
    
    async def execute(self, task: Task) -> AgentResult:
        """Execute a task using Ollama LLM inference."""
        start_time = time.time()
        
        try:
            # Build prompt with context
            prompt = self._build_prompt(task)
            
            # Call Ollama
            response = await self.client.generate(
                prompt=prompt,
                system=self.system_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            latency_ms = (time.time() - start_time) * 1000
            output = response.get("response", "")
            tokens = response.get("eval_count", 0) + response.get("prompt_eval_count", 0)
            
            # Evaluate response quality
            quality = self._evaluate_quality(output, task)
            
            # Record metrics
            self.metrics.record_success(latency_ms, quality, tokens)
            
            return AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                agent_role=self.role.value,
                output=output,
                quality_score=quality,
                latency_ms=latency_ms,
                tokens_used=tokens,
                success=True,
                metadata={"model": self.model}
            )
            
        except Exception as e:
            self.metrics.record_failure()
            return AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                agent_role=self.role.value,
                output="",
                quality_score=0.0,
                latency_ms=(time.time() - start_time) * 1000,
                tokens_used=0,
                success=False,
                error=str(e),
                metadata={"model": self.model}
            )
    
    def _build_prompt(self, task: Task) -> str:
        """Build the prompt from task and context."""
        parts = [f"Task: {task.description}"]
        
        if task.context:
            parts.append("\nContext:")
            for key, value in task.context.items():
                if isinstance(value, str) and len(value) > 500:
                    value = value[:500] + "..."
                parts.append(f"- {key}: {value}")
        
        # Include recent messages if available
        recent_messages = self.get_recent_messages(3)
        if recent_messages:
            parts.append("\nRelevant messages from other agents:")
            for msg in recent_messages:
                parts.append(f"- [{msg.sender_id}]: {msg.content[:200]}")
        
        parts.append("\nProvide your response:")
        return "\n".join(parts)
    
    def _evaluate_quality(self, output: str, task: Task) -> float:
        """
        Evaluate response quality using heuristics.
        
        This is a basic quality scorer - Phase 2 will implement
        more sophisticated evaluation methods.
        """
        if not output or not output.strip():
            return 0.0
        
        score = 0.5  # Base score
        
        # Length appropriateness
        word_count = len(output.split())
        if 20 <= word_count <= 400:
            score += 0.2
        elif word_count < 10:
            score -= 0.2
        elif word_count > 500:
            score -= 0.1
        
        # Structure indicators (bullet points, code blocks, numbered lists)
        structure_markers = ['- ', '* ', '1.', '2.', '```', '•']
        if any(marker in output for marker in structure_markers):
            score += 0.15
        
        # Task keyword relevance
        task_words = set(task.description.lower().split())
        output_words = set(output.lower().split())
        common_stopwords = {'the', 'a', 'an', 'is', 'are', 'to', 'and', 'or', 'of', 'in', 'for'}
        task_words -= common_stopwords
        overlap = len(task_words & output_words)
        relevance = overlap / max(len(task_words), 1)
        score += relevance * 0.15
        
        return max(0.0, min(1.0, score))
    
    async def close(self) -> None:
        """Clean up resources."""
        await self.client.close()
    
    def __repr__(self) -> str:
        return f"OllamaAgent(id={self.agent_id}, role={self.role.value}, model={self.model})"
