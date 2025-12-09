"""Helper utilities for APEE."""

import hashlib
import time
from typing import Any, Callable, TypeVar
from functools import wraps

T = TypeVar("T")


def generate_id(prefix: str = "") -> str:
    """Generate a unique ID with optional prefix."""
    timestamp = str(time.time_ns())
    hash_val = hashlib.md5(timestamp.encode()).hexdigest()[:8]
    if prefix:
        return f"{prefix}_{hash_val}"
    return hash_val


def timed(func: Callable[..., T]) -> Callable[..., tuple[T, float]]:
    """
    Decorator that returns both result and execution time in ms.
    
    Usage:
        @timed
        def my_function():
            ...
        result, elapsed_ms = my_function()
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> tuple[T, float]:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return result, elapsed_ms
    return wrapper


async def timed_async(coro: Any) -> tuple[Any, float]:
    """
    Time an async coroutine.
    
    Usage:
        result, elapsed_ms = await timed_async(some_async_call())
    """
    start = time.perf_counter()
    result = await coro
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text with ellipsis if too long."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def format_duration(ms: float) -> str:
    """Format milliseconds into human-readable duration."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    elif ms < 60000:
        return f"{ms/1000:.1f}s"
    else:
        minutes = int(ms // 60000)
        seconds = (ms % 60000) / 1000
        return f"{minutes}m {seconds:.0f}s"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that returns default on division by zero."""
    if denominator == 0:
        return default
    return numerator / denominator


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, calls_per_second: float = 10.0):
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0.0
    
    async def acquire(self) -> None:
        """Wait if necessary to respect rate limit."""
        import asyncio
        
        now = time.time()
        elapsed = now - self.last_call
        
        if elapsed < self.min_interval:
            await asyncio.sleep(self.min_interval - elapsed)
        
        self.last_call = time.time()


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay_ms: float = 100,
        max_delay_ms: float = 5000,
        exponential: bool = True,
    ):
        self.max_attempts = max_attempts
        self.base_delay_ms = base_delay_ms
        self.max_delay_ms = max_delay_ms
        self.exponential = exponential
    
    def get_delay(self, attempt: int) -> float:
        """Get delay in seconds for given attempt number."""
        if self.exponential:
            delay_ms = self.base_delay_ms * (2 ** attempt)
        else:
            delay_ms = self.base_delay_ms
        
        return min(delay_ms, self.max_delay_ms) / 1000
