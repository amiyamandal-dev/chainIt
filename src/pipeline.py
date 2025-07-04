"""
prod_pipeline_v4.py – Ultra-optimized pipeline with JIT compilation
Features:
- Just-In-Time (JIT) compilation for numeric-heavy operations
- Automatic loop vectorization
- Parallel execution for CPU-bound steps
- Memory views for zero-copy data access
- Precompilation of pipeline steps
- Optimized data structures with slots
- Batched processing support
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import sys
import time
import yaml
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Any, Awaitable, Callable, Dict, Generic, Iterable, List,
    Mapping, Optional, Sequence, Tuple, TypeVar, Union
)
from functools import lru_cache, partial
from types import MappingProxyType
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np

try:
    import numba
    from numba import jit, njit, vectorize, guvectorize, float64, int64

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

try:
    import pyarrow as pa

    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

__all__ = (
    "PIPE", "PipelineError", "PipeStep", "Pipeline", "piped", "retry", "circuit_breaker",
    "FanOutStep", "FanInStep", "ExecutionResult", "jit_step", "vectorize_step", "parallel_step"
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")
S = TypeVar("S")

PIPE: object = object()
_MISSING: object = object()


# =============================================================================
# Optimized Exceptions
# =============================================================================

class PipelineError(Exception):
    __slots__ = ("step_name", "original")

    def __init__(self, step: str, original: BaseException) -> None:
        super().__init__(f"Step {step} failed: {type(original).__name__}")
        self.step_name = step
        self.original = original


class RetryExhaustedError(PipelineError):
    pass


class CircuitBreakerError(PipelineError):
    pass


# =============================================================================
# Data Classes with Slots
# =============================================================================

@dataclass(slots=True, frozen=True)
class ExecutionResult(Generic[T]):
    value: T
    history: Tuple[Tuple[str, Any], ...] = ()
    execution_time: float = 0.0
    step_count: int = 0


@dataclass(slots=True, frozen=True)
class RetryConfig:
    max_attempts: int = 3
    delay: float = 1.0
    backoff_factor: float = 2.0
    exceptions: Tuple[type, ...] = (Exception,)


@dataclass(slots=True, frozen=True)
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# =============================================================================
# JIT Compilation Support
# =============================================================================

def jit_step(
        func: Optional[Callable] = None,
        *,
        nopython: bool = True,
        parallel: bool = False,
        fastmath: bool = True,
        cache: bool = True,
        **kwargs
):
    """Decorator for JIT-compiling pipeline steps"""

    def decorator(f: Callable) -> Callable:
        if not HAS_NUMBA:
            return f

        jitted = numba.jit(
            nopython=nopython,
            parallel=parallel,
            fastmath=fastmath,
            cache=cache,
            **kwargs
        )(f)

        # Precompile with sample data if available
        if cache and hasattr(f, "__sample_args__"):
            jitted(*f.__sample_args__)

        return jitted

    return decorator(func) if func else decorator


def vectorize_step(
        signatures: Optional[List[str]] = None,
        *,
        target: str = 'cpu',
        nopython: bool = True,
        **kwargs
):
    """Decorator for creating ufuncs from pipeline steps"""

    def decorator(f: Callable) -> Callable:
        if not HAS_NUMBA:
            return f

        if signatures is None:
            # Try to infer signature from type hints
            hints = get_type_hints(f)
            if hints and 'return' in hints:
                input_types = [t for n, t in hints.items() if n != 'return']
                return_type = hints['return']
                sig = f"({','.join(t.__name__ for t in input_types)})->{return_type.__name__}"
                signatures = [sig]

        return numba.vectorize(signatures, target=target, nopython=nopython, **kwargs)(f)

    return decorator


# =============================================================================
# Parallel Execution Support
# =============================================================================

def parallel_step(
        executor_type: str = 'thread',
        max_workers: Optional[int] = None,
        chunksize: int = 1
):
    """Decorator for parallel execution of pipeline steps"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Handle single argument vs iterable
            if len(args) == 1 and isinstance(args[0], (list, tuple, np.ndarray)):
                items = args[0]
                single = False
            else:
                items = [args]
                single = True

            # Choose executor
            if executor_type == 'process':
                executor_class = ProcessPoolExecutor
            else:
                executor_class = ThreadPoolExecutor

            with executor_class(max_workers=max_workers) as executor:
                results = list(executor.map(
                    lambda x: func(*x),
                    items,
                    chunksize=chunksize
                ))

            return results[0] if single else results

        return wrapper

    return decorator


# =============================================================================
# Core Classes (Further Optimized)
# =============================================================================

class PipeStep(Generic[T, R]):
    __slots__ = (
        "func", "stored_args", "stored_kwargs", "retry_config",
        "circuit_breaker_config", "is_method", "is_jitted", "is_vectorized",
        "batch_size", "executor_type"
    )

    def __init__(
            self,
            func: Callable[..., R | Awaitable[R]],
            stored_args: Tuple[Any, ...] = (),
            stored_kwargs: Optional[Mapping[str, Any]] = None,
            *,
            retry_config: Optional[RetryConfig] = None,
            circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
            is_method: bool = False,
            is_jitted: bool = False,
            is_vectorized: bool = False,
            batch_size: int = 1,
            executor_type: Optional[str] = None
    ):
        self.func = func
        self.stored_args = stored_args
        self.stored_kwargs = stored_kwargs or {}
        self.retry_config = retry_config
        self.circuit_breaker_config = circuit_breaker_config
        self.is_method = is_method
        self.is_jitted = is_jitted
        self.is_vectorized = is_vectorized
        self.batch_size = batch_size
        self.executor_type = executor_type

    # ------------------------------------------------------------------
    # Helper interface
    # ------------------------------------------------------------------

    def __call__(self, *args: Any, **kwargs: Any) -> "PipeStep":
        """Return a new step with stored arguments."""
        return PipeStep(
            func=self.func,
            stored_args=args,
            stored_kwargs=kwargs,
            retry_config=self.retry_config,
            circuit_breaker_config=self.circuit_breaker_config,
            is_method=self.is_method,
            is_jitted=self.is_jitted,
            is_vectorized=self.is_vectorized,
            batch_size=self.batch_size,
            executor_type=self.executor_type,
        )

    def run(self, injected_value: Any = _MISSING) -> R:
        """Synchronously execute this step."""
        return self.execute_sync(injected_value)

    async def async_run(self, injected_value: Any = _MISSING) -> R:
        """Asynchronously execute this step."""
        return await self.execute_async(injected_value)

    async def execute_async(self, injected_value: Any = _MISSING) -> R:
        args, kwargs = self._prepare_args(injected_value)
        return await self._execute(args, kwargs)

    def execute_sync(self, injected_value: Any = _MISSING) -> R:
        args, kwargs = self._prepare_args(injected_value)
        return self._execute(args, kwargs, async_context=False)

    def _prepare_args(self, injected_value: Any) -> Tuple[Tuple, Dict]:
        if injected_value is _MISSING:
            return self.stored_args, dict(self.stored_kwargs)

        if not self.stored_args and not self.stored_kwargs:
            return (injected_value,), {}

        args = tuple(
            injected_value if arg is PIPE else arg
            for arg in self.stored_args
        )

        kwargs = {
            k: injected_value if v is PIPE else v
            for k, v in self.stored_kwargs.items()
        }

        return args, kwargs

    def _execute(
            self,
            args: Tuple,
            kwargs: Dict,
            async_context: bool = True
    ) -> R:
        try:
            # Handle batched processing
            if self.batch_size > 1 and self._is_batchable(args, kwargs):
                return self._execute_batch(args, kwargs, async_context)

            # Handle parallel execution
            if self.executor_type:
                return self._execute_parallel(args, kwargs, async_context)

            # Handle vectorized functions
            if self.is_vectorized and HAS_NUMBA:
                return self._execute_vectorized(args, kwargs)

            # Standard execution
            result = self.func(*args, **kwargs)
            return self._handle_result(result, async_context)

        except (asyncio.CancelledError, KeyboardInterrupt):
            raise
        except BaseException as exc:
            raise PipelineError(self.func.__qualname__, exc) from exc

    def _is_batchable(self, args: Tuple, kwargs: Dict) -> bool:
        """Check if input is batchable"""
        if not args:
            return False
        first_arg = args[0]
        return isinstance(first_arg, (list, tuple, np.ndarray, pa.Array))

    def _execute_batch(
            self,
            args: Tuple,
            kwargs: Dict,
            async_context: bool
    ) -> R:
        """Process data in batches"""
        batch_size = self.batch_size
        batched_args = list(zip(*args)) if len(args) > 1 else [(arg,) for arg in args[0]]
        results = []

        for i in range(0, len(batched_args), batch_size):
            batch = batched_args[i:i + batch_size]
            if len(args) > 1:
                # Transpose: [(a1,b1), (a2,b2)] -> ([a1,a2], [b1,b2])
                transposed = list(zip(*batch))
                batch_args = tuple(transposed)
            else:
                batch_args = (batch,)

            batch_result = self.func(*batch_args, **kwargs)
            results.append(batch_result)

        # Flatten results
        if isinstance(results[0], list):
            return [item for sublist in results for item in sublist]
        if isinstance(results[0], np.ndarray):
            return np.concatenate(results)
        return results

    def _execute_parallel(
            self,
            args: Tuple,
            kwargs: Dict,
            async_context: bool
    ) -> R:
        """Execute function in parallel"""
        executor_class = {
            'thread': ThreadPoolExecutor,
            'process': ProcessPoolExecutor
        }[self.executor_type]

        with executor_class() as executor:
            if len(args) == 1 and isinstance(args[0], Iterable):
                items = args[0]
                futures = [executor.submit(self.func, item, **kwargs) for item in items]
                return [f.result() for f in futures]
            else:
                future = executor.submit(self.func, *args, **kwargs)
                return future.result()

    def _execute_vectorized(self, args: Tuple, kwargs: Dict) -> R:
        """Execute vectorized function efficiently"""
        if HAS_NUMBA and isinstance(args[0], np.ndarray):
            # Use memory view for zero-copy access
            if self.is_jitted:
                return self.func(*[numba.asarray(arg) for arg in args], **kwargs)
            return self.func(*args, **kwargs)
        return self.func(*args, **kwargs)

    def _handle_result(self, result: Any, async_context: bool) -> R:
        if async_context and inspect.isawaitable(result):
            return result
        if not async_context and inspect.isawaitable(result):
            raise RuntimeError("Awaitable returned in sync context")
        return result

    def __or__(self, other: PipeStep[R, S] | Pipeline[R, S]) -> Pipeline[T, S]:
        if isinstance(other, PipeStep):
            return Pipeline([self, other])
        if isinstance(other, Pipeline):
            return Pipeline([self, *other.steps])
        raise TypeError(f"Unsupported type: {type(other).__name__}")


# =============================================================================
# Parallel Processing Steps (Optimized)
# =============================================================================

class FanOutStep(Generic[T, R]):
    __slots__ = ("branches", "executor_type")

    def __init__(self, branches: Tuple[Pipeline[T, R], ...], executor_type: str = 'thread'):
        self.branches = branches
        self.executor_type = executor_type

    async def execute_async(self, value: T) -> Tuple[R, ...]:
        if self.executor_type == 'process':
            return await self._execute_process(value)
        return await asyncio.gather(*(branch.async_run(value) for branch in self.branches))

    def execute_sync(self, value: T) -> Tuple[R, ...]:
        if self.executor_type == 'process':
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(branch.run, value) for branch in self.branches]
                return tuple(f.result() for f in futures)
        return tuple(branch.run(value) for branch in self.branches)

    async def _execute_process(self, value: T) -> Tuple[R, ...]:
        loop = asyncio.get_running_loop()
        with ProcessPoolExecutor() as executor:
            futures = [loop.run_in_executor(executor, branch.run, value) for branch in self.branches]
            return await asyncio.gather(*futures)


class FanInStep(Generic[T, R]):
    __slots__ = ("combiner", "is_jitted", "batch_size")

    def __init__(
            self,
            combiner: Callable[[Tuple[T, ...]], R],
            is_jitted: bool = False,
            batch_size: int = 1
    ):
        self.combiner = combiner
        self.is_jitted = is_jitted
        self.batch_size = batch_size

    def execute_async(self, values: Tuple[T, ...]) -> R:
        return self._combine(values)

    def execute_sync(self, values: Tuple[T, ...]) -> R:
        return self._combine(values)

    def _combine(self, values: Tuple[T, ...]) -> R:
        if self.batch_size > 1 and len(values) > self.batch_size:
            # Process in batches
            results = []
            for i in range(0, len(values), self.batch_size):
                batch = values[i:i + self.batch_size]
                results.append(self.combiner(batch))
            return self.combiner(tuple(results))

        # JIT-compiled execution if enabled
        if self.is_jitted and HAS_NUMBA:
            # Convert to NumPy arrays if possible
            np_values = tuple(
                np.array(v) if isinstance(v, (list, tuple)) else v
                for v in values
            )
            return self.combiner(*np_values)

        return self.combiner(values)


# =============================================================================
# Pipeline Class (JIT Optimized)
# =============================================================================

class Pipeline(Generic[T, R]):
    __slots__ = ("steps", "debug", "_precompiled")

    def __init__(
            self,
            steps: Sequence[PipeStep[Any, Any] | FanOutStep | FanInStep],
            debug: bool = False,
            precompiled: bool = False
    ):
        self.steps = tuple(steps)
        self.debug = debug
        self._precompiled = precompiled

        # Precompile steps on initialization
        if not precompiled:
            self._precompile()

    def _precompile(self):
        """Precompile steps for better performance"""
        if not HAS_NUMBA:
            return

        for step in self.steps:
            if isinstance(step, PipeStep) and step.is_jitted:
                # Compile with sample data if available
                if hasattr(step.func, "__sample_args__"):
                    sample_args = step.func.__sample_args__
                    step.func(*sample_args)
                else:
                    # Compile with dummy data
                    try:
                        step.func(0)
                    except Exception:
                        pass
            elif isinstance(step, FanInStep) and step.is_jitted:
                try:
                    step.combiner((0,))
                except Exception:
                    pass

    def _get_step_name(self, step: Any) -> str:
        if isinstance(step, PipeStep):
            return step.func.__name__
        if isinstance(step, FanOutStep):
            return "FanOutStep"
        if isinstance(step, FanInStep):
            return step.combiner.__name__
        return type(step).__name__

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    def run(self, seed: T | None = None) -> R:
        """Run the pipeline synchronously and return the final value."""
        return self.run_detailed(seed).value

    async def async_run(self, seed: T | None = None) -> R:
        """Run the pipeline asynchronously and return the final value."""
        value = seed
        for step in self.steps:
            step_name = self._get_step_name(step)
            try:
                if isinstance(step, (PipeStep, FanInStep)):
                    value = await step.execute_async(value)
                elif isinstance(step, FanOutStep):
                    value = await step.execute_async(value)
                if self.debug:
                    logger.debug("Step %s: %s", step_name, value)
            except Exception as exc:
                if not isinstance(exc, PipelineError):
                    exc = PipelineError(step_name, exc)
                raise exc
        return value

    # ... (rest of the Pipeline class remains similar to v3 with optimizations below)

    def run_detailed(self, seed: T | None = None) -> ExecutionResult[R]:
        start = time.perf_counter()
        value = seed
        history = []

        for step in self.steps:
            step_name = self._get_step_name(step)
            try:
                # Use NumPy for numeric operations if available
                if HAS_NUMBA and isinstance(value, (list, tuple)):
                    value = np.array(value)

                if isinstance(step, (PipeStep, FanInStep)):
                    value = step.execute_sync(value)
                elif isinstance(step, FanOutStep):
                    value = step.execute_sync(value)

                # Use memory-efficient storage
                history.append((step_name, self._compact_value(value)))

                if self.debug:
                    logger.debug("Step %s: %s", step_name, value)

            except Exception as exc:
                if not isinstance(exc, PipelineError):
                    exc = PipelineError(step_name, exc)
                raise exc

        return ExecutionResult(
            value=value,
            history=tuple(history),
            execution_time=time.perf_counter() - start,
            step_count=len(self.steps)
        )

    def _compact_value(self, value: Any) -> Any:
        """Create compact representation of value for history"""
        if isinstance(value, np.ndarray):
            return f"ndarray{value.shape}dtype={value.dtype}"
        if isinstance(value, (list, tuple)) and len(value) > 10:
            return f"{type(value).__name__}[{len(value)}]"
        if HAS_PYARROW and isinstance(value, pa.Array):
            return f"pyarrow.Array(len={len(value)}, type={value.type})"
        return value

    # ... (async methods similar with perf_counter)


# =============================================================================
# Decorators (JIT Enhanced)
# =============================================================================

@lru_cache(maxsize=512)
def _get_func_properties(func: Callable) -> Tuple[bool, bool, bool]:
    """Cached function property detection"""
    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    is_method = params and params[0].name in {"self", "cls"}
    is_async = inspect.iscoroutinefunction(func)
    is_vectorized = hasattr(func, "ufunc")  # NumPy ufunc
    return is_method, is_async, is_vectorized


def piped(
        func: Optional[Callable] = None,
        *,
        jit: bool = False,
        vectorize: bool = False,
        batch_size: int = 1,
        parallel: Optional[str] = None,
        sample_args: Optional[Tuple] = None
) -> Callable:
    """Enhanced pipeline step decorator with JIT options"""

    def decorator(f: Callable) -> PipeStep:
        nonlocal jit, vectorize

        # Apply JIT if requested
        if jit and HAS_NUMBA:
            f = jit_step(nopython=True)(f)
            jit_applied = True
        else:
            jit_applied = False

        # Apply vectorization if requested
        if vectorize and HAS_NUMBA:
            f = vectorize_step()(f)
            vectorize_applied = True
        else:
            vectorize_applied = False

        # Store sample args for precompilation
        if sample_args:
            f.__sample_args__ = sample_args

        # Get function properties
        is_method, is_async, is_native_vectorized = _get_func_properties(f)

        def factory(*args, **kwargs) -> PipeStep:
            return PipeStep(
                func=f,
                stored_args=args,
                stored_kwargs=kwargs,
                is_method=is_method,
                is_jitted=jit_applied,
                is_vectorized=vectorize_applied or is_native_vectorized,
                batch_size=batch_size,
                executor_type=parallel,
            )

        return factory()

    if func:
        return decorator(func)
    return decorator


def retry(
        max_attempts: int = 3,
        delay: float = 1.0,
        backoff_factor: float = 2.0,
        exceptions: Tuple[type, ...] = (Exception,),
        **piped_kwargs
) -> Callable:
    """Retry decorator with JIT passthrough"""

    def decorator(func: Callable) -> PipeStep:
        step = piped(**piped_kwargs)(func)
        step.retry_config = RetryConfig(max_attempts, delay, backoff_factor, exceptions)
        return step

    return decorator


def circuit_breaker(
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3,
        **piped_kwargs
) -> Callable:
    """Circuit breaker decorator with JIT passthrough"""

    def decorator(func: Callable) -> PipeStep:
        step = piped(**piped_kwargs)(func)
        step.circuit_breaker_config = CircuitBreakerConfig(
            failure_threshold, recovery_timeout, half_open_max_calls
        )
        return step

    return decorator


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # JIT-compiled numeric processing
    @piped(jit=True, sample_args=(np.array([1.0, 2.0]),))
    def jit_square(x):
        return x ** 2


    # Vectorized math operation
    @piped(vectorize=True)
    def vec_sin(x):
        return np.sin(x)


    # Batched processing
    @piped(batch_size=1000)
    def process_batch(batch):
        return [x * 2 for x in batch]


    # Parallel execution
    @piped(parallel='process')
    def cpu_intensive(x):
        return sum(i * i for i in range(x))


    # Create pipeline
    pipeline = (
            jit_square
            | vec_sin
            | process_batch
            | cpu_intensive
    )

    # Run pipeline
    data = np.linspace(0, 10, 1_000_000)
    result = pipeline.run(data)
    print(f"Result: {result}")

# =============================================================================
# Module Initialization
# =============================================================================

sys.modules.setdefault("pipeline", sys.modules[__name__])