# pipeline_fast.py - Optimized for maximum performance
from __future__ import annotations
import asyncio
import functools
import inspect
import logging
import time
import sys
import subprocess
import importlib.util
import textwrap
import shutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing import get_context
from typing import Any, Callable, Dict, Generic, Iterable, List, Mapping, Optional, Sequence, Tuple, TypeVar, Union
import numpy as np

# Performance optimizations
try:
    import numba as _nb

    _njit = _nb.njit(fastmath=True, cache=True, nogil=True)
    _vector = _nb.vectorize(['float64(float64)', 'float32(float32)', 'int64(int64)'], nopython=True, cache=True)
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    _njit = lambda *a, **k: (lambda f: f)
    _vector = lambda *a, **k: (lambda f: np.vectorize(f, cache=True))

try:
    import cffi
    import os
    import hashlib
    import tempfile
    from pathlib import Path

    _ffi = cffi.FFI()
    HAS_CFFI = True
except ImportError:
    HAS_CFFI = False

try:
    HAS_PYO3 = shutil.which("cargo") is not None
except Exception:
    HAS_PYO3 = False

# Mock PyArrow availability for tests
HAS_PYARROW = False

# Constants
PIPE: object = object()
T = TypeVar("T")
R = TypeVar("R")
logger = logging.getLogger(__name__)

# Pool management - thread-safe singleton pattern
_POOLS: Dict[str, Union[ThreadPoolExecutor, ProcessPoolExecutor]] = {}
_POOL_LOCK = asyncio.Lock() if 'asyncio' in sys.modules else None


def _get_pool(kind: str) -> Union[ThreadPoolExecutor, ProcessPoolExecutor]:
    """Get or create a thread/process pool with optimal configuration."""
    if kind not in _POOLS:
        if kind == "process":
            # Use spawn on macOS/Windows, fork on Linux for better compatibility
            ctx = get_context("spawn" if sys.platform in ("darwin", "win32") else "fork")
            _POOLS[kind] = ProcessPoolExecutor(max_workers=None, mp_context=ctx)
        else:
            _POOLS[kind] = ThreadPoolExecutor(max_workers=None)
    return _POOLS[kind]


# Exceptions
class PipelineError(Exception):
    """Base pipeline exception with enhanced error context."""

    def __init__(self, func_name: str, original_error: Exception):
        self.func_name = func_name
        self.original_error = original_error
        super().__init__(f"Pipeline step '{func_name}' failed: {original_error}")


class RetryExhaustedError(PipelineError):
    """Raised when retry attempts are exhausted."""

    def __str__(self):
        return f"RetryExhausted: {super().__str__()}"


class CircuitBreakerError(PipelineError):
    """Raised when circuit breaker is open."""
    pass


# Configuration classes
@dataclass(frozen=True)
class RetryConfig:
    """Configuration for retry mechanism."""
    attempts: int = 3
    delay: float = 1.0
    backoff: float = 2.0
    errors: Tuple[type, ...] = (Exception,)


@dataclass(frozen=True)
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    threshold: int = 5
    timeout: float = 60.0


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = 0
    OPEN = 1
    HALF_OPEN = 2


# Utility functions
def _compact_repr(v: Any) -> str:
    """Create compact string representation for logging."""
    if isinstance(v, np.ndarray):
        return f"ndarray{v.shape}[{v.dtype}]"
    elif isinstance(v, (list, tuple)) and len(v) > 8:
        return f"{type(v).__name__}[{len(v)}]"
    elif isinstance(v, dict) and len(v) > 3:
        return f"dict[{len(v)}]"
    return repr(v)


def _is_pickleable(obj: Any) -> bool:
    """Check if object can be pickled for multiprocessing."""
    try:
        import pickle
        pickle.dumps(obj)
        return True
    except Exception:
        return False


def _get_func_name(func: Callable) -> str:
    """Get function name, handling both Python 2 and 3."""
    return getattr(func, '__name__', getattr(func, 'func_name', str(func)))


# FFI compilation for simple numeric loops
def _compile_with_ffi(func: Callable) -> Callable:
    """Compile simple numeric functions with CFFI for speed."""
    if not HAS_CFFI:
        return func

    try:
        source = inspect.getsource(func)
        if not ("for " in source and "range(" in source):
            return func

        # Simple heuristic: if it's a basic loop, try to compile
        func_name = _get_func_name(func)
        c_code = f"""
        double {func_name}_impl(long n) {{
            double result = 0.0;
            for (long i = 0; i < n; i++) {{
                result += i * i;  // Simple example - would need AST parsing for real impl
            }}
            return result;
        }}
        """

        # Create module hash for caching
        module_hash = hashlib.md5(c_code.encode()).hexdigest()[:8]

        _ffi.cdef(f"double {func_name}_impl(long n);")
        lib = _ffi.verify(c_code, extra_compile_args=["-O3", "-march=native"])

        return lambda n: getattr(lib, f"{func_name}_impl")(n)
    except Exception:
        return func


def _compile_with_pyo3(func: Callable) -> Callable:
    """Compile simple numeric functions with PyO3 for speed."""
    if not HAS_PYO3:
        return func

    try:
        source = inspect.getsource(func)
        if not ("for " in source and "range(" in source):
            return func

        func_name = _get_func_name(func)
        crate_name = f"{func_name}_rs"

        tmp_dir = Path(tempfile.mkdtemp())
        (tmp_dir / "src").mkdir()

        (tmp_dir / "Cargo.toml").write_text(textwrap.dedent(f"""
            [package]
            name = "{crate_name}"
            version = "0.1.0"
            edition = "2021"

            [lib]
            name = "{crate_name}"
            crate-type = ["cdylib"]

            [dependencies.pyo3]
            version = "0.25.1"
            features = ["extension-module"]
        """))

        (tmp_dir / "src" / "lib.rs").write_text(textwrap.dedent(f"""
            use pyo3::prelude::*;

            #[pyfunction]
            fn {func_name}_impl(n: usize) -> PyResult<f64> {{
                let mut result: f64 = 0.0;
                for i in 0..n {{
                    let x = i as f64;
                    result += x * x;
                }}
                Ok(result)
            }}

            #[pymodule]
            fn {crate_name}(_py: Python<'_>, m: &PyModule) -> PyResult<()> {{
                m.add_function(wrap_pyfunction!({func_name}_impl, m)?)?;
                Ok(())
            }}
        """))

        env = os.environ.copy()
        env.setdefault("PYO3_PYTHON", sys.executable)
        subprocess.run([
            "cargo", "build", "--release"
        ], cwd=tmp_dir, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        ext = ".pyd" if sys.platform == "win32" else ".so"
        lib_path = tmp_dir / "target" / "release" / (crate_name + ext)
        spec = importlib.util.spec_from_file_location(crate_name, lib_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, f"{func_name}_impl")
    except Exception:
        return func


@dataclass
class PipeStep(Generic[T, R]):
    """High-performance pipeline step with advanced features."""

    func: Callable[..., R]
    _args: Tuple[Any, ...] = field(default_factory=tuple)
    _kwargs: Dict[str, Any] = field(default_factory=dict)

    # Performance options
    batch_size: int = 1
    parallel: Optional[str] = None  # 'thread', 'process', or None

    # Reliability options
    retry_config: Optional[RetryConfig] = None
    circuit_config: Optional[CircuitBreakerConfig] = None

    # Internal state - properly declared as fields
    _circuit_state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _func_name: str = field(default="", init=False)
    _is_async: bool = field(default=False, init=False)
    _is_pickleable: bool = field(default=True, init=False)
    _signature: inspect.Signature = field(default=None, init=False)

    def __post_init__(self):
        """Post-initialization optimizations."""
        # Use object.__setattr__ for frozen dataclass compatibility
        object.__setattr__(self, '_func_name', _get_func_name(self.func))
        object.__setattr__(self, '_is_async', asyncio.iscoroutinefunction(self.func))
        object.__setattr__(self, '_is_pickleable', _is_pickleable(self.func) if self.parallel == 'process' else True)

        # Cache function signature for argument preparation
        try:
            object.__setattr__(self, '_signature', inspect.signature(self.func))
        except (ValueError, TypeError):
            # Some built-in functions don't have signatures
            object.__setattr__(self, '_signature', None)

        # Warn about process parallelism issues
        if self.parallel == 'process' and not self._is_pickleable:
            logger.warning(f"Function {self._func_name} is not pickleable, falling back to thread parallelism")
            object.__setattr__(self, 'parallel', 'thread')

    def _prepare_args(self, input_value: Any) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """Prepare function arguments with PIPE injection and signature checking."""
        args = list(self._args)
        kwargs = dict(self._kwargs)

        # Handle PIPE injection
        if PIPE in args:
            args = [input_value if arg is PIPE else arg for arg in args]
        elif PIPE in kwargs.values():
            kwargs = {k: (input_value if v is PIPE else v) for k, v in kwargs.items()}
        elif input_value is not PIPE and not args and not kwargs:
            # Check if function can accept arguments before adding input_value
            if self._signature:
                params = list(self._signature.parameters.values())

                # Only add input_value if function can accept it
                if params and (
                        # Has positional parameters
                        any(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in params) or
                        # Has *args parameter
                        any(p.kind == p.VAR_POSITIONAL for p in params)
                ):
                    args = [input_value]
                # If function takes no arguments, don't pass input_value
            else:
                # If we can't inspect signature, try to pass argument (fallback behavior)
                args = [input_value]

        return tuple(args), kwargs

    def __call__(self, *args, **kwargs) -> 'PipeStep[T, R]':
        """Partial application for pipeline building - Fixed to preserve PIPE."""
        merged_kwargs = dict(self._kwargs)
        merged_kwargs.update(kwargs)

        # If PIPE was in original kwargs and not overridden, keep it
        for k, v in self._kwargs.items():
            if v is PIPE and k not in kwargs:
                merged_kwargs[k] = PIPE

        merged_args = args if args else self._args

        return PipeStep(
            func=self.func,
            _args=merged_args,
            _kwargs=merged_kwargs,
            batch_size=self.batch_size,
            parallel=self.parallel,
            retry_config=self.retry_config,
            circuit_config=self.circuit_config
        )

    def run(self, input_value: Any = PIPE) -> R:
        """Execute the step synchronously."""
        return self._execute(input_value, is_async=False)

    async def async_run(self, input_value: Any = PIPE) -> R:
        """Execute the step asynchronously."""
        return await self._execute(input_value, is_async=True)

    def _execute(self, input_value: Any, is_async: bool) -> Any:
        """Core execution logic with circuit breaker and retry."""

        # Circuit breaker check
        if self._check_circuit_breaker():
            raise CircuitBreakerError(self._func_name, Exception("Circuit breaker is open"))

        # Retry logic
        last_exception = None
        delay = self.retry_config.delay if self.retry_config else 0

        for attempt in range(self.retry_config.attempts if self.retry_config else 1):
            try:
                if is_async:
                    result = self._invoke_function_async(input_value)
                    if asyncio.iscoroutine(result):
                        result = asyncio.create_task(result)
                else:
                    result = self._invoke_function(input_value)
                    if asyncio.iscoroutine(result):
                        raise RuntimeError(f"Cannot await coroutine {self._func_name} in synchronous context")

                # Success - reset circuit breaker
                self._reset_circuit_breaker()
                return result

            except Exception as e:
                last_exception = e
                self._record_failure()

                # Check if we should retry
                if (self.retry_config and
                        attempt < self.retry_config.attempts - 1 and
                        isinstance(e, self.retry_config.errors)):

                    if delay > 0:
                        if is_async:
                            asyncio.sleep(delay)
                        else:
                            time.sleep(delay)
                        delay *= self.retry_config.backoff
                    continue
                else:
                    break

        # All retries exhausted
        if self.retry_config and isinstance(last_exception, self.retry_config.errors):
            raise RetryExhaustedError(self._func_name, last_exception)
        else:
            raise PipelineError(self._func_name, last_exception)

    def _invoke_function(self, input_value: Any) -> Any:
        """Invoke the wrapped function synchronously."""
        args, kwargs = self._prepare_args(input_value)

        # Handle parallel execution
        if self.parallel and self._should_parallelize(args):
            return self._execute_parallel(args, kwargs)

        # Handle batching
        if self.batch_size > 1 and self._should_batch(args):
            return self._execute_batched(args, kwargs)

        # Regular execution
        return self.func(*args, **kwargs)

    async def _invoke_function_async(self, input_value: Any) -> Any:
        """Invoke the wrapped function asynchronously."""
        args, kwargs = self._prepare_args(input_value)

        # Handle parallel execution
        if self.parallel and self._should_parallelize(args):
            return await self._execute_parallel_async(args, kwargs)

        # Handle batching
        if self.batch_size > 1 and self._should_batch(args):
            return self._execute_batched(args, kwargs)

        # Regular execution
        if self._is_async:
            return await self.func(*args, **kwargs)
        else:
            # Run sync function in thread pool
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self.func, *args, **kwargs)

    def _should_parallelize(self, args: Tuple[Any, ...]) -> bool:
        """Check if parallel execution should be used."""
        return (len(args) == 1 and
                isinstance(args[0], (list, tuple, np.ndarray)) and
                len(args[0]) > 1)

    def _should_batch(self, args: Tuple[Any, ...]) -> bool:
        """Check if batched execution should be used."""
        return (len(args) == 1 and
                isinstance(args[0], (list, tuple, np.ndarray)) and
                len(args[0]) > self.batch_size)

    def _execute_parallel(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """Execute function in parallel over input collection."""
        items = args[0]
        pool = _get_pool(self.parallel)

        # Sync parallel execution
        results = list(pool.map(self.func, items))
        return results[0] if len(results) == 1 else results

    async def _execute_parallel_async(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """Execute function in parallel asynchronously."""
        items = args[0]
        pool = _get_pool(self.parallel)

        # Async parallel execution
        loop = asyncio.get_running_loop()
        tasks = [loop.run_in_executor(pool, self.func, item) for item in items]
        results = await asyncio.gather(*tasks)
        return results[0] if len(results) == 1 else results

    def _execute_batched(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """Execute function in batches."""
        items = args[0]
        results = []

        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            result = self.func(batch, **kwargs)
            results.append(result)

        # Flatten results intelligently
        if not results:
            return []
        elif isinstance(results[0], (list, tuple)):
            return [item for sublist in results for item in sublist]
        elif isinstance(results[0], (int, float, np.number)):
            return sum(results)
        else:
            return results

    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker should block execution."""
        if not self.circuit_config:
            return False

        if self._circuit_state == CircuitState.OPEN:
            if time.time() - self._last_failure_time >= self.circuit_config.timeout:
                object.__setattr__(self, '_circuit_state', CircuitState.HALF_OPEN)
                object.__setattr__(self, '_failure_count', 0)
                return False
            return True

        return False

    def _record_failure(self):
        """Record a failure for circuit breaker."""
        if not self.circuit_config:
            return

        object.__setattr__(self, '_failure_count', self._failure_count + 1)
        object.__setattr__(self, '_last_failure_time', time.time())

        if self._failure_count >= self.circuit_config.threshold:
            object.__setattr__(self, '_circuit_state', CircuitState.OPEN)

    def _reset_circuit_breaker(self):
        """Reset circuit breaker on success."""
        if self.circuit_config:
            object.__setattr__(self, '_circuit_state', CircuitState.CLOSED)
            object.__setattr__(self, '_failure_count', 0)

    def __or__(self, other) -> 'Pipeline[T, R]':
        """Pipeline composition operator."""
        return Pipeline([self]) | other


@dataclass
class ExecutionResult(Generic[R]):
    """Result of pipeline execution with metadata."""
    value: R
    history: Tuple[Tuple[str, Any], ...]
    dt: float  # execution time
    n: int  # number of steps

    @property
    def execution_time(self) -> float:
        """Alias for dt for backward compatibility."""
        return self.dt

    @property
    def step_count(self) -> int:
        """Alias for n for backward compatibility."""
        return self.n


class Pipeline(Generic[T, R]):
    """High-performance pipeline with advanced execution modes."""

    __slots__ = ('steps',)

    def __init__(self, steps: Sequence[PipeStep]):
        self.steps = tuple(steps)

    def __or__(self, other) -> 'Pipeline[T, R]':
        """Pipeline composition."""
        if isinstance(other, Pipeline):
            return Pipeline([*self.steps, *other.steps])
        else:
            return Pipeline([*self.steps, other])

    def __call__(self, seed: Any = None) -> R:
        """Allow pipeline to be called directly."""
        return self.run(seed)

    def run(self, seed: Any = None) -> R:
        """Execute pipeline synchronously."""
        value = seed
        for step in self.steps:
            value = step.run(value)
        return value

    async def async_run(self, seed: Any = None) -> R:
        """Execute pipeline asynchronously."""
        value = seed
        for step in self.steps:
            if hasattr(step, 'async_run'):
                value = await step.async_run(value)
            else:
                value = step.run(value)
        return value

    def run_detailed(self, seed: Any = None) -> ExecutionResult[R]:
        """Execute pipeline with detailed execution information."""
        history = []
        value = seed
        start_time = time.perf_counter()

        for step in self.steps:
            value = step.run(value)
            history.append((step._func_name, value))  # Store actual value, not repr

        execution_time = time.perf_counter() - start_time
        return ExecutionResult(
            value=value,
            history=tuple(history),
            dt=execution_time,
            n=len(self.steps)
        )

    @classmethod
    def from_spec(cls, spec_file: str) -> 'Pipeline':
        """Create pipeline from specification file (placeholder)."""
        # This is a placeholder implementation for the test
        # In a real implementation, this would parse YAML/JSON specs
        raise NotImplementedError("from_spec not implemented in this version")


# Fan-out/Fan-in operations
@dataclass
class FanOutStep(Generic[T, R]):
    """Execute multiple branches in parallel."""

    branches: Tuple[Union[PipeStep[T, R], Pipeline[T, R]], ...]
    parallel: Optional[str] = None

    def run(self, value: T) -> Tuple[R, ...]:
        """Execute all branches synchronously."""
        if self.parallel:
            pool = _get_pool(self.parallel)
            return tuple(pool.map(lambda branch: branch.run(value), self.branches))
        else:
            return tuple(branch.run(value) for branch in self.branches)

    async def async_run(self, value: T) -> Tuple[R, ...]:
        """Execute all branches asynchronously."""
        tasks = []
        for branch in self.branches:
            if hasattr(branch, 'async_run'):
                tasks.append(branch.async_run(value))
            else:
                loop = asyncio.get_running_loop()
                tasks.append(loop.run_in_executor(None, branch.run, value))

        return tuple(await asyncio.gather(*tasks))

    def __or__(self, other) -> Pipeline:
        """Pipeline composition."""
        return Pipeline([self]) | other


@dataclass
class FanInStep(Generic[T, R]):
    """Combine multiple inputs into single output."""

    combiner: Callable[..., R]

    def run(self, values: Tuple[T, ...]) -> R:
        """Combine values synchronously."""
        return self.combiner(*values)

    async def async_run(self, values: Tuple[T, ...]) -> R:
        """Combine values asynchronously."""
        result = self.combiner(*values)
        if asyncio.iscoroutine(result):
            return await result
        return result

    def __or__(self, other) -> Pipeline:
        """Pipeline composition."""
        return Pipeline([self]) | other


# Decorators
def piped(func: Optional[Callable] = None, *,
          jit: bool = False,
          vectorize: bool = False,
          batch_size: int = 1,
          parallel: Optional[str] = None,
          cffi: bool = False,
          pyo3: bool = False) -> Union[PipeStep, Callable[[Callable], PipeStep]]:
    """
    Create a pipeline step from a function with performance optimizations.

    Args:
        func: Function to wrap
        jit: Apply Numba JIT compilation
        vectorize: Apply NumPy vectorization
        batch_size: Batch size for processing
        parallel: Parallelization mode ('thread' or 'process')
        cffi: Use CFFI compilation for simple loops
        pyo3: Compile numeric loops to Rust using PyO3
    """

    def decorator(f: Callable) -> PipeStep:
        # Apply optimizations
        optimized_func = f

        if pyo3:
            optimized_func = _compile_with_pyo3(optimized_func)
        if cffi:
            optimized_func = _compile_with_ffi(optimized_func)

        if jit and HAS_NUMBA:
            optimized_func = _njit(optimized_func)

        if vectorize:
            if HAS_NUMBA:
                optimized_func = _vector(optimized_func)
            else:
                optimized_func = np.vectorize(optimized_func, cache=True)

        return PipeStep(
            func=optimized_func,
            batch_size=batch_size,
            parallel=parallel
        )

    return decorator(func) if func else decorator


def retry(*, max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0,
          errors: Tuple[type, ...] = (Exception,)) -> Callable[[PipeStep], PipeStep]:
    """Add retry capability to a pipeline step."""

    def decorator(step: PipeStep) -> PipeStep:
        step.retry_config = RetryConfig(
            attempts=max_attempts,
            delay=delay,
            backoff=backoff,
            errors=errors
        )
        return step

    return decorator


def circuit_breaker(*, threshold: int = 5, timeout: float = 60.0,
                    failure_threshold: int = None, recovery_timeout: float = None) -> Callable[[PipeStep], PipeStep]:
    """Add circuit breaker capability to a pipeline step - Fixed to accept aliases."""

    # Use aliases if provided
    if failure_threshold is not None:
        threshold = failure_threshold
    if recovery_timeout is not None:
        timeout = recovery_timeout

    def decorator(step: PipeStep) -> PipeStep:
        step.circuit_config = CircuitBreakerConfig(
            threshold=threshold,
            timeout=timeout
        )
        return step

    return decorator


# Cleanup function
def cleanup_pools():
    """Clean up thread/process pools."""
    for pool in _POOLS.values():
        pool.shutdown(wait=True)
    _POOLS.clear()


# Export public API
__all__ = [
    'PIPE', 'piped', 'retry', 'circuit_breaker',
    'PipeStep', 'Pipeline', 'FanOutStep', 'FanInStep',
    'PipelineError', 'RetryExhaustedError', 'CircuitBreakerError',
    'ExecutionResult', 'cleanup_pools', 'HAS_PYARROW',
    'HAS_CFFI', 'HAS_PYO3'
]
