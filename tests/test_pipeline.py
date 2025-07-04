import pytest
import time
import numpy as np
from pipeline import (
    PIPE, Pipeline, piped, retry, circuit_breaker,
    FanOutStep, FanInStep, PipelineError, ExecutionResult
)


# =============================================================================
# Test Utilities
# =============================================================================

class MockException(Exception):
    pass


class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1
        return self.count


# =============================================================================
# Basic Pipeline Tests
# =============================================================================

def test_basic_pipeline():
    @piped
    def add_one(x):
        return x + 1

    @piped
    def double(x):
        return x * 2

    pipeline = add_one | double
    result = pipeline.run(5)
    assert result == 12  # (5+1)*2


def test_pipe_injection():
    @piped
    def add(a, b):
        return a + b

    result = add(3, PIPE).run(5)
    assert result == 8  # 5 + 3


def test_kwarg_injection():
    @piped
    def multiply(a, b):
        return a * b

    result = multiply(b=PIPE).run(5).run(3)
    assert result == 15  # 5*3


# =============================================================================
# Error Handling Tests
# =============================================================================

def test_error_propagation():
    @piped
    def fails(x):
        raise ValueError("Test error")

    pipeline = fails
    with pytest.raises(PipelineError) as exc_info:
        pipeline.run(5)
    assert "fails" in str(exc_info.value)
    assert "Test error" in str(exc_info.value)


def test_error_history():
    counter = Counter()

    @piped
    def step1(x):
        counter.increment()
        return x + 1

    @piped
    def step2(x):
        counter.increment()
        raise MockException("Failed")

    pipeline = step1 | step2
    with pytest.raises(PipelineError):
        pipeline.run(5)

    assert counter.count == 2  # Both steps executed before failure


# =============================================================================
# Retry Mechanism Tests
# =============================================================================

def test_retry_success():
    counter = Counter()

    @retry(max_attempts=3)
    @piped
    def flaky(x):
        counter.increment()
        if counter.count < 3:
            raise MockException("Flaky")
        return x * 2

    result = flaky.run(5)
    assert result == 10
    assert counter.count == 3


def test_retry_exhaustion():
    counter = Counter()

    @retry(max_attempts=3)
    @piped
    def always_fails(x):
        counter.increment()
        raise MockException("Always fails")

    with pytest.raises(PipelineError) as exc_info:
        always_fails.run(5)

    assert "RetryExhausted" in str(exc_info.value)
    assert counter.count == 3


# =============================================================================
# Circuit Breaker Tests
# =============================================================================

def test_circuit_breaker_trip():
    counter = Counter()

    @circuit_breaker(failure_threshold=2)
    @piped
    def faulty(x):
        counter.increment()
        raise MockException("Faulty")

    # First two failures trip the breaker
    with pytest.raises(PipelineError):
        faulty.run(1)
    with pytest.raises(PipelineError):
        faulty.run(2)

    # Third call should be blocked by circuit breaker
    with pytest.raises(CircuitBreakerError):
        faulty.run(3)

    assert counter.count == 2


def test_circuit_breaker_recovery():
    counter = Counter()

    @circuit_breaker(failure_threshold=2, recovery_timeout=0.1)
    @piped
    def sometimes_fails(x):
        counter.increment()
        if x % 2 == 0:
            raise MockException("Failed on even")
        return x * 2

    # Trip the breaker
    with pytest.raises(PipelineError):
        sometimes_fails.run(2)
    with pytest.raises(PipelineError):
        sometimes_fails.run(4)

    # Should be open
    with pytest.raises(CircuitBreakerError):
        sometimes_fails.run(6)

    # Wait for recovery
    time.sleep(0.15)

    # Should be half-open now
    result = sometimes_fails.run(3)  # Should succeed
    assert result == 6

    # Should be closed again
    result = sometimes_fails.run(5)
    assert result == 10


# =============================================================================
# CPU-Bound Operation Tests
# =============================================================================

def test_jit_compilation(monkeypatch):
    # Mock numba if not installed
    try:
        import numba
    except ImportError:
        numba = None
        monkeypatch.setattr("pipeline.HAS_NUMBA", False)

    @piped(jit=True)
    def sum_squares(n):
        total = 0
        for i in range(n):
            total += i ** 2
        return total

    # Test performance (should be faster than pure Python)
    import time
    n = 1000000

    start = time.perf_counter()
    py_result = sum_squares.func(n)  # Raw Python version
    py_time = time.perf_counter() - start

    start = time.perf_counter()
    jit_result = sum_squares.run(n)
    jit_time = time.perf_counter() - start

    assert jit_result == py_result
    if numba:
        assert jit_time < py_time / 2  # JIT should be significantly faster


def test_vectorized_operations():
    @piped(vectorize=True)
    def double(x):
        return x * 2

    data = np.array([1, 2, 3, 4])
    result = double.run(data)
    assert np.array_equal(result, np.array([2, 4, 6, 8]))


# =============================================================================
# Parallel Execution Tests
# =============================================================================

def test_thread_parallel():
    @piped(parallel='thread')
    def slow_square(x):
        time.sleep(0.01)
        return x * x

    numbers = list(range(10))

    start = time.perf_counter()
    results = slow_square.run(numbers)
    parallel_time = time.perf_counter() - start

    # Sequential execution would take ~0.1s, parallel should be faster
    assert results == [x * x for x in numbers]
    assert parallel_time < 0.05  # Should be much faster than 0.1s


def test_process_parallel():
    @piped(parallel='process')
    def cpu_intensive(x):
        # Simulate CPU-bound work
        return sum(i * i for i in range(x))

    numbers = list(range(1000, 1010))

    start = time.perf_counter()
    results = cpu_intensive.run(numbers)
    parallel_time = time.perf_counter() - start

    # Verify results
    assert results == [sum(i * i for i in range(n)) for n in numbers]

    # Sequential execution time for comparison
    start = time.perf_counter()
    [sum(i * i for i in range(n)) for n in numbers]
    seq_time = time.perf_counter() - start

    # Parallel should be faster for CPU-bound tasks
    assert parallel_time < seq_time * 0.8


# =============================================================================
# Batched Processing Tests
# =============================================================================

def test_batch_processing():
    counter = Counter()

    @piped(batch_size=3)
    def batch_sum(batch):
        counter.increment()
        return sum(batch)

    data = list(range(10))  # 10 items
    result = batch_sum.run(data)

    # Should be batched into ceil(10/3)=4 batches
    assert counter.count == 4
    assert result == sum(data)


# =============================================================================
# Fan-out/Fan-in Tests
# =============================================================================

def test_fan_out_fan_in():
    @piped
    def branch1(x):
        return x * 2

    @piped
    def branch2(x):
        return x + 3

    @piped
    def combine(a, b):
        return a + b

    fan_out = FanOutStep((branch1, branch2))
    fan_in = FanInStep(combine)

    pipeline = fan_out | fan_in
    result = pipeline.run(5)
    assert result == (5 * 2) + (5 + 3)  # 10 + 8 = 18


def test_async_fan_out():
    @piped
    async def async_branch(x):
        await asyncio.sleep(0.01)
        return x * 2

    fan_out = FanOutStep((async_branch, async_branch))
    pipeline = fan_out

    start = time.perf_counter()
    result = asyncio.run(pipeline.async_run(5))
    duration = time.perf_counter() - start

    assert result == (10, 10)
    assert duration < 0.02  # Should run in parallel


# =============================================================================
# Async Execution Tests
# =============================================================================

@pytest.mark.asyncio
async def test_async_pipeline():
    @piped
    async def async_add_one(x):
        await asyncio.sleep(0.01)
        return x + 1

    @piped
    async def async_double(x):
        await asyncio.sleep(0.01)
        return x * 2

    pipeline = async_add_one | async_double
    result = await pipeline.async_run(5)
    assert result == 12  # (5+1)*2


# =============================================================================
# Performance Tests
# =============================================================================

def test_large_data_throughput():
    @piped(jit=True, batch_size=10000)
    def process_chunk(chunk):
        return np.mean(chunk)

    # Generate 1M data points
    data = np.random.rand(1_000_000)

    start = time.perf_counter()
    result = process_chunk.run(data)
    duration = time.perf_counter() - start

    assert abs(result - np.mean(data)) < 1e-6
    assert duration < 0.1  # Should process quickly


def test_memory_efficiency(monkeypatch):
    # Monkeypatch to track memory usage
    import tracemalloc
    tracemalloc.start()

    @piped
    def memory_intensive(x):
        # Create a large temporary array
        temp = np.zeros((1000, 1000))
        return x + 1

    snapshot1 = tracemalloc.take_snapshot()
    result = memory_intensive.run(5)
    snapshot2 = tracemalloc.take_snapshot()

    # Calculate memory difference
    diff = snapshot2.compare_to(snapshot1, 'lineno')
    memory_increase = sum(stat.size_diff for stat in diff)

    assert result == 6
    assert memory_increase < 10_000_000  # Should be <10MB increase


# =============================================================================
# Edge Case Tests
# =============================================================================

def test_empty_pipeline():
    pipeline = Pipeline([])
    result = pipeline.run(5)
    assert result == 5


def test_single_step_pipeline():
    @piped
    def identity(x):
        return x

    pipeline = identity
    result = pipeline.run(5)
    assert result == 5


def test_none_handling():
    @piped
    def handle_none(x):
        return x is None

    result = handle_none.run(None)
    assert result is True


# =============================================================================
# Integration Tests
# =============================================================================

def test_full_integration():
    # Create a comprehensive pipeline with multiple features
    counter = Counter()

    @piped
    def fetch_data():
        return list(range(1, 6))

    @piped(parallel='thread')
    def process_item(x):
        time.sleep(0.01)
        return x * 2

    @circuit_breaker(failure_threshold=2)
    @retry(max_attempts=3)
    @piped
    def flaky_operation(x):
        counter.increment()
        if x == 4 and counter.count < 3:
            raise MockException("Flaky on 4")
        return x + 1

    @piped(batch_size=2)
    def batch_sum(items):
        return sum(items)

    # Build pipeline
    pipeline = (
            fetch_data
            | process_item  # Parallel processing
            | flaky_operation  # With retry and circuit breaker
            | batch_sum  # Batched processing
    )

    result = pipeline.run()
    assert result == (3 + 5 + 7 + 9 + 11)  # (1*2+1)=3, (2*2+1)=5, ...
    assert counter.count == 5 + 2  # 5 items + 2 retries


# =============================================================================
# Mock Tests for External Dependencies
# =============================================================================

def test_without_numba(monkeypatch):
    monkeypatch.setattr("pipeline.HAS_NUMBA", False)

    @piped(jit=True)
    def simple_func(x):
        return x + 1

    result = simple_func.run(5)
    assert result == 6  # Should still work without numba


def test_without_pyarrow(monkeypatch):
    monkeypatch.setattr("pipeline.HAS_PYARROW", False)

    @piped(batch_size=10)
    def batch_func(items):
        return len(items)

    result = batch_func.run(list(range(25)))
    assert result == 25  # Should still work without pyarrow


# =============================================================================
# Execution Result Tests
# =============================================================================

def test_execution_result():
    @piped
    def step1(x):
        return x + 1

    @piped
    def step2(x):
        return x * 2

    pipeline = step1 | step2
    result = pipeline.run_detailed(5)

    assert result.value == 12
    assert len(result.history) == 2
    assert result.history[0] == ("step1", 6)
    assert result.history[1] == ("step2", 12)
    assert result.execution_time > 0
    assert result.step_count == 2


# =============================================================================
# Declarative Pipeline Tests
# =============================================================================

def test_declarative_pipeline(tmp_path):
    # Create a YAML pipeline spec
    spec = """
    name: test_pipeline
    debug: true
    steps:
      - type: pipe
        function: step1
        args: [1]
      - type: pipe
        function: step2
        kwargs: {b: 2}
    """
    spec_file = tmp_path / "pipeline.yaml"
    spec_file.write_text(spec)

    # Mock step implementations
    def step1(a):
        return a + 1

    def step2(a, b):
        return a + b

    # Registry would normally map names to functions
    # For test, we'll monkeypatch the creation function
    def mock_create_pipe_step(spec):
        if spec["function"] == "step1":
            return PipeStep(step1, stored_args=(1,), stored_kwargs={})
        elif spec["function"] == "step2":
            return PipeStep(step2, stored_args=(), stored_kwargs={"b": 2})

    # Create and run pipeline
    pipeline = Pipeline.from_spec(spec_file)
    result = pipeline.run()

    # step1: 1 (arg) -> returns 1+1=2
    # step2: 2 (from previous) + 2 (kwarg) = 4
    assert result == 4


if __name__ == "__main__":
    pytest.main(["-v", "-s", "--durations=0"])