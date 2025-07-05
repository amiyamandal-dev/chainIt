# ⚡️ High‑Performance Pipeline Framework

> **Turn plain Python functions into production‑ready, observable pipelines in minutes.**

[![PyPI](https://img.shields.io/pypi/v/pipeline-framework.svg)](https://pypi.org/project/pipeline-framework/) 
![Python](https://img.shields.io/pypi/pyversions/pipeline-framework.svg) 
[![License](https://img.shields.io/github/license/your-org/pipeline-framework.svg)](LICENSE)

---

## ✨ Why choose this library?

| What you get                      | How it helps                                                         |       |                                |
| --------------------------------- | -------------------------------------------------------------------- | ----- | ------------------------------ |
| **Zero‑boilerplate DSL**          | Compose with \`step1                                                 | step2 | step3\` – no classes required. |
| **Sync ✓  Async ✓**               | Same codebase for notebooks **and** production services.             |       |                                |
| **Built‑in performance knobs**    | Numba JIT, NumPy vectorization, CFFI, PyO3/Rust, batching & pools.   |       |                                |
| **Reliability patterns**          | Automatic retry, exponential back‑off, circuit‑breaker.              |       |                                |
| **Fan‑out / Fan‑in & Map‑Reduce** | Express complex graphs without 3rd‑party DAG engines.                |       |                                |
| **Rich telemetry**                | Per‑step timings and full execution history for debugging/profiling. |       |                                |

> **Ideal for** ETL jobs, scientific data‑flows, ML preprocessing, microservice glue code, or any place you’d otherwise write ad‑hoc `for` loops.

---

## 🔧 Installation

```bash
pip install pipeline-framework  # core (NumPy dependency only)

# Optional performance extras
pip install "pipeline-framework[numba,cffi]"  # JIT + CFFI loops
# PyO3 requires Rust and cargo in PATH (see docs)
```

> The library autodetects extras at runtime and degrades gracefully when they’re missing.

---

## 🚀 Quick Start

```python
from pipeline import piped, PIPE, Pipeline

@piped
def add_one(x):
    return x + 1

@piped(batch_size=4, parallel='thread')
def square(nums):
    return [n * n for n in nums]

pipeline = add_one | square
print(pipeline.run([1, 2, 3, 4]))  # 👉 [4, 9, 16, 25]
```

Need async? Same code:

```python
import asyncio
asyncio.run(pipeline.async_run([10, 11, 12]))
```

---

## 🧩 Core Concepts

### 1. `PipeStep`

Lightweight wrapper that stores **how** a function should run (batch size, pool, retry policy, etc.). Create steps with the `@piped` decorator.

### 2. `Pipeline`

Immutable sequence of `PipeStep`s. Use the `|` operator or the fluent `PipelineBuilder`.

### 3. Special Steps

* **`FanOutStep`** – broadcast input to multiple branches (optionally in parallel).
* **`FanInStep`**  – merge branch outputs with a custom combiner.
* **`MapReduceStep`** – convenience for batched map → reduce patterns.

---

## 🏎️ Performance Toggles

```python
@piped(jit=True)                      # Numba                                           
@piped(vectorize=True)                # NumPy/Numba ufuncs                              
@piped(parallel='process')            # Process pool (auto‑picklable check)            
@piped(batch_size=1024)               # Batch incoming iterables                        
@piped(cffi=True)                     # Compile simple loops to C                       
@piped(pyo3=True)                     # Compile numeric loop to Rust via PyO3           
```

Choose any combination – conflicting options warn and fall back to safe defaults.

---

## 🛡️ Reliability

```python
from pipeline import retry, circuit_breaker

step = retry(max_attempts=5, delay=0.5, backoff=2)(step)
step = circuit_breaker(threshold=3, timeout=60)(step)
```

Both decorators work for sync and async code. On repeated failures the circuit opens and short‑circuits upstream calls until the timeout elapses.

---

## 🌐 Async API Example (Threaded I/O + Process CPU)

The framework plays nicely with `asyncio` — even when parts of your workload are blocking. Below is a **complete recipe** that:

1. **Fetches JSON** from three public HTTP endpoints concurrently (thread pool, I/O‑bound).
2. **Parses & summarises** the payload in a separate **process pool** (CPU‑bound).
3. Streams the combined result back to the main coroutine.

```python
import asyncio, aiohttp, json
from pipeline import piped, PIPE, FanOutStep, FanInStep

# ── 1. I/O‑bound: threaded HTTP fetch ─────────────────────────────┐
@piped(parallel='thread')
async def fetch_json(url: str) -> dict:
    async with aiohttp.ClientSession() as sess:
        async with sess.get(url, timeout=10) as resp:
            return await resp.json()
# ───────────────────────────────────────────────────────────────────┘

# ── 2. CPU‑bound: process‑pool parsing ─────────────────────────────┐
@piped(parallel='process')
def summarise(data: dict) -> dict:
    # pretend this is heavy crunching
    title  = data.get('title', '')[:50]
    length = len(json.dumps(data))
    return {'title': title, 'bytes': length}
# ───────────────────────────────────────────────────────────────────┘

# Build three identical fetch→summarise branches
BRANCH = fetch_json | summarise
urls = [
    'https://jsonplaceholder.typicode.com/posts/1',
    'https://jsonplaceholder.typicode.com/posts/2',
    'https://jsonplaceholder.typicode.com/posts/3',
]
branches = tuple(BRANCH(url) for url in urls)  # partial‑apply URL

pipeline = (
    FanOutStep(branches) |                 # run 3 branches in parallel threads
    FanInStep(lambda *results: list(results))
)

async def main():
    summaries = await pipeline.async_run()
    for s in summaries:
        print(s)

asyncio.run(main())
```

**What to notice**

| Aspect               | Detail                                                                                                           |
| -------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **Thread pool**      | `parallel='thread'` lets blocking `aiohttp` calls coexist with the event‑loop without freezing it.               |
| **Process pool**     | `parallel='process'` offloads heavy CPU work; each summary happens in a separate OS process.                     |
| **Fan‑out / Fan‑in** | `FanOutStep` rockets the same seed into multiple branches; `FanInStep` stitches their outputs together in order. |
| **Error handling**   | Both steps inherit retry/circuit‑breaker wrappers automatically if you add them.                                 |

---

## 🕸 Advanced Usage

### Fan‑out / Fan‑in Example

```python
branch_a = piped(lambda x: x ** 2)
branch_b = piped(lambda x: x ** 3)

combined = (
    FanOutStep((branch_a, branch_b), parallel='thread') |
    FanInStep(lambda sq, cube: sq + cube)
)

print(combined.run(3))  # 👉 36
```

### Map‑Reduce Example

```python
mapper  = piped(lambda x: x * x)
reducer = lambda xs: sum(xs)

square_sum = MapReduceStep(mapper, reducer, batch_size=256)
print(square_sum.run(range(1_000_000)))
```

### Cancellation & Timeouts

```python
with pipeline as p:          # pools auto‑cleaned on exit
    try:
        p.cancel()           # from another thread/task
    except asyncio.CancelledError:
        ...                  # handle cleanly
```

Each `PipeStep` can also specify a `timeout=...` (seconds).

---

## 📚 API Cheatsheet

```text
piped(...):               → PipeStep              # decorator
retry(...):               PipeStep → PipeStep     # wrapper
circuit_breaker(...):     PipeStep → PipeStep

Pipeline.run(seed=None)           → result
Pipeline.async_run(seed=None)     → awaitable
Pipeline.run_detailed(seed=None)  → ExecutionResult(value, history, dt, n)

FanOutStep/ FanInStep / MapReduceStep  – graph helpers
cleanup_pools()                     – free all executors
```

Full doc‑strings are available in the source; type hints are 100 %.

---

## 💡 Best Practices

* Keep steps **pure** and side‑effect‑free whenever possible.
* Use `parallel='process'` only for CPU‑bound, pickle‑friendly work.
* Prefer **batching** for I/O‑bound tasks (database, network).
* Chain retries **before** circuit‑breakers: `retry(...)` → `circuit_breaker(...)`.
* Call `cleanup_pools()` or use a `with` block when embedding in long‑lived services.

---

## 🛠 Troubleshooting & FAQ

| Symptom                      | Fix                                                           |
| ---------------------------- | ------------------------------------------------------------- |
| Hanging on Windows           | Ensure `parallel='process'` uses the default *spawn* context. |
| `CircuitBreakerError` raised | Wait `timeout` seconds or reset by restarting the pipeline.   |
| `RetryExhaustedError` raised | Increase `max_attempts` or inspect the underlying error.      |
| Function not pickleable      | Switch to `parallel='thread'` or make the function top‑level  |

---

## 🤝 Contributing

1. **Fork** → **create feature branch** → **commit** → **PR**.
2. Run `pre-commit install` to auto‑format with **ruff** & **black**.
3. All features need unit tests (`pytest`).

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

---

### Acknowledgements

Inspired by ideas from Luigi, Apache Beam, Prefect, and countless Reddit threads on "why is my pipeline so slow?" 🙃
