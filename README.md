# High‑Performance Pipeline Framework

A flexible, **zero‑boilerplate** data‑processing pipeline library for Python 3.9+ that combines:

* ✨ *Declarative* pipeline construction (`step1 | step2 | step3`)
* 🚀 Runtime performance boosts (Numba JIT, NumPy vectorize, CFFI, PyO3/Rust)
* 🪄 Transparent sync **and** async execution
* 🧵 Thread / 🧩 process parallelism + batching
* ♻️ Automatic retry & circuit‑breaker patterns
* 🗂 Fan‑out / fan‑in and map‑reduce helpers
* ⏱ Per‑step timeouts and rich execution telemetry

If you need to turn a list of plain Python functions into a production‑ready, observable data pipeline **without reaching for heavy frameworks**, this repository is for you.

---

## Table of contents

* [Installation](#installation)
* [Quick start](#quick-start)
* [Pipeline primitives](#pipeline-primitives)
* [Performance optimisations](#performance-optimisations)
* [Reliability features](#reliability-features)
* [Advanced patterns](#advanced-patterns)
* [API reference](#api-reference)
* [Contributing](#contributing)
* [License](#license)

---

## Installation

```bash
# clone the repo
git clone https://github.com/<your‑org>/pipeline.git
cd pipeline

# create a virtual env (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# core dependencies
pip install -r requirements.txt
# or bare minimum
pip install numpy

# optional perf extras
pip install numba        # JIT
pip install cffi         # CFFI loop compilation
# for PyO3/Rust you need the Rust tool‑chain & cargo in PATH
```

> **Note**
> The library checks at runtime whether optional dependencies are present and
> falls back gracefully if they are not.

---

## Quick start

```python
from pipeline import piped, PIPE, Pipeline, retry, circuit_breaker

@piped
def step_add(x: int) -> int:
    return x + 1

@piped(batch_size=4, parallel='thread')
def step_square(nums):
    return [n * n for n in nums]

@piped(vectorize=True)
def step_sqrt(x: float) -> float:
    import math; return math.sqrt(x)

# reliability decorators
step_square = retry(max_attempts=3)(step_square)
step_sqrt   = circuit_breaker(threshold=5, timeout=30)(step_sqrt)

pipeline = step_add | step_square | step_sqrt

result = pipeline.run([1, 2, 3, 4])
print(result)            # -> [1.4142, 1.7320, 2.0, 2.2360]
```

### Async execution

```python
import asyncio
result = asyncio.run(pipeline.async_run([10, 11, 12, 13]))
```

---

## Pipeline primitives

| Primitive         | Purpose                                                          |            |
| ----------------- | ---------------------------------------------------------------- | ---------- |
| `PipeStep`        | Thin wrapper around a callable; exposes perf & reliability knobs |            |
| `Pipeline`        | Immutable sequence of `PipeStep`s (\`step1                       | step2 …\`) |
| `FanOutStep`      | Broadcasts a value to N branches in parallel                     |            |
| `FanInStep`       | Combines outputs from N branches with a custom combiner          |            |
| `MapReduceStep`   | Map‑phase (with batching) + reduce‑phase convenience             |            |
| `PipelineBuilder` | Type‑driven fluent builder (alternative to \`                    | \` pipe)   |

---

## Performance optimisations

| Technique     | How to enable                                | Speed‑up  |
| ------------- | -------------------------------------------- | --------- |
| **Numba JIT** | `@piped(jit=True)`                           | 2‑100×    |
| **Vectorize** | `@piped(vectorize=True)` (Numba or NumPy)    | 10‑30×    |
| **CFFI**      | `@piped(cffi=True)` for simple numeric loops | 5‑20×     |
| **PyO3/Rust** | `@piped(pyo3=True)` (requires Rust)          | 10‑50×    |
| **Parallel**  | `@piped(parallel='thread')` or `'process'`   | CPU bound |
| **Batching**  | `@piped(batch_size=1024)`                    | I/O bound |

All optimisations are **opt‑in** at decoration time; unsafe combinations are rejected at runtime (e.g. non‑pickleable func + process pool).

---

## Reliability features

### Retry

```python
step = retry(max_attempts=5, delay=0.5, backoff=2)(step)
```

### Circuit breaker

```python
step = circuit_breaker(threshold=3, timeout=60)(step)
```

Both wrap the underlying `PipeStep` and work in sync/async contexts.

---

## Advanced patterns

### Fan‑out / fan‑in

```python
branch1 = step_a | step_b
branch2 = step_c

joined = FanOutStep((branch1, branch2), parallel='thread') |
         FanInStep(lambda x, y: (x, y))
```

### Map‑reduce

```python
mapper  = piped(lambda x: x * x)
reducer = lambda xs: sum(xs)

square_sum = MapReduceStep(mapper, reducer, batch_size=100)
```

### Timeouts

Every `PipeStep` accepts a `timeout` parameter that terminates long‑running work.

---

## API reference

Full doc‑strings are available at runtime (`help(PipeStep)` etc.).  Key types:

* `piped(...)` – decorator returning a `PipeStep`
* `retry(...)` – adds exponential back‑off
* `circuit_breaker(...)`
* `Pipeline.run(seed)` / `Pipeline.async_run(seed)`
* `ExecutionResult` – returned by `Pipeline.run_detailed()`
* `cleanup_pools()` – free thread / process pools (automatically invoked on `with` block exit)

---

## Contributing

1. Fork the repo & create a feature branch
2. `pre‑commit install`
3. Add **tests** for any new behaviour (`pytest`)
4. Ensure `ruff` passes (`ruff --fix .`)
5. Submit a PR 🚀

---

