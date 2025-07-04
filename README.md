# chainIt

High performance utilities for building flexible function pipelines. Steps can
be decorated to enable JIT compilation with Numba, lightweight CFFI or PyO3
compilation, as well as automatic batching and parallel execution.

## Installation

```bash
pip install chainit
```

## Usage

```python
from pipeline import piped, Pipeline

@piped
def double(x):
    return x * 2

pipeline = double
print(pipeline.run(3))  # 6
```

### PyO3 acceleration

When Rust and `cargo` are available, numeric loops can be compiled using PyO3
by passing `pyo3=True` to the decorator:

```python
@piped(pyo3=True)
def loop_sum(n: int) -> float:
    total = 0.0
    for i in range(n):
        total += i * i
    return total

print(loop_sum.run(10))  # 285.0
```
