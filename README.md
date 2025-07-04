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

### Advanced features

- **Type-driven builder** allows chaining steps with strong typing:

```python
from pipeline import PipelineBuilder

builder = PipelineBuilder()
builder.add(double)
pipeline = builder.build()
```

- **Context manager** cleans up thread pools automatically:

```python
with pipeline as p:
    result = p.run(3)
```

- **MapReduceStep** processes batches efficiently:

```python
from pipeline import MapReduceStep

step = MapReduceStep(mapper=lambda x: x * 2,
                     reducer=sum,
                     batch_size=2)
print(step.run([1, 2, 3, 4]))  # 20
```

- **Per-step timeout** safeguards long running functions:

```python
@piped(timeout=0.5)
def slow(x):
    time.sleep(1)
    return x

import pytest, time

with pytest.raises(Exception):
    slow.run(1)
```
