# chainIt

High performance utilities for building flexible function pipelines. Steps can
be decorated to enable JIT compilation with Numba, lightweight CFFI
optimisation or automatic batching and parallel execution.

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
