# Week 01 Baseline

Status: scaffolded

## Environment
- Machine: HP OMEN-35L (x86-64)
- OS: Ubuntu 24.04.4 LTS
- Python:3.14.7
- Numpy: 2.5.4
- GPU: RTX 5060 TI (16GB), CUDA 13.2

## Commands I Ran
- `python -m pip install -e ".[dev]"`
- `pytest`
- `python examples/reference_bench.py`
- `make bootstrap-results`
- `make bench`

## Correctness Result

`pytest` passed.

## Reference Timings

`python examples/reference_bench.py` output the following:

```
vector_add: 0.000001s
matmul: 0.000009s
softmax: 0.000018s
attention: 0.000019s
```

## Reference Functions:
- `vector_add`: adds matching positions from two arrays.
- `softmax`: converts a row of scores into normalized probabilities.

## My CPU vs GPU Mental Model
A CPU is built for flexible control and fast response to individual work. It is designed for serial processing and complex decision making, featuring a small number of powerful, versatile cores that handles tasks one after another with very low latency (quick response for a single task).

A GPU is good at doing a huge number of similar operations over large arrays of data. It has high throughput i.e. maximum volume of data processed.

## Test vs Benchmark

- What does `pytest` prove?
    - `pytest` runs a check to test if for a particular set of inputs to a function matches the expected output.

- What does `python examples/reference_bench.py` measure?
    - It runs four functions - `vector_add`, `matmul`, `softmax`, `attention` for `500`, `50`, `100` and `20` times respectively for a set of inputs and calculates the output and finally prints the average time taken by each function per run. \
    The following values were obtained:\
    ```
    | Kernel     | Seconds / run |
    | :---: | :---: |
    | vector_add | 0.000001 |
    | matmul     | 0.000010 |
    | softmax    | 0.000018 |
    | attention  | 0.000019 |
    ```
## Write Down

- Why do baselines come before kernels?
    - A baseline is basically the standard we need to set so we can compare our kernels. 
- What did the reference benchmark teach you?
    - The reference benchmarks tests execution times of different functions and ran them couple of times because execution time fluctuates due to background processes.
- What would you compare once a real kernel exists?
    - I'd compare the speed of real kernel against the existing numpy based implementation and triton based implementation.
- What is still fuzzy about GPU execution?
    - Everything seemed easy to me as I had some previous experience.
