# Fusion: GEMM + ReLU

Если после GEMM запустить отдельный ReLU, результат C сначала запишется в global
memory, а затем снова прочитается. Эпилог внутри того же ядра применяет ReLU к
`C_local` до store. Это классический operator fusion.

Важно различать две оптимизации:

- tiling переиспользует A/B и ускоряет основное вычисление;
- fusion убирает промежуточный global-memory round trip.

## Задание

Скопируйте рабочую структуру GEMM из предыдущего задания в
`build_matmul_relu`. После завершения K-loop, но до записи C, примените
`T.max(C_local[i, j], 0)` через `T.Parallel(block_M, block_N)`.

Сохраните Metal-совместимые параметры предыдущего задания: shared accumulator,
`num_stages=0`, тайлы `32 × 32 × 16` и `float32` output.

Не вызывайте отдельное PyTorch `relu` внутри решения.

## Материалы

- [официальный quickstart GEMM + ReLU](https://github.com/tile-ai/tilelang/blob/main/examples/quickstart.py)
- [Language Basics](https://tilelang.com/programming_guides/language_basics.html)

## Проверка

```bash
uv run pytest -vv test_task.py
```
