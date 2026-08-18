# Тайловое матричное умножение

GEMM показывает, зачем нужны тайлы. Один block считает тайл результата
`C[block_M, block_N]`. На каждом шаге по K он загружает два небольших тайла A и
B в shared memory и переиспользует их во множестве умножений. Для переносимости
на M4 аккумулятор `C_local` тоже хранится в shared scope. CUDA-ядра обычно
используют fragment/register accumulator, но это будет отдельной оптимизацией.

Схема одного K-шага:

```text
A global -> A_shared --\
                         T.gemm -> C_local
B global -> B_shared --/
```

## Задание

Реализуйте `build_matmul(M, N, K, block_M=32, block_N=32, block_K=16)` для
`float16` входов и `float32` выхода:

- двумерный `T.Kernel` и 128 threads;
- `T.alloc_shared` для A/B;
- `T.alloc_shared((block_M, block_N), "float32")` и `T.clear` для C;
- цикл `T.Pipelined(T.ceildiv(K, block_K), num_stages=0)`;
- две `T.copy`, затем `T.gemm`;
- финальная `T.copy` в C.

Начальные размеры в тесте кратны тайлам, чтобы сосредоточиться на dataflow.
Именно shared-output + `num_stages=0` используется в исполняемых Metal GEMM
тестах TileLang 0.1.13 и работает без Metal 4.

## Материалы

- [Tiled GEMM skeleton](https://tilelang.com/programming_guides/language_basics.html#tiled-gemm-skeleton)
- [Software pipeline](https://tilelang.com/programming_guides/software_pipeline.html)
- [Metal GEMM test v0.1.13](https://github.com/tile-ai/tilelang/blob/v0.1.13/testing/python/metal/test_metal_gemm_v2.py)

## Проверка

```bash
uv run pytest -vv test_task.py
```
