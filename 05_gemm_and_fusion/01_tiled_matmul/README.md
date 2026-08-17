# Тайловое матричное умножение

GEMM показывает, зачем нужны тайлы. Один block считает тайл результата
`C[block_M, block_N]`. На каждом шаге по K он загружает два небольших тайла A и
B в shared memory и переиспользует их во множестве умножений. Аккумулятор
`C_local` хранится во fragment/register scope.

Схема одного K-шага:

```text
A global -> A_shared --\
                         T.gemm -> C_local
B global -> B_shared --/
```

## Задание

Реализуйте `build_matmul(M, N, K, block_M=64, block_N=64, block_K=32)` для
`float16` входов и выхода, с аккумуляцией `float32`:

- двумерный `T.Kernel` и 128 threads;
- `T.alloc_shared` для A/B;
- `T.alloc_fragment` и `T.clear` для C;
- цикл `T.Pipelined(T.ceildiv(K, block_K), num_stages=2)`;
- две `T.copy`, затем `T.gemm`;
- финальная `T.copy` в C.

Начальные размеры в тесте кратны тайлам, чтобы сосредоточиться на dataflow.

## Материалы

- [Tiled GEMM skeleton](https://tilelang.com/programming_guides/language_basics.html#tiled-gemm-skeleton)
- [Software pipeline](https://tilelang.com/programming_guides/software_pipeline.html)

## Проверка

```bash
uv run pytest -vv test_task.py
```

