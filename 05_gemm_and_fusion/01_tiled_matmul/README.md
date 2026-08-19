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

## Новые конструкции TileLang

- `T.clear(C_local)` заполняет аккумулятор нулями. GEMM прибавляет частичные
  произведения, поэтому неизвестное начальное значение испортит весь результат.
- `T.Pipelined(iterations, num_stages=0)` описывает цикл по K-тайлам. Эта
  конструкция умеет сообщать compiler возможность перекрывать загрузки и
  вычисления. Здесь `num_stages=0` выбран ради переносимого Metal baseline и
  не включает многоступенчатый software pipeline.
- `T.gemm(A_shared, B_shared, C_local)` вычисляет произведение двух тайлов и
  накапливает его в `C_local`. По смыслу это
  `C_local += A_shared @ B_shared`.
- Двумерный `T.copy` переносит прямоугольную область. Начальный индекс
  `A[row_start, k_start]` вместе с формой destination определяет размер
  копируемого тайла.

Три размера тайла имеют разные роли: `block_M` — число строк результата,
`block_N` — число его столбцов, `block_K` — длина одного шага редукции.

## Параллель с WebGPU

Сетка `T.Kernel(ceildiv(N, block_N), ceildiv(M, block_M), threads=128)` близка к
двумерному `dispatchWorkgroups`: одна workgroup отвечает за один тайл `C`.
`A_shared`, `B_shared` и в этом baseline `C_local` похожи на массивы
`var<workgroup>`.

В ручном WGSL-ядре потребовались бы циклы совместной загрузки, проверки границ,
`workgroupBarrier()` и внутренний цикл умножений. `T.copy` скрывает
коллективную загрузку и синхронизацию, а `T.gemm` выражает всё тайловое
умножение одной операцией. Это важное отличие уровня абстракции TileLang.

## Как решить по шагам

1. Объявите JIT-фабрику и внутренний kernel с формами
   `A: (M, K) float16`, `B: (K, N) float16`, `C: (M, N) float32`.
2. Создайте двумерную сетку по тайлам результата: ось `x` покрывает `N`, ось
   `y` покрывает `M`. Передайте `threads=128` и получите `bx, by`.
3. Внутри блока выделите три shared-буфера:
   `A_shared(block_M, block_K)`, `B_shared(block_K, block_N)` и
   `C_local(block_M, block_N)`. Типы A/B — `float16`, C — `float32`.
4. Один раз, до K-loop, вызовите `T.clear(C_local)`.
5. Число K-шагов равно `T.ceildiv(K, block_K)`. Откройте цикл
   `for k in T.Pipelined(..., num_stages=0)`.
6. На шаге `k` загрузите тайл A, начинающийся с
   `[by * block_M, k * block_K]`, в `A_shared`.
7. Загрузите тайл B, начинающийся с
   `[k * block_K, bx * block_N]`, в `B_shared`.
8. Вызовите `T.gemm(A_shared, B_shared, C_local)`. Не очищайте C внутри цикла:
   каждый K-тайл должен добавляться к уже накопленной сумме.
9. После завершения всего K-loop скопируйте `C_local` в C, начиная с
   `[by * block_M, bx * block_N]`.
10. Верните kernel и проверьте dtype: тест ожидает `float32` output, хотя входы
    имеют `float16`.

Полезно проверить форму одного шага вручную:

```text
(block_M × block_K) @ (block_K × block_N)
                         ↓
                (block_M × block_N)
```

## Материалы

- [Tiled GEMM skeleton](https://tilelang.com/programming_guides/language_basics.html#tiled-gemm-skeleton)
- [Software pipeline](https://tilelang.com/programming_guides/software_pipeline.html)
- [Metal GEMM test v0.1.13](https://github.com/tile-ai/tilelang/blob/v0.1.13/testing/python/metal/test_metal_gemm_v2.py)

## Проверка

```bash
uv run pytest -vv test_task.py
```
