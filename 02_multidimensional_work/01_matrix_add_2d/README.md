# Двумерная сетка

Для матрицы удобнее сохранить двумерную структуру задачи. Пусть `bx` выбирает
тайл столбцов, `by` — тайл строк, а `T.Parallel(block_M, block_N)` распределяет
элементы внутри тайла. Так формула индекса остаётся связана с формой данных.

Порядок координат в `T.Kernel(grid_x, grid_y)` важен: `bx` связан с осью N
(столбцы), `by` — с осью M (строки).

## Задание

Реализуйте `build_matrix_add(M, N, block_M=16, block_N=16)` для двух матриц
`float32`. Сетка должна иметь форму
`(T.ceildiv(N, block_N), T.ceildiv(M, block_M))`.

Вычислите:

```python
row = by * block_M + i
col = bx * block_N + j
```

## Новые конструкции TileLang

- `T.Kernel(grid_x, grid_y)` создаёт двумерную сетку блоков. Возвращаемые
  индексы нужно читать в том же порядке: `as bx, by`.
- `T.Parallel(block_M, block_N)` создаёт двумерное пространство независимых
  итераций. В теле цикла доступны два индекса: `i` по строкам тайла и `j` по
  столбцам тайла.
- `T.Tensor((M, N), "float32")` описывает двумерный буфер. К элементу
  обращаются как `A[row, col]`, а не через ручной линейный индекс.

`T.Parallel` по-прежнему является логическим описанием параллельной работы.
Произведение `block_M * block_N` не обязано буквально означать такое же число
аппаратных threads: отображение выбирает compiler.

## Параллель с WebGPU

Сетка соответствует `dispatchWorkgroups(grid_x, grid_y)`, где ось `x` идёт по
столбцам матрицы, а `y` — по строкам. Полезная карта индексов:

```text
bx ≈ workgroup_id.x             col = bx * block_N + j
by ≈ workgroup_id.y             row = by * block_M + i
```

В обычном WGSL часто выбирают `@workgroup_size(block_N, block_M)` и получают
`i`, `j` из `local_invocation_id`. TileLang позволяет сначала описать
двумерный `T.Parallel`, а конкретное отображение выполнить compiler.

## Как решить по шагам

1. Добавьте `@tilelang.jit` к `build_matrix_add` и объявите внутри
   `@T.prim_func` с буферами `A`, `B`, `C` формы `(M, N)`.
2. Посчитайте число блоков по каждой оси отдельно. По `x` нужно покрыть `N`
   столбцов блоками ширины `block_N`; по `y` — `M` строк блоками высоты
   `block_M`.
3. Откройте kernel как
   `T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M)) as bx, by`.
   Сохраните именно порядок `N`, затем `M`.
4. Создайте двумерный цикл
   `for i, j in T.Parallel(block_M, block_N)`.
5. Вычислите глобальные `row` и `col` по формулам из условия.
6. Запишите `C[row, col] = A[row, col] + B[row, col]`.
7. Верните kernel из фабрики.
8. Проверьте обе формы из теста. `(64, 96)` полностью делится на тайлы, а
   `(37, 53)` проверяет неполные блоки сразу по обеим осям.

## Материалы

- [`T.Kernel`](https://tilelang.com/programming_guides/language_basics.html#launching-work-with-t-kernel)
- [`T.Parallel`](https://tilelang.com/programming_guides/language_basics.html#loops-and-control-flow)

## Проверка

```bash
uv run pytest -vv test_task.py
```
