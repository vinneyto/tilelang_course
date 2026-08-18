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

## Материалы

- [`T.Kernel`](https://tilelang.com/programming_guides/language_basics.html#launching-work-with-t-kernel)
- [`T.Parallel`](https://tilelang.com/programming_guides/language_basics.html#loops-and-control-flow)

## Проверка

```bash
uv run pytest -vv test_task.py
```

