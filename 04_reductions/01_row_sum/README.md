# Редукция по строкам

Редукция отличается от elementwise-ядра: несколько значений должны сойтись в
один результат. `T.alloc_fragment` описывает локальный фрагмент, а
`T.reduce_sum(..., dim=1)` сворачивает его вторую ось.

Для первого опыта берём ширину 256 — степень двойки и число, удобное для
параллельной reduction. Произвольные размеры требуют внимательной работы с
padding, layout и нейтральными значениями.

## Задание

Реализуйте `build_row_sum(M, N=256)`. Один block обрабатывает одну строку:

1. `with T.Kernel(M, threads=128) as row`;
2. фрагмент входа формы `(1, N)` и результата формы `(1,)`;
3. `T.copy(X[row, 0], local)`;
4. `T.reduce_sum(local, reduced, dim=1)`;
5. `T.copy(reduced, Y[row])`.

## Материалы

- [Reduction instructions](https://tilelang.com/programming_guides/instructions.html#compute-primitives)
- [Language Basics](https://tilelang.com/programming_guides/language_basics.html)

## Проверка

```bash
uv run pytest -vv test_task.py
```

