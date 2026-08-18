# Первая редукция по строкам

Редукция отличается от elementwise-ядра: несколько значений должны сойтись в
один результат. Сначала реализуем переносимый baseline через scalar accumulator
и `T.Serial`. Он намеренно использует один thread на строку и будет медленным,
но одинаково выражается на CUDA и Metal. Это полезная точка отсчёта: сначала
корректная формула, затем параллельная редукция и benchmark.

## Задание

Реализуйте `build_row_sum(M, N=256)`. Один block обрабатывает одну строку:

1. `with T.Kernel(M, threads=1) as row`;
2. `acc = T.alloc_var("float32")` и обнуление accumulator;
3. цикл `for col in T.Serial(N)`;
4. `acc += X[row, col]`;
5. запись `Y[row] = acc`.

После прохождения курса сравните этот baseline с `T.reduce_sum` на вашем
backend. У параллельных reduction больше требований к ширине, layout и числу
threads, поэтому они вынесены за пределы гарантированного M4-маршрута.

## Материалы

- [Control flow](https://tilelang.com/programming_guides/control_flow.html)
- [Reduction instructions для следующего шага](https://tilelang.com/programming_guides/instructions.html#compute-primitives)
- [Language Basics](https://tilelang.com/programming_guides/language_basics.html)

## Проверка

```bash
uv run pytest -vv test_task.py
```
