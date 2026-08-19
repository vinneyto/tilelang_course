# Границы и fusion: affine + ReLU

GPU-ядро выгодно использовать не только ради одной операции, но и чтобы не
писать промежуточный тензор в global memory. Выражение
`relu(x * scale + bias)` можно вычислить за один проход.

`scale` и `bias` в этом задании — compile-time параметры фабрики ядра. Они
попадают в сгенерированную программу как константы. Это специализация: меньше
гибкости во время запуска, но больше возможностей для оптимизации.

## Задание

Реализуйте `build_affine_relu(N, scale, bias, block=256)` для `float32`.
Сделайте одно выражение store через `T.max(x, 0.0)`. Обязательно используйте
`T.ceildiv`: тест содержит размер, не кратный блоку.

## Новые конструкции TileLang

- `scale` и `bias` — параметры внешней `@tilelang.jit`-фабрики, поэтому это
  compile-time значения. Они не являются отдельными GPU-буферами.
- `T.max(a, b)` — поэлементный максимум в выражении TileLang. Для одного
  скаляра `T.max(value, 0.0)` реализует ReLU.
- Fusion здесь означает, что умножение, прибавление и ReLU находятся в одном
  kernel и завершаются одной записью в `Y`.

Все остальные конструкции — `T.Tensor`, `T.Kernel`, `T.ceildiv` и
`T.Parallel` — работают так же, как в предыдущем задании.

## Параллель с WebGPU

В WGSL `scale` и `bias` можно было бы передать через uniform buffer или
зафиксировать как override constants. В этом упражнении они ближе именно к
override constants: изменение значения вызывает новую JIT-специализацию.

Одна invocation вычисляла бы примерно такое выражение:

```wgsl
y[index] = max(x[index] * scale + bias, 0.0);
```

Отдельный dispatch для ReLU не нужен. Это экономит запись промежуточного
`x * scale + bias` в global memory и его повторное чтение.

## Как решить по шагам

1. Возьмите структуру решения `01_vector_add`: внешний `@tilelang.jit`,
   внутренний `@T.prim_func`, `T.Kernel` и `T.Parallel` остаются теми же.
2. У внутреннего kernel объявите только два буфера: входной `X` и выходной
   `Y`, оба `T.Tensor((N,), "float32")`. `scale` и `bias` уже доступны из
   внешней функции и аргументами kernel не становятся.
3. Запустите `T.ceildiv(N, block)` блоков по `threads=block` потоков. Не
   используйте `N // block`: при `N=1003` он потеряет хвост.
4. Для каждой итерации `T.Parallel(block)` снова вычислите
   `global_i = bx * block + i`.
5. Сначала составьте скалярное выражение `X[global_i] * scale + bias`.
6. Оберните его в `T.max(..., 0.0)` и сразу присвойте `Y[global_i]`. Не
   создавайте промежуточный тензор и не запускайте второй kernel.
7. Верните внутренний `kernel` из фабрики.
8. Сверьте формулу с reference из теста:
   `torch.relu(x * scale + bias)`. Порядок операций должен совпадать.

## Материалы

- [Language Basics: control flow](https://tilelang.com/programming_guides/language_basics.html#loops-and-control-flow)
- [Instruction reference](https://tilelang.com/programming_guides/instructions.html)

## Проверка

```bash
uv run pytest -vv test_task.py
```
