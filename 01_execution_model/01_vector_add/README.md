# Первое ядро: сложение векторов

Обычный цикл по миллиону элементов плохо описывает GPU. Вместо него мы запускаем
сетку блоков. `bx` выбирает блок, `i` — работу внутри блока, а
`global_i = bx * block + i` находит элемент исходного вектора.

`T.ceildiv(N, block)` создаёт достаточно блоков и для последнего неполного
тайла. TileLang умеет легализовать потенциально выходящие за границу обращения,
но границу всё равно нужно понимать и тестировать.

## Задание

Реализуйте `build_vector_add(N, block=256)`. Функция должна быть JIT-фабрикой,
возвращающей ядро с тремя `float32` буферами `A`, `B`, `C`. Используйте
`T.Kernel`, `T.Parallel` и присваивание `C[global_i] = A[global_i] + B[global_i]`.

Сигнатура внутри `@T.prim_func`:

```python
A: T.Tensor((N,), "float32")
B: T.Tensor((N,), "float32")
C: T.Tensor((N,), "float32")
```

Форма JIT-фабрики выглядит так (многоточия нужно заменить):

```python
@tilelang.jit
def build_vector_add(N: int, block: int = 256):
    @T.prim_func
    def kernel(A: T.Tensor((N,), "float32"), ...):
        with T.Kernel(...) as bx:
            ...

    return kernel
```

`build_vector_add(N)` компилирует специализацию и возвращает вызываемый kernel.
Эту же внешнюю конструкцию используйте в следующих заданиях.

Не задавайте `target="cuda"`: автоматический target выбирает Metal при запуске
с MPS-тензорами и CUDA при запуске с CUDA-тензорами.

## Новые конструкции TileLang

- `@tilelang.jit` превращает обычную Python-функцию в фабрику
  скомпилированных специализаций. `N` и `block` известны во время компиляции, а
  тензоры `A`, `B`, `C` передаются уже при запуске готового kernel.
- `@T.prim_func` помечает функцию, тело которой описывает GPU-ядро. Аннотации
  `T.Tensor((N,), "float32")` задают форму и тип буфера.
- `T.ceildiv(a, b)` вычисляет целочисленное округление `a / b` вверх. Поэтому
  при `N=1003`, `block=256` получится 4 блока, а не 3.
- `T.Kernel(grid, threads=block)` задаёт сетку из `grid` блоков и номинальное
  число потоков в каждом блоке. Переменная `bx` — индекс блока.
- `T.Parallel(block)` описывает `block` независимых итераций. Компилятор
  распределяет их между потоками. В этом простом 1D-ядре удобно представлять
  `i` как локальный индекс потока, хотя в общем случае это более
  высокоуровневая конструкция.

## Параллель с WebGPU

Для этого упражнения полезна почти буквальная модель:

```text
T.Kernel(ceildiv(N, block), threads=block)
    ≈ dispatchWorkgroups(ceil(N / block)) + @workgroup_size(block)

bx ≈ workgroup_id.x
i  ≈ local_invocation_id.x  (только как модель для этого простого kernel)
```

`A`, `B` и `C` похожи на три storage buffer. В WebGPU пришлось бы отдельно
создать bind group и compute pipeline; здесь JIT и вызов `kernel(a, b, out)`
берут эту обвязку на себя.

## Как решить по шагам

1. Откройте `task.py` и замените `raise NotImplementedError` декорированной
   фабрикой: поставьте `@tilelang.jit` прямо над `build_vector_add`.
2. Внутри фабрики объявите `@T.prim_func def kernel(...)` с тремя аргументами
   `A`, `B`, `C` и типами из условия.
3. Посчитайте число блоков прямо в `T.Kernel` как
   `T.ceildiv(N, block)`. Передайте `threads=block` и получите индекс `bx`.
4. Внутри блока создайте `for i in T.Parallel(block)`. Каждая итерация должна
   отвечать только за один элемент результата.
5. Переведите пару индексов `(bx, i)` в индекс массива:
   `global_i = bx * block + i`.
6. Запишите сумму `A[global_i] + B[global_i]` в `C[global_i]`.
7. После определения внутренней функции обязательно верните `kernel` из
   `build_vector_add`. Без `return kernel` фабрика вернёт `None`.
8. Сначала мысленно проверьте `N=1024`: получится ровно 4 полных блока. Затем
   проверьте `N=1003`: четвёртый блок частичный. Именно оба случая есть в тесте.

Каркас после шагов 1–3 должен выглядеть так:

```python
@tilelang.jit
def build_vector_add(N: int, block: int = 256):
    @T.prim_func
    def kernel(A: T.Tensor((N,), "float32"), ...):
        with T.Kernel(T.ceildiv(N, block), threads=block) as bx:
            # Параллельный цикл, вычисление global_i и одна запись в C.
            ...

    return kernel
```

## Материалы

- [Language Basics: Vector Add](https://tilelang.com/programming_guides/language_basics.html#id6)
- [`T.Kernel` и циклы](https://tilelang.com/programming_guides/language_basics.html#launching-work-with-t-kernel)

## Проверка

```bash
uv run pytest -vv test_task.py
```
