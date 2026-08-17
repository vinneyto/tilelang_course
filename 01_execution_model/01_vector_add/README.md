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

## Материалы

- [Language Basics: Vector Add](https://tilelang.com/programming_guides/language_basics.html#id6)
- [`T.Kernel` и циклы](https://tilelang.com/programming_guides/language_basics.html#launching-work-with-t-kernel)

## Проверка

```bash
uv run pytest -vv test_task.py
```
