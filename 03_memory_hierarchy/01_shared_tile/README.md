# Тайлы и shared memory

Аргументы `T.Tensor` находятся в global memory. `T.alloc_shared` создаёт
быстрый буфер, общий для потоков блока, а `T.copy` описывает коллективное
перемещение тайла. Shared memory полезна, когда данные переиспользуются или
между потоками есть обмен; бессмысленное копирование через shared может только
замедлить ядро.

В этом упражнении копирование учебное: мы хотим увидеть полный путь
`global → shared → compute → global` до перехода к GEMM, где reuse становится
реальным.

## Задание

Реализуйте `build_tiled_square(N, block=256)`:

1. выделите `tile = T.alloc_shared((block,), "float32")`;
2. скопируйте `X[bx * block]` в `tile` через `T.copy`;
3. возведите элементы `tile` в квадрат через `T.Parallel`;
4. скопируйте тайл в `Y[bx * block]`.

Сначала используйте `N`, кратный `block`: handling хвостов для коллективных
copy зависит от lowering и здесь отвлекает от модели памяти.

## Новые конструкции TileLang

- `T.alloc_shared((block,), "float32")` выделяет один временный тайл в shared
  memory блока. Он создаётся отдельно для каждого блока и доступен всем его
  потокам. После завершения блока данные исчезают.
- `T.copy(source, destination)` описывает коллективное копирование целого тайла.
  Запись `X[bx * block]` здесь означает начало области, из которой TileLang
  должен перенести столько элементов, сколько вмещает `destination`.
- `T.Parallel(block)` выполняет независимое вычисление над уже загруженными
  элементами shared-тайла.

Важно различать `tile[i]` и `X[bx * block + i]`: первый адрес относится к
локальному shared-буферу блока, второй — к общему входному тензору в global
memory.

## Параллель с WebGPU

`T.alloc_shared` ближе всего к массиву `var<workgroup>` в WGSL. Но
`T.copy` заметно выше уровнем. В WebGPU коллективную загрузку обычно пришлось
бы развернуть вручную:

```text
tile[local_id] = X[block_start + local_id]
workgroupBarrier()
```

После вычисления потребовался бы ещё один barrier, если запись зависит от
работы других invocation. TileLang получает dataflow из `T.copy` и операций с
тайлом и вставляет необходимую синхронизацию при lowering.

## Как решить по шагам

1. Создайте обычную JIT-фабрику и внутренний kernel с `X` и `Y` формы `(N,)`.
2. Запустите `T.ceildiv(N, block)` блоков с `threads=block`. Тест использует
   `N=2048`, поэтому при `block=256` получится 8 полных блоков.
3. Внутри `T.Kernel`, но до вычислительного цикла, выделите
   `tile = T.alloc_shared((block,), "float32")`. Буфер нельзя выделять снаружи
   kernel.
4. Вычислите начало текущего глобального участка как `bx * block`.
5. Вызовите `T.copy(X[bx * block], tile)`. После этого `tile[0]` соответствует
   первому элементу текущего блока, а не `X[0]` для всех блоков.
6. Запустите `for i in T.Parallel(block)` и замените каждый элемент tile его
   квадратом: прочитайте `tile[i]` дважды и запишите результат обратно туда же.
7. После цикла вызовите `T.copy(tile, Y[bx * block])`, чтобы выгрузить весь
   обработанный тайл в output.
8. Верните kernel и проверьте путь данных:
   `X global → tile shared → tile shared → Y global`.

Каркас тела kernel:

```python
with T.Kernel(T.ceildiv(N, block), threads=block) as bx:
    tile = T.alloc_shared((block,), "float32")
    T.copy(X[bx * block], tile)
    for i in T.Parallel(block):
        ...
    T.copy(tile, Y[bx * block])
```

## Материалы

- [Memory scopes](https://tilelang.com/programming_guides/language_basics.html#memory-scopes-and-allocation)
- [`T.copy`](https://tilelang.com/programming_guides/language_basics.html#moving-data-t-copy)

## Проверка

```bash
uv run pytest -vv test_task.py
```
