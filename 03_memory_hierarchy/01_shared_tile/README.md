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

## Материалы

- [Memory scopes](https://tilelang.com/programming_guides/language_basics.html#memory-scopes-and-allocation)
- [`T.copy`](https://tilelang.com/programming_guides/language_basics.html#moving-data-t-copy)

## Проверка

```bash
uv run pytest -vv test_task.py
```

