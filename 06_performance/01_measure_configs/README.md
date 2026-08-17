# Производительность: сначала измерение

Размер тайла влияет на число блоков, shared memory, registers, occupancy и набор
hardware instructions. Универсально лучшей конфигурации нет. TileLang позволяет
получить profiler у скомпилированного ядра и позже автоматизировать поиск через
autotuner.

В этом задании не нужно победить библиотечный GEMM. Нужно провести корректный
маленький эксперимент: несколько допустимых конфигураций, одинаковые входы,
проверка результата и benchmark после компиляции.

## Задание

Реализуйте две функции:

1. `candidate_configs()` возвращает не менее трёх различных словарей с ключами
   `block_M`, `block_N`, `block_K`, `num_stages`, `threads`;
2. `measure(kernel, args, device, warmup=5, repeat=20)` возвращает положительную
   latency в миллисекундах. Используйте `time.perf_counter`, несколько warmup и
   обязательную синхронизацию: `torch.mps.synchronize()` для MPS или
   `torch.cuda.synchronize()` для CUDA.

Затем вручную соберите GEMM для каждой конфигурации и запишите в комментарии к
`task.py`: GPU, формы матриц, TileLang version и полученные latency. Не делайте
вывод по одному запуску или без проверки результата.

## Материалы

- [Autotuning](https://tilelang.com/programming_guides/autotuning.html)
- [официальный quickstart](https://github.com/tile-ai/tilelang/blob/main/examples/quickstart.py)

## Проверка

```bash
uv run pytest -vv test_task.py
```
