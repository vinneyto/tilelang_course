# Практикум: основы TileLang

Небольшой практический курс по написанию вычислительных ядер на
[TileLang](https://tilelang.com/). Он не пытается сразу довести ядра до рекордов
производительности: цель — научиться читать программу TileLang, понимать сетку
запуска, перемещение данных и связь ядра с тензорами PyTorch.

Курс подготовлен по API TileLang 0.1.13. Проект быстро развивается, поэтому при
работе с другой версией сверяйтесь с официальной документацией.

## Что такое TileLang

TileLang — Python-подобный DSL поверх компиляторной инфраструктуры TVM. В нём
явно задаются сетка запуска, тайлы и уровни памяти, а компилятор опускает эту
программу в код конкретного backend: CUDA, HIP, CPU или развивающийся Metal.

Ментальная модель курса:

1. PyTorch создаёт входные и эталонные тензоры.
2. TileLang описывает вычислительное ядро.
3. `@tilelang.jit` компилирует специализацию ядра.
4. Результат сравнивается с простой реализацией на PyTorch.
5. Только после проверки корректности измеряется производительность.

## Как устроен курс

Пронумерованные каталоги — темы, вложенные каталоги — задания. В каждом задании:

- `README.md` — теория, разбор новых конструкций, параллель с WebGPU и
  пошаговый план решения;
- `task.py` — заготовка с `NotImplementedError`;
- `test_task.py` — проверка против PyTorch.

Меняйте только `task.py`. Проходите задания по порядку: каждое вводит один новый
слой модели GPU, а не просто ещё одну функцию API.

### Если вы пришли из WebGPU

TileLang и WGSL описывают GPU-вычисления на разных уровнях. В WebGPU вы вручную
создаёте pipeline, bind groups и вызываете `dispatchWorkgroups`. В упражнениях
курса PyTorch-тензоры играют роль уже созданных storage buffers, а TileLang JIT
берёт на себя pipeline, привязку аргументов и компиляцию shader/kernel.

Полезная начальная карта соответствий:

| TileLang | Ближайшая идея в WebGPU | Важное отличие |
| --- | --- | --- |
| `@T.prim_func` | entry point compute shader | Описывает kernel, но не содержит WGSL-декораторы |
| `T.Tensor` | storage buffer | Форма и dtype входят в тип аргумента |
| `T.Kernel(grid, threads=...)` | `dispatchWorkgroups(...)` + `@workgroup_size(...)` | Grid и размер группы задаются в одном месте |
| `T.Parallel(...)` | параллельная работа invocation внутри dispatch | Это логический цикл: compiler сам отображает итерации на threads |
| `T.alloc_shared` | `var<workgroup>` | TileLang знает форму и тип тайла |
| `T.copy` | коллективная загрузка/выгрузка тайла | В WGSL её обычно пришлось бы писать вручную |
| `T.Serial` | обычный `for` внутри invocation | Итерации выполняются последовательно |

Это именно параллели для построения ментальной модели, а не обещание
построчного соответствия сгенерированному Metal/CUDA-коду. В частности,
`T.Parallel` не следует всегда читать как прямой доступ к
`local_invocation_id`: layouts и lowering могут распределить работу иначе.

## Окружение

Нужны Python 3.11+, [uv](https://docs.astral.sh/uv/) и поддерживаемое устройство.
Наиболее предсказуемый вариант для первого прохождения — Linux/Windows с NVIDIA
GPU и CUDA:

```bash
uv venv
uv sync --dev
uv run python scripts/check_environment.py
```

TileLang закреплён на версии `0.1.13`, под API которой написан курс. PyTorch
ограничен веткой `<2.12`, потому что выбранный `2.11.0` сохраняет wheel для
macOS 13 arm64, тогда как более новые релизы требуют macOS 14. `uv` установит
PyTorch, TileLang и pytest из `pyproject.toml`; вручную вызывать
`pip install` не нужно. После появления `uv.lock` используйте
`uv sync --dev --locked`, чтобы получить ровно проверенный набор зависимостей.

### Apple Silicon / Metal

В TileLang 0.1.13 есть готовый macOS arm64 wheel и поддерживаемый Metal backend,
поэтому на M4 используйте обычный `uv sync --dev`. Базовые ядра курса специально
ограничены Metal-совместимым подмножеством. GEMM использует simdgroup fallback:
shared-аккумулятор, небольшие тайлы и `num_stages=0`. Это не самый быстрый
вариант для CUDA, зато он соответствует Metal-тестам TileLang и не требует
Metal 4 cooperative tensors, доступных только на более новом железе.

Проверьте установку и выполнение готового ядра до решения заданий:

```bash
uv sync --dev --locked
uv run pytest -vv tests/test_metal_smoke.py
```

Smoke-test создаёт MPS-тензоры, компилирует настоящее TileLang-ядро и сравнивает
результат с PyTorch. На машине без MPS он будет корректно пропущен.

Официальные инструкции:

- [установка TileLang](https://tilelang.com/get_started/Installation.html);
- [устройство языка](https://tilelang.com/programming_guides/language_basics.html);
- [внутренности Metal backend](https://tilelang.com/compiler_internals/metal_tilelang_development.html).

## Запуск задания

```bash
cd 01_execution_model/01_vector_add
uv run pytest -vv test_task.py
```

Если совместимого accelerator нет, GPU-тест будет пропущен. До решения задания
его тест ожидаемо падает с `NotImplementedError`. Структуру курса и
синтаксис всех файлов можно проверить без TileLang и GPU:

```bash
uv run python scripts/check_course.py
```

## Темы

1. `01_execution_model` — JIT, сетка запуска, блоки, `T.Parallel`, границы.
2. `02_multidimensional_work` — двумерная сетка и broadcasting, записанный явно.
3. `03_memory_hierarchy` — global/shared/fragment и `T.copy`.
4. `04_reductions` — параллельная редукция по строкам.
5. `05_gemm_and_fusion` — переносимый тайловый GEMM и fusion эпилога ReLU.
6. `06_performance` — generated source, benchmark и сравнение конфигураций.

## Правила работы с GPU-ядрами

- Сначала напишите PyTorch reference и проверьте корректность.
- Начинайте с маленьких размеров; отдельно проверяйте размер с неполным хвостом.
- Не сравнивайте `float16` через точное равенство: используйте `rtol`/`atol`.
- Не считайте первое выполнение временем ядра: оно включает JIT-компиляцию.
- Меняйте один параметр запуска за раз и записывайте результат измерения.
- Ускорение без теста корректности не считается результатом.

## Что останется за рамками

Курс только подводит к FlashAttention и другим сложным AI-ядрам. Здесь нет TMA,
warp specialization, ручных layout, distributed kernels и архитектурных
intrinsics. После курса переходите к официальным примерам TileLang и сначала
разбирайте их как комбинацию уже знакомых: сетка → память → вычисление → store.
