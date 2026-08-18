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

## Материалы

- [Language Basics: control flow](https://tilelang.com/programming_guides/language_basics.html#loops-and-control-flow)
- [Instruction reference](https://tilelang.com/programming_guides/instructions.html)

## Проверка

```bash
uv run pytest -vv test_task.py
```

