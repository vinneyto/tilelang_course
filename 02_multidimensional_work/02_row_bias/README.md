# Broadcasting без магии: bias по строкам

PyTorch автоматически растягивает вектор `bias` формы `[N]` на матрицу
`X[M, N]`. В низкоуровневом ядре broadcasting превращается в явную формулу:
каждый элемент `Y[row, col]` читает `bias[col]`.

Это полезная связь между тензорной математикой и GPU-кодом: broadcasting не
создаёт физическую матрицу bias, а задаёт способ переиспользовать значения.

## Задание

Реализуйте fused-ядро `build_row_bias_relu(M, N, ...)`, вычисляющее
`Y = relu(X + bias)`. Используйте двумерную сетку из предыдущего задания и не
создавайте промежуточный буфер.

## Материалы

- [Language Basics](https://tilelang.com/programming_guides/language_basics.html)
- [Instructions](https://tilelang.com/programming_guides/instructions.html)

## Проверка

```bash
uv run pytest -vv test_task.py
```

