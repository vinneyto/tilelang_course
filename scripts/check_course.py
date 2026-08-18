"""Статическая проверка курса, не требующая TileLang или GPU."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    themes = sorted(path for path in ROOT.glob("[0-9][0-9]_*") if path.is_dir())
    assert themes, "Не найдены каталоги тем"
    exercise_count = 0

    for theme in themes:
        exercises = sorted(path for path in theme.glob("[0-9][0-9]_*") if path.is_dir())
        assert exercises, f"В теме {theme.name} нет заданий"
        exercise_count += len(exercises)

        for exercise in exercises:
            expected = [exercise / name for name in ("README.md", "task.py", "test_task.py")]
            missing = [path.name for path in expected if not path.is_file()]
            assert not missing, f"{exercise.relative_to(ROOT)}: отсутствуют {missing}"

            readme = expected[0].read_text(encoding="utf-8")
            assert "## Задание" in readme, f"{exercise.relative_to(ROOT)}: нет задания"
            assert "tilelang.com" in readme, f"{exercise.relative_to(ROOT)}: нет ссылки на TileLang"

            for source in expected[1:]:
                ast.parse(source.read_text(encoding="utf-8"), filename=str(source))

    print(f"Курс корректен: {len(themes)} тем, {exercise_count} заданий")


if __name__ == "__main__":
    main()

