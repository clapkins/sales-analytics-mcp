"""Пути к рабочим директориям проекта.

Единая точка правды для путей — скиллы не хардкодят относительные
пути каждый на свой лад, а импортируют константы отсюда.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHARTS_DIR = PROJECT_ROOT / "charts"
REPORTS_DIR = PROJECT_ROOT / "reports"

CHARTS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
