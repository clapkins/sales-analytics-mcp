"""Генератор синтетического датасета продаж для тестирования системы.

Создаёт ``data/sales_data.csv`` и ``data/README.md`` — эталонное описание
заложенных дефектов, по которому на этапе 3 проверяется, что
``DataCleaningSkill`` нашёл ровно то, что было внесено, не больше и не
меньше.

Порядок важен: сначала строятся реалистичные связи между колонками
(``Sales`` через ``Quantity`` и цену товара, ``Profit`` через ``Sales`` и
маржу товара), и только потом поверх них вносятся дефекты — пропуски,
выбросы, разнобой в датах и регионах, дубли строк. Если сделать наоборот,
дефекты исказят сами зависимости, на которых строится анализ.

Результат воспроизводим: фиксированный ``random_state`` (см. ``SEED``)
даёт побайтово идентичный файл при повторном запуске.

Запуск:
    python scripts/generate_data.py
"""

import calendar
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
YEARS = (2023, 2024)
N_UNIQUE_ROWS = 175
N_DUPLICATES = 5
OUTLIER_COUNT = 4
OUTLIER_FACTOR_RANGE = (8.0, 15.0)
MISSING_SHARE = 0.07
REGION_DEFECT_SHARE = 0.15
DATE_FORMATS = ("%Y-%m-%d", "%d.%m.%Y", "%m/%d/%Y")
DATE_FORMAT_WEIGHTS = (0.5, 0.25, 0.25)
HIGH_TURNOVER_LOW_MARGIN_PRODUCT = "Wireless Mouse"


@dataclass(frozen=True)
class ProductProfile:
    """Профиль товара, задающий его цену, маржу и объёмы продаж.

    Attributes:
        unit_price: Средняя цена за единицу товара в долларах.
        margin: Средняя маржа (доля прибыли в выручке), 0..1.
        weight: Вероятность выбора товара для очередной строки.
        qty_range: Диапазон (min, max) количества единиц за одну продажу.
    """

    unit_price: float
    margin: float
    weight: float
    qty_range: tuple[int, int]


PRODUCTS: dict[str, ProductProfile] = {
    "Laptop Pro": ProductProfile(850.0, 0.22, 0.16, (1, 6)),
    "Monitor 27": ProductProfile(220.0, 0.18, 0.18, (1, 10)),
    "Office Chair": ProductProfile(150.0, 0.25, 0.14, (1, 8)),
    "Desk Lamp": ProductProfile(35.0, 0.30, 0.16, (2, 20)),
    "Webcam HD": ProductProfile(45.0, 0.20, 0.14, (1, 15)),
    HIGH_TURNOVER_LOW_MARGIN_PRODUCT: ProductProfile(15.0, 0.08, 0.22, (10, 120)),
}

REGIONS = ("North", "South", "East", "West")
REGION_WEIGHTS = (0.22, 0.18, 0.28, 0.32)
REGION_DEMAND = {"North": 0.95, "South": 0.85, "East": 1.05, "West": 1.15}

# Сезонный множитель по месяцу: пик в ноябре-декабре, спад летом и в начале года.
SEASONALITY = {
    1: 0.85,
    2: 0.80,
    3: 0.90,
    4: 0.95,
    5: 1.00,
    6: 0.90,
    7: 0.85,
    8: 0.90,
    9: 1.05,
    10: 1.15,
    11: 1.35,
    12: 1.45,
}


def _build_year_month_weights() -> tuple[list[tuple[int, int]], np.ndarray]:
    """Строит список пар (год, месяц) и их веса с учётом сезонности."""
    year_months = [(year, month) for year in YEARS for month in range(1, 13)]
    weights = np.array([SEASONALITY[month] for _, month in year_months], dtype=float)
    weights /= weights.sum()
    return year_months, weights


def build_base_dataframe(rng: np.random.Generator) -> pd.DataFrame:
    """Строит датасет с реалистичными связями между колонками.

    ``Sales`` определяется через ``Quantity`` и цену товара, ``Profit`` —
    через ``Sales`` и маржу товара; обе зависимости зашумлены, чтобы
    корреляция была заметной, но не идеальной. Сезонность (пик в Q4) и
    рост год-к-году (+12% в 2024) заложены через множители по месяцу и
    году, разница между регионами — через ``REGION_DEMAND``.

    Args:
        rng: Инициализированный генератор случайных чисел.

    Returns:
        Датафрейм без дефектов, с колонками Date, Product, Region, Sales,
        Quantity, Profit.
    """
    year_months, year_month_weights = _build_year_month_weights()
    products = list(PRODUCTS)
    product_weights = [PRODUCTS[name].weight for name in products]

    rows = []
    for _ in range(N_UNIQUE_ROWS):
        year, month = year_months[rng.choice(len(year_months), p=year_month_weights)]
        day = int(rng.integers(1, calendar.monthrange(year, month)[1] + 1))
        date = pd.Timestamp(year=year, month=month, day=day)

        product_name = str(rng.choice(products, p=product_weights))
        profile = PRODUCTS[product_name]
        region = str(rng.choice(REGIONS, p=REGION_WEIGHTS))

        growth = 1.12 if year == 2024 else 1.0
        season_factor = SEASONALITY[month]
        region_factor = REGION_DEMAND[region]

        low, high = profile.qty_range
        base_qty = int(rng.integers(low, high + 1))
        quantity = max(
            1,
            round(base_qty * season_factor * region_factor * growth * rng.normal(1.0, 0.05)),
        )

        price = max(profile.unit_price * rng.normal(1.0, 0.03), 1.0)
        sales = round(quantity * price, 2)

        margin = min(max(profile.margin * rng.normal(1.0, 0.12), 0.02), 0.6)
        profit = round(sales * margin, 2)

        rows.append(
            {
                "Date": date,
                "Product": product_name,
                "Region": region,
                "Sales": sales,
                "Quantity": quantity,
                "Profit": profit,
            }
        )

    return pd.DataFrame(rows)


def inject_outliers(
    df: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Вносит выбросы по Sales — на порядок выше нормы для своего товара.

    Выполняется сразу после расчёта связей между колонками и до всех
    остальных дефектов: если внести выбросы позже (например, после
    дублирования), в датасете случайно окажется больше или меньше
    аномалий, чем задумано.

    Args:
        df: Датафрейм со связями между колонками, но без дефектов.
        rng: Генератор случайных чисел.

    Returns:
        Кортеж: датафрейм с изменённым Sales, индексы затронутых строк,
        применённые множители.
    """
    df = df.copy()
    indices = rng.choice(len(df), size=OUTLIER_COUNT, replace=False)
    factors = rng.uniform(*OUTLIER_FACTOR_RANGE, size=OUTLIER_COUNT)
    df.loc[indices, "Sales"] = (df.loc[indices, "Sales"] * factors).round(2)
    return df, indices, factors


def inject_missing_values(
    df: pd.DataFrame, rng: np.random.Generator, sales_exclude: np.ndarray
) -> pd.DataFrame:
    """Проставляет пропуски в Sales и Profit (~7% строк по каждой колонке).

    Строки, уже испорченные выбросом по Sales, исключаются из выборки
    под пропуски в Sales — иначе выброс мог бы затереться пропуском и
    перестать быть обнаружимым.

    Args:
        df: Датафрейм после внесения выбросов.
        rng: Генератор случайных чисел.
        sales_exclude: Индексы строк, которые нельзя трогать в Sales.

    Returns:
        Датафрейм с пропусками (NaN) в Sales и Profit.
    """
    df = df.copy()
    n_missing = round(MISSING_SHARE * len(df))

    sales_pool = np.setdiff1d(np.arange(len(df)), sales_exclude)
    sales_idx = rng.choice(sales_pool, size=n_missing, replace=False)
    df.loc[sales_idx, "Sales"] = np.nan

    profit_idx = rng.choice(len(df), size=n_missing, replace=False)
    df.loc[profit_idx, "Profit"] = np.nan

    return df


def scramble_date_formats(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Переводит колонку Date в смесь из трёх текстовых форматов.

    Реальные выгрузки редко приходят в едином формате дат — это
    эмулируется явно, чтобы ``DataLoadingSkill`` было что распознавать.

    Args:
        df: Датафрейм с колонкой Date типа datetime.
        rng: Генератор случайных чисел.

    Returns:
        Датафрейм, где Date — строки в одном из трёх форматов.
    """
    df = df.copy()
    fmt_choice = rng.choice(len(DATE_FORMATS), size=len(df), p=DATE_FORMAT_WEIGHTS)
    df["Date"] = [
        date.strftime(DATE_FORMATS[idx]) for date, idx in zip(df["Date"], fmt_choice, strict=True)
    ]
    return df


def _dirty_region(value: str, rng: np.random.Generator) -> str:
    """Возвращает испорченный вариант названия региона."""
    variant = rng.integers(0, 3)
    if variant == 0:
        return f"  {value}  "
    if variant == 1:
        return value.upper()
    return value.lower()


def scramble_region_text(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Портит регистр и добавляет пробелы в части значений Region.

    Симулирует то, что данные вводились вручную из разных источников.

    Args:
        df: Датафрейм с каноническими значениями Region.
        rng: Генератор случайных чисел.

    Returns:
        Датафрейм с частью значений Region в «грязном» виде.
    """
    df = df.copy()
    n_defects = round(REGION_DEFECT_SHARE * len(df))
    idx = rng.choice(len(df), size=n_defects, replace=False)
    df.loc[idx, "Region"] = [_dirty_region(value, rng) for value in df.loc[idx, "Region"]]
    return df


def inject_duplicates(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Добавляет точные дубли уже готовых (испорченных) строк.

    Выполняется в самом конце, после всех остальных дефектов: иначе
    копии одной и той же строки могли бы разъехаться по формату даты
    или пропускам и перестать быть полными дублями.

    Args:
        df: Датафрейм со всеми остальными дефектами.
        rng: Генератор случайных чисел.

    Returns:
        Датафрейм с добавленными дублями строк.
    """
    idx = rng.choice(len(df), size=N_DUPLICATES, replace=False)
    duplicates = df.iloc[idx].copy()
    return pd.concat([df, duplicates], ignore_index=True)


def shuffle_rows(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Перемешивает строки, чтобы дубли не лежали рядом друг с другом."""
    order = rng.permutation(len(df))
    return df.iloc[order].reset_index(drop=True)


def _classify_date_format(value: str) -> str:
    """Определяет формат строки даты по разделителю."""
    if "-" in value:
        return "%Y-%m-%d"
    if "." in value:
        return "%d.%m.%Y"
    return "%m/%d/%Y"


def build_readme(
    df: pd.DataFrame,
    period: tuple[pd.Timestamp, pd.Timestamp],
    outlier_factors: np.ndarray,
) -> str:
    """Формирует текст ``data/README.md`` — эталонное описание дефектов.

    Числа считаются из фактически сгенерированного датафрейма, а не
    пишутся руками, поэтому документ не может разойтись с содержимым CSV.

    Args:
        df: Итоговый датафрейм (после всех дефектов, дублей и перемешивания).
        period: Минимальная и максимальная дата до перевода в строки.
        outlier_factors: Множители, применённые при внесении выбросов.

    Returns:
        Текст файла в формате Markdown.
    """
    period_start, period_end = period

    format_counts = df["Date"].map(_classify_date_format).value_counts().to_dict()
    formats_desc = ", ".join(f"`{fmt}` — {count} строк" for fmt, count in format_counts.items())

    is_canonical = df["Region"].str.strip().str.capitalize() == df["Region"]
    region_defect_count = int((~is_canonical).sum())

    products_desc = "\n".join(
        f"  - **{name}** — цена ~${profile.unit_price:.0f}, маржа ~{profile.margin:.0%}"
        + (
            " _(высокий оборот, низкая маржа — материал для инсайта)_"
            if name == HIGH_TURNOVER_LOW_MARGIN_PRODUCT
            else ""
        )
        for name, profile in PRODUCTS.items()
    )
    regions_desc = ", ".join(f"{name} (×{REGION_DEMAND[name]:.2f})" for name in REGIONS)

    return f"""# data/sales_data.csv — синтетический датасет продаж

Сгенерирован скриптом `scripts/generate_data.py` с фиксированным
`random_state = {SEED}`: повторный запуск воспроизводит файл побайтово
идентично.

Этот файл — эталон для проверки на этапе 3: `DataCleaningSkill` должен
найти дефекты, перечисленные ниже.

## Общие параметры

- Итоговых строк: **{len(df)}** ({N_UNIQUE_ROWS} уникальных + {N_DUPLICATES} дублей).
- Период: {period_start:%Y-%m-%d} — {period_end:%Y-%m-%d}.
- Товары (6):
{products_desc}
- Регионы (4) и множитель спроса относительно среднего: {regions_desc}.
- Заложена сезонность (пик в ноябре-декабре, спад зимой и летом) и рост
  ~12% в 2024 году относительно 2023.

## Связи между колонками (заложены до внесения дефектов)

- `Sales = Quantity × цена товара × шум` — цена и объём зависят от
  товара, сезона и региона.
- `Profit = Sales × маржа товара × шум` — маржа своя у каждого товара.
- `{HIGH_TURNOVER_LOW_MARGIN_PRODUCT}` — намеренно высокий оборот при
  низкой марже: единственный товар, способный давать много выручки при
  малой прибыли, — ожидаемая находка для отчёта LLM.

## Намеренные дефекты (внесены после расчёта связей выше)

1. **Выбросы по Sales** — {len(outlier_factors)} строки, значение Sales
   завышено в {outlier_factors.min():.1f}–{outlier_factors.max():.1f} раз
   относительно нормы для своего товара. Quantity и Profit в этих строках
   не менялись — это эмулирует ошибку ввода данных, а не реальный
   всплеск продаж.
2. **Пропуски** — {int(df["Sales"].isna().sum())} пропущенных значений в
   `Sales` и {int(df["Profit"].isna().sum())} в `Profit`
   (~{MISSING_SHARE:.0%} каждая колонка от {N_UNIQUE_ROWS} строк до
   дублирования).
3. **Полные дубли строк** — {int(df.duplicated().sum())} строк являются
   точными копиями других строк датасета (совпадают все поля, включая
   уже испорченные формат даты и регистр региона).
4. **Разнобой в формате дат** — три формата вперемешку: {formats_desc}.
5. **Разнобой в `Region`** — {region_defect_count} строк с лишними
   пробелами по краям или изменённым регистром (например, `"  South  "`,
   `"NORTH"`, `"east"`) вместо канонических {", ".join(REGIONS)}.

## Как пересоздать

```
python scripts/generate_data.py
```
"""


def main() -> None:
    """Генерирует датасет и сопроводительный data/README.md."""
    rng = np.random.default_rng(SEED)

    df = build_base_dataframe(rng)
    period = (df["Date"].min(), df["Date"].max())

    df, outlier_indices, outlier_factors = inject_outliers(df, rng)
    df = inject_missing_values(df, rng, sales_exclude=outlier_indices)
    df = scramble_date_formats(df, rng)
    df = scramble_region_text(df, rng)
    df = inject_duplicates(df, rng)
    df = shuffle_rows(df, rng)
    df = df[["Date", "Product", "Region", "Sales", "Quantity", "Profit"]]

    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    df.to_csv(data_dir / "sales_data.csv", index=False, encoding="utf-8")

    readme = build_readme(df, period, outlier_factors)
    (data_dir / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
