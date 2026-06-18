# Grokking Carries

Исследование механистической интерпретируемости (mechanistic interpretability) арифметических операций в малых GPT-подобных моделях. Проект нацелен на анализ внутренних представлений числовой линии и алгоритмов переноса разрядов (carries/borrows) с использованием probing, causal tracing и activation patching.

## Структура репозитория

```text
grokking-carries/
├── pyproject.toml
├── uv.lock
├── README.md
├── notebooks/                  # Jupyter ноутбуки для исследовательского анализа (SVD, Logit Lens)
├── scripts/                    # Точки входа для запуска пайплайнов
│   ├── 01_generate_data.py
│   └── 02_run_training.py
└── src/
    └── grokking_carries/
        ├── __init__.py
        ├── config.py           # Единый файл с гиперпараметрами (размер словаря, d_model, слои)
        ├── data/
        │   ├── __init__.py
        │   ├── generator.py    # Скрипт генерации 250k примеров сложения/вычитания
        │   └── tokenizer.py    # Character-level токенизатор + маппинг спецтокенов (<c0>, <b1> и т.д.)
        ├── model/
        │   ├── __init__.py
        │   ├── transformer.py  # Сборка трансформера с поддержкой хуков
        │   └── components.py   # Отдельные классы: Attention, MLP, Positional Encoding
        ├── training/
        │   ├── __init__.py
        │   └── engine.py       # Цикл обучения (train_step, eval_step), логирование
        └── interpretability/
            ├── __init__.py
            ├── hooks.py        # Инфраструктура для извлечения/подмены активаций
            ├── probing.py      # Линейные классификаторы для скрытых состояний
            └── attribution.py  # Integrated Gradients и логика для Logit Lens

```

## Установка и запуск

Проект использует менеджер `uv` для детерминированного и быстрого управления окружением. PyTorch и тензорные операции автоматически используют CUDA при наличии GPU (на сервере) или fallback на CPU (при локальной разработке в WSL).

### 1. Клонирование

```bash
git clone [https://github.com/](https://github.com/)<USERNAME>/grokking-carries.git
cd grokking-carries

```

### 2. Установка `uv` (если не установлен)

Для Linux / WSL:

```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

```

### 3. Инициализация окружения

Разворачивает виртуальное окружение `.venv` и устанавливает зафиксированные зависимости из `uv.lock`.

```bash
uv sync

```

### 4. Использование

Все команды исполняются строго через `uv run` для привязки к локальному окружению проекта.

Запуск скриптов:

```bash
uv run scripts/01_generate_data.py
uv run scripts/02_run_training.py

```

Запуск исследовательской среды (Jupyter):

```bash
uv run jupyter notebook

```

