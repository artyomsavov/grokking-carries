# Grokking Carries

Исследование механистической интерпретируемости (mechanistic interpretability) арифметических операций в малых GPT-подобных моделях. Проект нацелен на анализ внутренних представлений числовой линии и алгоритмов переноса разрядов (carries/borrows) с использованием probing, causal tracing и activation patching.

## Структура репозитория

```text
grokking-carries/
├── notebooks/                  # Jupyter-ноутбуки для SVD, Logit Lens и визуализаций
├── scripts/                    # Исполняемые скрипты (pipeline)
│   ├── 01_generate_data.py     
│   └── 02_run_training.py      
└── src/grokking_carries/
    ├── config.py               # Единый конфигуратор (архитектура, гиперпараметры, device)
    ├── data/                   # Генератор CoT-датасета и character-level токенизатор
    ├── model/                  # Архитектура трансформера, спроектированная под систему хуков
    ├── training/               # Цикл обучения с частым чекпоинтингом для отлова grokking
    └── interpretability/       # Инфраструктура извлечения/подмены активаций и атрибуции

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

