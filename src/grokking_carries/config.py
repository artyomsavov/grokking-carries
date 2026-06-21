import json
import torch
from dataclasses import dataclass, replace, asdict
from pathlib import Path

@dataclass(frozen=True, slots=True)
class ModelConfig:
    n_layers: int = 4
    n_heads: int = 4        # Снижено до 4 для четкой локализации алгоритмов
    d_model: int = 256
    d_mlp: int = 1024       # Явно задаем 4 * d_model
    max_seq_len: int = 64
    vocab_size: int = 19    # 10 (цифры) + 4 (спецсимволы) + 4 (CoT) + 1 (pad)
    
    # Индексы токенов (обязательно для Probing и генерации)
    pad_token_id: int = 18
    
    seed: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_heads == 0, 'd_model must be divisible by n_heads'
        return self.d_model // self.n_heads

    def to_device(self, new_device: str) -> 'ModelConfig':
        return replace(self, device=new_device)

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def load(cls, path: str | Path) -> 'ModelConfig':
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

@dataclass(frozen=True, slots=True)
class TrainConfig:
    batch_size: int = 1024      # Увеличено для стабилизации градиентов
    lr: float = 1e-3
    weight_decay: float = 0.1   # Агрессивный WD (Вариант 1) для триггера грокинга
    warmup_steps: int = 500     # Warmup для защиты от выгорания нейронов на старте
    
    train_size: int = 240_000
    test_size: int = 10_000
    
    # Динамическое логирование: сохраняем реже в начале, 
    # в кастомном цикле обучения (engine.py) можно будет сделать экспоненциальным
    checkpoint_freq_epochs: int = 10

