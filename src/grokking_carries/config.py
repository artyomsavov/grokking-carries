import torch
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelConfig:
    n_layers: int = 4
    n_heads: int = 8
    d_model: int = 256
    d_mlp: int = 1024  # 4 * d_model
    max_seq_len: int = 64
    vocab_size: int = 19  # 10 цифр + 4 спецсимвола + 4 CoT + 1 pad
    
    # Автоопределение устройства (AMD CPU локально / Nvidia GPU на сервере)
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_heads == 0, 'd_model must be divisible by n_heads'
        return self.d_model // self.n_heads
        
    def to_device(self, new_device: str) -> 'ModelConfig':
        return replace(self, device=new_device)

@dataclass(frozen=True, slots=True)
class TrainConfig:
    batch_size: int = 128
    lr: float = 1e-3
    train_size: int = 240_000
    test_size: int = 10_000
    checkpoint_freq: int = 50 # Частота сохранения для анализа динамики

