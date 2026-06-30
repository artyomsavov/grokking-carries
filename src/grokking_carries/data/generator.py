from typing import Literal
from torch.utils.data import DataLoader, Dataset
from grokking_carries.data.tokenizer import ArithmeticTokenizer


class ArithmeticDataset(Dataset):
    def __init__(self, data: list[str], tokenizer: ArithmeticTokenizer) -> None:
        ...

class EquationGenerator:
    def __init__(self, num_digits: int = 5) -> None:
        ...
        
    def generate_raw_data(self, size: int) -> list[str]:
        '''Генерирует массив строк со сложением/вычитанием и CoT разметкой.'''
        ...

    def build_loader(
        self, 
        size: int, 
        batch_size: int, 
        tokenizer: ArithmeticTokenizer,
        shuffle: bool = True
    ) -> DataLoader:
        ...

