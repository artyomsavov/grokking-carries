import re
import torch
from typing import Sequence
from grokking_carries.types import TokenIndices


class ArithmeticTokenizer:
    def __init__(self) -> None:
        tokens = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            '+', '-', '=', 
            '<c0>', '<c1>', '<b0>', '<b1>', 
            '<pad>', '<unk>' # Добавим <unk> на случай мусора и добьем до 19
        ]
        
        self.vocab: dict[str, int] = {token: idx for idx, token in enumerate(tokens)}
        self.inverse_vocab: dict[int, str] = {idx: token for token, idx in self.vocab.items()}
        self.pad_token_id = self.vocab['<pad>']
        self.unk_token_id = self.vocab['<unk>']

    def encode(self, texts: Sequence[str], pad_to_max: bool = True) -> TokenIndices:
        """
        Превращает список строк-уравнений в тензор индексов.
        Ожидает строки, где токены разделены пробелами, например:
        "0 0 0 4 5 + 0 0 0 9 2 ="
        """

        encoded_batch = []
        max_len = 0
        
        for text in texts:
            tokens = text.strip().split()
            encoded_seq = [self.vocab.get(t, self.unk_token_id) for t in tokens]
            encoded_batch.append(encoded_seq)
            if len(encoded_seq) > max_len:
                max_len = len(encoded_seq)
                
        if pad_to_max:
            for seq in encoded_batch:
                padding_length = max_len - len(seq)
                seq.extend([self.pad_token_id] * padding_length)
                
        return torch.tensor(encoded_batch, dtype=torch.long)

    def decode(self, token_ids: TokenIndices) -> list[str]:
        """Превращает тензор индексов обратно в список строк."""

        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
            
        token_ids_list = token_ids.tolist()
        decoded_batch = []
        
        for seq in token_ids_list:
            tokens = [self.inverse_vocab.get(idx, '<unk>') for idx in seq if idx != self.pad_token_id]
            decoded_batch.append(" ".join(tokens))
            
        return decoded_batch

class MathTokenizer:
    def __init__(self):
        # 10 цифр + 4 спецсимвола + 4 CoT + 1 pad = 19 токенов (как в ModelConfig)
        self.vocab = [str(i) for i in range(10)] + ['+', '-', '=', ' ', '<c0>', '<c1>', '<b0>', '<b1>', '<pad>']
        self.v2i = {v: i for i, v in enumerate(self.vocab)}
        self.i2v = {i: v for i, v in enumerate(self.vocab)}
        self.pad_token_id = self.v2i['<pad>']
        self.vocab_size = len(self.vocab)

    def encode(self, text: str, from_split: bool = False) -> list[int]:
        """
        Если from_split=True, разбиваем по пробелам (как при обучении).
        Если False, используем регулярное выражение для сохранения пробелов (инференс).
        """
        if from_split:
            tokens = text.split()
        else:
            # Регулярка для вытаскивания CoT токенов целиком, либо отдельных символов
            tokens = re.findall(r'<[cb][01]>|<pad>|\d|\+|-|=| ', text)
        return [self.v2i[t] for t in tokens if t in self.v2i]

    def decode(self, ids: list[int]) -> str:
        return " ".join([self.i2v[i] for i in ids])
