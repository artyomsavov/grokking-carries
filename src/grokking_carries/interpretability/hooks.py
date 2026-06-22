import contextlib
from typing import Generator
import torch.nn as nn
from torch import Tensor
from grokking_carries.types import HookName, HookFunction

class HookPoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.name: HookName | None = None
        self._hooks: list[HookFunction] = []

    def set_name(self, name: HookName) -> None:
        self.name = name

    def add_hook(self, hook: HookFunction) -> None:
        self._hooks.append(hook)

    def remove_hook(self, hook: HookFunction) -> None:
        if hook in self._hooks:
            self._hooks.remove(hook)

    def clear_hooks(self) -> None:
        self._hooks.clear()

    def forward(self, x: Tensor) -> Tensor:
        if not self._hooks:
            return x
        for hook in self._hooks:
            result = hook(x, self.name)
            if result is not None:
                x = result
        return x

class HookedRootModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.mod_dict: dict[HookName, HookPoint] = {}
        self._is_setup = False

    def setup_hooks(self) -> None:
        self.mod_dict.clear()
        for name, module in self.named_modules():
            if isinstance(module, HookPoint):
                module.set_name(name)
                self.mod_dict[name] = module
        self._is_setup = True

    def add_hook(self, name: HookName, hook: HookFunction) -> None:
        if not self._is_setup:
            self.setup_hooks()
        if name not in self.mod_dict:
            raise KeyError(f"HookPoint с именем '{name}' не найден.")
        self.mod_dict[name].add_hook(hook)

    def remove_all_hooks(self) -> None:
        for hp in self.mod_dict.values():
            hp.clear_hooks()

    @contextlib.contextmanager
    def hooks(self, fwd_hooks: list[tuple[HookName, HookFunction]]) -> Generator[None, None, None]:
        try:
            for name, hook in fwd_hooks:
                self.add_hook(name, hook)
            yield
        finally:
            for name, hook in fwd_hooks:
                if name in self.mod_dict:
                    self.mod_dict[name].remove_hook(hook)
