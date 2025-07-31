from ..node import Node
from typing import List, Tuple

class Module:
    def __init__(self):
        self._parameters: List[Node] = []
        self._modules: List[Module] = []

    def _add_to_params (self, l: List[Node]):
        current_ids = {i.id:1 for i in self._parameters}
        for n in l:
            if not (n.id in current_ids):
                self._parameters.append(n)

    def __setattr__(self, name, value):

        if isinstance(value, Node):
            self._add_to_params([value])
        elif isinstance(value, List) and len(value) > 0 and isinstance(value[0], Node):
            self._add_to_params(value)
        elif isinstance(value, Tuple) and len(value) > 0 and isinstance(value[0], Node):
            self._add_to_params(value)

        elif isinstance(value, Module):
            self._modules.append(value)
        elif isinstance(value, List) and len(value) > 0 and isinstance(value[0], Module):
            self._modules.extend(value)
        elif isinstance(value, Tuple) and len(value) > 0 and isinstance(value[0], Module):
            self._modules.extend(value)

        super().__setattr__(name, value)

    def parameters(self):
        for m in self._modules:
            self._add_to_params(m.parameters())

        return self._parameters

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Node:
        raise NotImplementedError("You must override forward()")
