from typing import Callable, List

from autodiff.expr import *
from autodiff.kernalize import KMatrix
from ...node import Node

class Feeder (Node):
    __match_args__ = ("func", "children", "name")

    # we don't support dynamic shapes at the moment
    # name must be unique
    def __init__ (self, func:Callable, shape: List[int], name:str=""):
        super().__init__([], shape, None)
        self.func = func
        self.name = name

    def bck (self, _:Node):
        pass # no backward on feeder node

    def __repr__(self) -> str:
        if self.kres.is_none():
            return f"{self.id} = Feeder \"{self.name}\" (dim: {self.shape})"
        else:
            return f"{self.kres} = Feeder \"{self.name}\" (dim: {self.shape})"

    def node_eq (self, other) -> bool:
        if not isinstance(other, Feeder):
            return False

        return self.id == other.id and self.shape == other.shape 
