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

    def _bck (self, _:Node):
        return None # no backward on feeder node

    def __repr__(self) -> str:
        op_str = stylize("Feeder", fore("turquoise_2"))
        if self.kres.is_none():
            return f"{self.id} = {op_str} \"{self.name}\" (dim: {self.shape})"
        else:
            return f"{self.kres} = {op_str} \"{self.name}\" (dim: {self.shape})"

    def repeat_helper(self, is_child):
        return (
            "Receiver",
            self.id
        )
