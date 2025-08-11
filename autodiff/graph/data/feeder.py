from typing import Callable, List

from autodiff.expr import *
from autodiff.kernalize import KMatrix
from ...node import Node

class Feeder (Node):
    __match_args__ = ("func", "children", "name")

    # we don't support dynamic shapes at the moment
    # name must be unique
    def __init__ (self, func:Callable, shape: List[int], name:str):
        super().__init__([], shape, None)
        self.func = func
        self.name = name

        # filled at kernalize/device
        self.kres_id = None
        self.offset = 0 

    def assert_offset (self):
        assert isinstance(self.kres, KMatrix), "Kres is not matrix?"
        match self.kres.access:
            case Global():
                assert self.offset == 0, "Assert offset false at Global()"
            case Add(Global(), Val(Constant(_ as c))):
                assert self.offset == c, "Assert offset false at Add(..., ...)"
            case _:
                print(self.kres)
                raise Exception("Invalid Feeder setting")

    def bck (self, _:Node):
        pass # no backward on feeder node

    def __repr__(self) -> str:
        if self.kres.is_none():
            return f"{self.id} = Feeder (dim: {self.shape})"
        else:
            return f"{self.kres} (should be offset: {self.offset}) = Feeder (dim: {self.shape})"

    def node_eq (self, other) -> bool:
        if not isinstance(other, Feeder):
            return False

        return self.name == other.name and self.shape == other.shape 
