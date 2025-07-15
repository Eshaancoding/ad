from dataclasses import dataclass
from colored import stylize, fore

############################################
## Value
class Value:
    pass

@dataclass
class Constant(Value):
    val: int
    
    def __repr__(self):
        return stylize(str(self.val), fore("green"))
    
    def get_const (self):
        return self.val

class Global(Value): 
    def __repr__(self):
        return stylize("Global", fore("green"))
    
class X(Value): 
    def __repr__(self):
        return stylize("X", fore("green"))

class Y(Value): 
    def __repr__(self):
        return stylize("Y", fore("green"))

############################################
## Expressions
class Expression:
    def get_const (self) -> int:
        raise TypeError("Expression is not a const")

    pass

class NoneExpr (Expression):
    def __repr__(self):
        return stylize("None", fore("green"))

# Leaf node
@dataclass
class Val(Expression):
    v: Value

    def __repr__(self):
        return self.v.__repr__()
    
    def get_const (self) -> int:
        return self.v.get_const()

@dataclass
class Add(Expression):
    a: Expression
    b: Expression
    
    def __repr__(self):
        return f"{self.a} + {self.b}"

@dataclass
class Minus(Expression):
    a: Expression
    b: Expression
    
    def __repr__(self):
        return f"{self.a} - {self.b}"

@dataclass
class Mult(Expression):
    a: Expression
    b: Expression

    def __repr__(self):
        return f"{self.a} * {self.b}"

@dataclass
class Div(Expression):
    a: Expression
    b: Expression
    
    def __repr__(self):
        return f"{self.a} / {self.b}"

@dataclass
class Remainder(Expression):
    a: Expression
    b: Expression

    def __repr__(self):
        return f"{self.a} % {self.b}"

@dataclass
class ShiftRight(Expression):
    a: Expression
    b: Expression
    
    def __repr__(self):
        return f"{self.a} << {self.b}"

@dataclass
class ShiftLeft(Expression):
    a: Expression
    b: Expression
    
    def __repr__(self):
        return f"{self.a} >> {self.b}"

@dataclass
class BitwiseAnd(Expression):
    a: Expression
    b: Expression

    def __repr__(self):
        return f"{self.a} & {self.b}"

@dataclass
class MoreThan(Expression):
    a: Expression
    b: Expression

    def __repr__(self):
        return f"{self.a} > {self.b}"

@dataclass
class LessThan(Expression):
    a: Expression
    b: Expression

    def __repr__(self):
        return f"{self.a} < {self.b}"
    