import math

class Value:
    def __init__(self, data, children=()):
        self.data = data
        self.prev = children
        self.grad = 0.0
        self.backward = lambda : None

    def __repr__(self):
        return f'Value(data={self.data}, grad={self.grad}))'

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out.backward = _backward

        return out

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out.backward = _backward

        return out
    
    def __truediv__(self, other):
        return self * other**-1

    def __pow__(self, other):
        out = Value(self.data**other, (self,))

        def _backward():
            self.grad += other * (self.data**(other - 1)) * out.grad
        out.backward = _backward

        return out

    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1
    
    def tanh(self):
        out = Value(math.tanh(self.data), (self,))

        def _backward():
            self.grad += (1 - math.tanh(self.data)**2) * out.grad
        out.backward = _backward

        return out
    
    def backprop(self):
        self.grad = 1.0

        def backward_recursion(parent_node):
            parent_node.backward()
            for node in parent_node.prev:
                backward_recursion(node)
        
        backward_recursion(self)