import numpy as np
from dezero.core import Function

class Sin(Function):
    def forward(self, x):
        return np.sin(x)
    
    def backward(self, gy):
        x = self.inputs[0]
        return cos(x) * gy

def sin(x):
    return Sin()(x)

class Cos(Function):
    def forward(self, x):
        return np.cos(x)
    
    def backward(self, gy):
        x = self.inputs[0]
        return -sin(x) * gy

def cos(x):
    return Cos()(x)

class Tanh(Function):
    def forward(self, x):
        return np.tanh(x)
    
    def backward(self, gy):
        return (1 - self.outputs[0]() ** 2) * gy

def tanh(x):
    return Tanh()(x)