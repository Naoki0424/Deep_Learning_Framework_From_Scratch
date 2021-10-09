from dezero import utils
import numpy as np
from numpy.core.fromnumeric import reshape
from dezero.core import Function
from dezero.core import as_variable 

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

class Reshape(Function):
    def __init__(self, shape) :
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)

def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)

    return Reshape(shape)(x)

class Transpose(Function):
    def forward(self, x):
        return np.transpose(x)

    def backward(self, gy):
        return transpose(gy)

def transpose(x):
    return Transpose()(x)

class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.shape, self.axis, self.keepdims)
        gx = np.broadcast_to(gy, self.x_shape)
        return gx

def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)

class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx

def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x) 

class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)