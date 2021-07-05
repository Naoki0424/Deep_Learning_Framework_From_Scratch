import numpy as np
from stage1 import *

def f():
    x = Variable(np.array(0.5))

    a = square(x)
    b = exp(a)
    y = square(b)
    y.backward()

    return x.gradient

def test(x):
    return square(exp(square(x)))

test2 = numerical_diff(test, Variable(np.array(0.5)))
print('数値微分', test2)

print('逆伝播', f())