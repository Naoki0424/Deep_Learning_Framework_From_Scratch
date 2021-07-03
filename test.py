import numpy as np
from Stage1 import *

test1 = Variable(np.array(2.0))
print(test1.data)

f1 = Square()
f2 = Exp()

a1 = f1(test1)
a2 = f2(a1)
a3 = f1(a2)
print(a3.data)

def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))


test2 = numerical_deff(f, Variable(np.array(0.5)))
print(test2)