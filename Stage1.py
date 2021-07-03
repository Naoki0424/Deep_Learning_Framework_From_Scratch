import numpy as np

class Variable:
    '変数を保持するクラス'
    def __init__(self, data):
        self.data = data

class Function:
    '関数の親クラス'
    # __call_はPythonの特殊メソッド
    # f = Function()としたときにf()で__call__を呼び出すことができる
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    # 計算ロジックを子クラスで実装する
    # 小クラスでforwardが実装されていない時は明示的にErrorを発生させる
    def forward(self, x):
        raise NotImplementedError()

class Square(Function):
    '2乗の計算を行う'
    def forward(self, x):
        return x ** 2

class Exp(Function):
    'eのx乗を計算するクラス'
    def forward(self, x):
        return np.exp(x)

def numerical_deff(f, x, exp = 1e-4):
    y1 = f(Variable(x.data - exp))
    y2 = f(Variable(x.data + exp))
    output = (y2.data - y1.data) / (2 * exp)
    return output
