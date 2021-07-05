import numpy as np

class Variable:
    '変数を保持するクラス'
    def __init__(self, data):
        # パラメータのdataはndarray型のみ許可する
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        
        # 順伝播の値
        self.data = data
        # 逆伝播で計算された値（勾配）
        self.gradient = None
        # 関数（生みの親）
        self.creator = None
    
    # 生みの親を設定する
    def set_creator(self, func):
        self.creator = func
    
    # 逆伝播を行う
    def backward(self):
        # 逆伝播で計算された値がない時は1.0を設定する。
        # この時、形状とデータ型は順伝播の値に合わせる
        if self.gradient is None:
            self.gradient = np.ones_like(self.data)
        # 関数（生みの親）を取得
        funcs = [self.creator]
        # 生みの親に対して逆伝播を行う
        while funcs:
            # 生みの親を取得
            f = funcs.pop()
            # 生みの親に対する入力と出力を取得する
            x, y = f.input, f.output
            # 逆伝播を行う
            x.gradient = f.backward(y.gradient)
            # 生みの親の入力にさらに生みの親が存在する場合は逆伝播対象として追加する
            if x.creator is not None:
                funcs.append(x.creator)


class Function:
    '関数の親クラス'
    # __call__はPythonの特殊メソッド
    # f = Function()としたときにf()で__call__を呼び出すことができる
    def __call__(self, input):
        # 入力された値を用いて計算を行う
        output = Variable(as_array(self.forward(input.data)))
        # 出力が決定したときに生みの親を設定する
        output.set_creator(self)
        # 入力された値を記録しておく。これは逆伝播の（勾配を求める）計算に利用する。
        self.input = input
        # 出力も記憶しておく
        self.output = output
        # 計算結果を返却する
        return output

    # 順伝播の計算。計算ロジックを子クラスで実装する
    # 小クラスでforwardが実装されていない時は明示的にErrorを発生させる
    def forward(self, x):
        raise NotImplementedError()

    # 逆伝播の計算。計算ロジックを子クラスで実装する
    # 小クラスでbackwardが実装されていない時は明示的にErrorを発生させる
    def backward(self, x):
        raise NotImplementedError()

class Square(Function):
    '2乗の計算を行う'
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        # y = x**2のxに関するyの微分は2xのため
        # その値と出力方向から伝播された値をかける
        x = self.input.data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    'eのx乗を計算するクラス'
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

def numerical_diff(f, x, exp = 1e-4):
    y1 = f(Variable(as_array(x.data - exp)))
    y2 = f(Variable(as_array(x.data + exp)))
    output = (y2.data - y1.data) / (2 * exp)
    return output

def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x