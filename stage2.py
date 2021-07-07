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
            # 出力値を取得（逆伝播で見たら入力値）
            gys = [output.gradient for output in f.output]
            # 逆伝播実施
            gxs = f.backward(*gys)
            # 上記の結果がタプルでない場合はタプルに変換する
            if not isinstance(gxs, tuple):
                gxs = (gxs, )
            # 取得した微分を設定する
            # 例：zip((Variable1, Variable2, Variable3), (3.0, 4.0, 5.0)) = ((Variable1, 3.0), (Variable2, 4.0), (Variable3, 5.0))
            for x, gx in zip(f.inputs, gxs):
                x.gradient = gx
                # 生みの親の入力にさらに生みの親が存在する場合は逆伝播対象として追加する
                if x.creator is not None:
                    funcs.append(x.creator)


class Function:
    '関数の親クラス'
    # __call__はPythonの特殊メソッド
    # f = Function()としたときにf()で__call__を呼び出すことができる
    def __call__(self, *inputs):
        # 入力値を全て取り出し配列に保持する
        xs = [x.data for x in inputs]
        # 入力された値を用いて計算を行う
        ys = self.forward(*xs)
        # 上の返却値がタプルではない時はタプルに変換する
        if not isinstance(ys, tuple):
            ys = (ys,)
        # Variable型への変換。スカラ値を考慮しながら（as_arrayにて）出力値を設定する
        outputs = [Variable(as_array(y)) for y in ys]
        # 出力が決定したときに生みの親を設定する
        for output in outputs:
            output.set_creator(self)
        # 入力された値を記録しておく。これは逆伝播の（勾配を求める）計算に利用する。
        self.inputs = inputs
        # 出力も記憶しておく
        self.outputs = outputs
        # 計算結果を返却する。返却値のタプルのサイズが1より大きくない場合は最初の要素のみ返却する
        return outputs if len(outputs) > 1 else outputs[0]

    # 順伝播の計算。計算ロジックを子クラスで実装する
    # 小クラスでforwardが実装されていない時は明示的にErrorを発生させる
    def forward(self, xs):
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
        x = self.inputs.data
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

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy

def numerical_diff(f, x, exp = 1e-4):
    '数値微分を行う関数。これは逆伝播の結果と比較する際に利用する'
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

def add(x0, x1):
    return Add()(x0, x1)