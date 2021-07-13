import numpy as np
import weakref
import contextlib

class Variable:
    '変数を保持するクラス'

    # 演算の優先順位
    __array__priority__ = 200

    def __init__(self, data, name = None):
        '初期化を行う'
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
        # 世代（逆伝播の求める順番）
        self.generation = 0
        # 変数の名前
        self.name = name

    def __len__(self):
        '要素数を求める'
        return len(self.data)

    def __repr__(self):
        'print(variable)で出力する内容'
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    def __mul__(self, other):
        '掛け算'
        return mul(self, other)

    def __add__(self, other):
        '足算'
        return add(self, other)

    def __rmul__(self, other):
        '''掛け算。
        
        a * b のaに__mul__がない時はbの__rmulが呼ばれる'''
        return mul(self, other)

    def __radd__(self, other):
        '''足算。
        
        a * b のaに__add__がない時はbの__raddが呼ばれる'''
        return add(self, other)

    def __neg__(x):
        '負数'
        return neg(x)

    def __sub__(self, other):
        '引き算'
        return sub(self, other)
    
    def __rsub__(self, other):
        '引き算'
        return sub(other, self)

    def __truediv__(self, other):
        '引き算'
        return div(self, other)
    
    def __rtruediv__(self, other):
        '引き算'
        return div(other, self)

    def __pow__(self, other):
        '累乗'
        return pow(self, other)

    def set_creator(self, func):
        '生みの親を設定する'
        self.creator = func
        self.generation = func.generation + 1
    
    def backward(self, retain_gradient=False):
        '逆伝播を行う'
        # 逆伝播で計算された値がない時は1.0を設定する。
        # この時、形状とデータ型は順伝播の値に合わせる
        if self.gradient is None:
            self.gradient = np.ones_like(self.data)
        # 関数（生みの親）を取得しながら、世代の実行順を計算する
        funcs = [] # 世代順に並び替えられた生みの親のリスト
        seen_set = set() # 生みの親の重複を排除するための集合。集合の初期化にはset()を用いる。a = {}ってやると辞書になる
        # 関数の中に関数の定義を行うことで親のメソッドないの変数にアクセスできる。ここでいえばfuncsとseen_set
        def add_func(f):
            # 重複していない場合は生みの親を追加する
            if f not in seen_set:
                # 世代順に並び替えられた生みの親のリストに追加
                funcs.append(f)
                # 重複チェックに使用する集合に追加
                seen_set.add(f)
                # 世代順に並び替える
                funcs.sort(key=lambda x: x.generation)
        # 今の出力（順伝播時の）の生みの親を設定する
        add_func(self.creator)

        # 生みの親に対して逆伝播を行う
        while funcs:
            # 生みの親を取得
            f = funcs.pop()
            # 出力値を取得（逆伝播で見たら入力値）。outputsの要素は弱参照なのでoutput()じゃないとだめ
            gys = [output().gradient for output in f.outputs]
            # 逆伝播実施
            gxs = f.backward(*gys)
            # 上記の結果がタプルでない場合はタプルに変換する
            if not isinstance(gxs, tuple):
                gxs = (gxs, )
            # 取得した微分を設定する
            # 例：zip((Variable1, Variable2, Variable3), (3.0, 4.0, 5.0)) = ((Variable1, 3.0), (Variable2, 4.0), (Variable3, 5.0))
            for x, gx in zip(f.inputs, gxs):
                # 関数への入力値に微分が設定されていない場合は微分を設定する
                if x.gradient is None:
                    x.gradient = gx
                # 既に微分が設定されている場合は足算を行う
                else:
                    x.gradient = x.gradient + gx
                # 生みの親の入力にさらに生みの親が存在する場合は逆伝播対象として追加する
                if x.creator is not None:
                    add_func(x.creator)
            # retain_gradient = Falseのとき、中間の変数は微分を保持しない
            if not retain_gradient:
                for y in f.outputs:
                    y().gradient = None

    def cleargradient(self):
        self.gradient = None

    @property
    def shape(self):
        '''形状
        
        np.array([[1, 2], [1, 2], [2, 2]]) => (3, 2)'''
        return self.data.shape

    @property
    def ndim(self):
        '''次元数
        
        np.array([[1, 2], [1, 2], [2, 2]]) => 2'''
        return self.data.ndim

    @property
    def size(self):
        '''要素数
        
        np.array([[1, 2], [1, 2], [2, 2]]) => 6'''
        return self.data.size

    @property
    def dtype(self):
        '型'
        return self.data.dtype

class Function:
    '関数の親クラス'
    def __call__(self, *inputs):
        '''
        __call__はPythonの特殊メソッド
        f = Function()としたときにf()で__call__を呼び出すことができる
        '''
        # 入力値をVariableに変換
        inputs = [as_variable(x) for x in inputs]
        # 入力値を全て取り出し配列に保持する
        xs = [x.data for x in inputs]
        # 入力された値を用いて計算を行う
        ys = self.forward(*xs)
        # 上の返却値がタプルではない時はタプルに変換する
        if not isinstance(ys, tuple):
            ys = (ys,)
        # Variable型への変換。スカラ値を考慮しながら（as_arrayにて）出力値を設定する
        outputs = [Variable(as_array(y)) for y in ys]
        # メモリの効率的使用のため、逆伝播の利用に応じて変数の設定を行う
        if Configuration.enable_backdrop:
            # 入力値と同じ世代を設定する
            self.generation = max([x.generation for x in inputs])
            # 出力が決定したときに生みの親を設定する
            for output in outputs:
                output.set_creator(self)
            # 入力された値を記録しておく。これは逆伝播の（勾配を求める）計算に利用する。
            self.inputs = inputs
            # 出力も記憶しておく。
            # 関数のoutputと変数のcreaterで循環参照が発生している。メモリ効率を考え関数のoutputは弱参照（weakref.ref()を使用する）にする
            # 弱参照とは参照カウントを増やさずに参照を行う機能（CPythonの場合）
            # 参照カウントはメモリ管理に使われる数字で格オブジェクトに割り振られる。参照カウントが1から0になったときにそのオブジェクトを削除しメモリを開放する
            self.outputs = [weakref.ref(output) for output in outputs]
        # 計算結果を返却する。返却値のタプルのサイズが1より大きくない場合は最初の要素のみ返却する
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        '''
        順伝播の計算。計算ロジックを子クラスで実装する
        子クラスでforwardが実装されていない時は明示的にErrorを発生させる
        '''
        raise NotImplementedError()

    def backward(self, x):
        '''
        逆伝播の計算。計算ロジックを子クラスで実装する
        小クラスでbackwardが実装されていない時は明示的にErrorを発生させる
        '''
        raise NotImplementedError()

class Square(Function):
    '2乗の計算を行う'
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        # y = x**2のxに関するyの微分は2xのため
        # その値と出力方向から伝播された値をかける
        x = self.inputs[0].data
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

class Mul(Function):
    def forward(self, x0, x1):
        return x0 * x1
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return x1 * gy, x0 * gy

class Neg(Function):
    def forward(self, x):
        return -x
    
    def backward(self, gy):
        return -gy

class Sub(Function):
    def forward(self, x0, x1):
        return x0 - x1
    
    def backward(self, gy):
        return gy, -gy

class Div(Function):
    def forward(self, x0, x1):
        return x0 / x1
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data 
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1

class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        return x ** self.c
    
    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx

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

def mul(x0, x1):
    return Mul()(x0, x1)

def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)

def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)

def pow(x, c):
    return Pow(c)(x)

def neg(x):
    return Neg()(x)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(np.array(obj))

def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)

class Configuration:
    '逆伝播を行うかの設定'
    # True:逆伝播を実施する（学習モード）、False:逆伝播は行わない（推論モード）
    enable_backdrop = True

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Configuration, name)
    setattr(Configuration, name, value)
    try:
        yield
    finally:
        setattr(Configuration, name, old_value)

def no_grad():
    return using_config('enable_backdrop', False)
