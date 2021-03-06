import os
import subprocess

def _dot_var(v, verbose=False):
    '変数用出力用のテキストを取得する'

    # テンプレートテキスト
    dot_var = '{}[label="{}", color=orange, style=filled]\n'
    # 変数に設定されている名称を取得
    name = '' if v.name is None else v.name
    # 形状とデータ型の追記を行う。デフォルト設定ではこの操作はスキップされる
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.dtype)
    # テキストに変数のIDと名称を設定し返却する
    return dot_var.format(id(v), name)

def _dot_func(f):
    '関数出力用のテキストを取得する'

    # 関数用のテンプレートテキスト
    dot_var = '{}[label="{}", color=lightblue, style=filled, shape=box]\n'
    # IDと関数名を設定する
    txt = dot_var.format(id(f), f.__class__.__name__)
    # 紐付け用のテンプレートテキスト
    dot_edge = '{} -> {}\n'
    # 関数の入力に対して紐付けを作成する
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    # 関数の出力に対して紐付けを作成する
    for y in f.outputs:
        # yは弱参照のため()をつける
        txt += dot_edge.format(id(f), id(y()))
    return txt

def get_dot_graph(output, verbose=True):
    'dotグラフ出力用のテキストを作成する'

    # 結果
    txt = ''
    # 関数
    funcs = []
    # 関数の重複確認用
    seen_set = set()
    # 関数を追加する関数
    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)
    # 出力の生みの親（関数）を設定する
    add_func(output.creator)
    # 同時に結果にDotグラフ出力用のテキストをつめる（出力）
    txt += _dot_var(output)
    # 関数分ループする
    while funcs:
        # 末尾の関数を取得する
        func  = funcs.pop()
        # 結果にDotグラフ出力用のテキストをつめる（関数分）
        txt += _dot_func(func)
        # 関数の入力分ループ
        for x in func.inputs:
            # Dotグラフ出力用のテキストをつめる（入力分）
            txt += _dot_var(x)
            # 入力の生みの親（関数）が存在する場合は追加を行う
            if x.creator is not None:
                add_func(x.creator)
    # 結果の返却を行う
    return 'digraph g {\n' + txt + '}'

def plot_dot_graph(output, verbose=True, to_file='dot/graph.png'):
    'グラフの描画を行う'
    
    # Dotグラフ表示用のテキストを取得する
    dot_graph = get_dot_graph(output, verbose)
    
    # １.dotデータをファイルに保存
    tmp_dir = os.path.join(os.path.expanduser('~'), '.dezero')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')

    with open(graph_path, 'w') as f:
        f.write(dot_graph)

    # ２.dotコマンドを実行
    extensinon = os.path.splitext(to_file)[1][1:]
    cmd = 'dot {} -T {} -o {}'.format(graph_path, extensinon, to_file)
    subprocess.run(cmd, shell=True)


# =============================================================================
# Utility functions for numpy (numpy magic)
# =============================================================================
def sum_to(x, shape):
    """Sum elements along axes to output an array of a given shape.

    Args:
        x (ndarray): Input array.
        shape:

    Returns:
        ndarray: Output array of the shape.
    """
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y


def reshape_sum_backward(gy, x_shape, axis, keepdims):
    """Reshape gradient appropriately for dezero.functions.sum's backward.

    Args:
        gy (dezero.Variable): Gradient variable from the output by backprop.
        x_shape (tuple): Shape used at sum function's forward.
        axis (None or int or tuple of ints): Axis used at sum function's
            forward.
        keepdims (bool): Keepdims used at sum function's forward.

    Returns:
        dezero.Variable: Gradient variable which is reshaped appropriately
    """
    ndim = len(x_shape)
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)

    if not (ndim == 0 or tupled_axis is None or keepdims):
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        shape = gy.shape

    gy = gy.reshape(shape)  # reshape
    return gy


def logsumexp(x, axis=1):
    xp = cuda.get_array_module(x)
    m = x.max(axis=axis, keepdims=True)
    y = x - m
    xp.exp(y, out=y)
    s = y.sum(axis=axis, keepdims=True)
    xp.log(s, out=s)
    m += s
    return m


def max_backward_shape(x, axis):
    if axis is None:
        axis = range(x.ndim)
    elif isinstance(axis, int):
        axis = (axis,)
    else:
        axis = axis

    shape = [s if ax not in axis else 1 for ax, s in enumerate(x.shape)]
    return shape
