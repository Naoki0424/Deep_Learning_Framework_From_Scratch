import unittest
import sys
sys.path.append(r'../')
from stage1 import *
import numpy as np

class Stage1Test(unittest.TestCase):
    def test_gradient_check(self):
        # 計算式と値を設定する
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()

        expected = numerical_diff(square, x)


        flg = np.allclose(x.gradient, expected)

        self.assertTrue(flg)
