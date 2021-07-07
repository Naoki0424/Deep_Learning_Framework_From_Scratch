import unittest
import sys
sys.path.append(r'../')
from stage2 import *
import numpy as np

class CalclateTest(unittest.TestCase):
    def test_add_normal(self):
        # 計算式と値を設定する
        x0 = Variable(np.array(5))
        x1 = Variable(np.array(7))
        y = add(x0, x1)
        expected = 12
        self.assertEqual(expected, y.data, msg='expected：{0}、y.data：{1}は違います'.format(expected, y.data))

    def test_square_normal(self):
        # 計算式と値を設定する
        x = Variable(np.array(5))
        y = square(x)
        expected = 25
        self.assertEqual(expected, y.data, msg='expected：{0}、y.data：{1}は違います'.format(expected, y.data))

    def test_add_square_normal(self):
        # 計算式と値を設定する
        x0 = Variable(np.array(2))
        x1 = Variable(np.array(3))
        y = add(square(x0), square(x1))
        expected = 13
        self.assertEqual(expected, y.data, msg='expected：{0}、y.data：{1}は違います'.format(expected, y.data))