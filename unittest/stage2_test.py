import unittest
import sys
sys.path.append(r'../')
from stage2 import *
import numpy as np

class CalclateTest(unittest.TestCase):
    def test_add_multiple(self):
        '足算と掛け算'
        x0 = Variable(np.array(2.0))
        y = x0 * np.array(5.0)
        y.backward()
        self.assertEqual(10, y.data)
        self.assertEqual(5, x0.gradient)

        x1 = Variable(np.array(3.0))
        y = 2.0 * x1 + np.array(4.0)
        y.backward()
        self.assertEqual(10, y.data)
        self.assertEqual(2, x1.gradient)

        x2 = Variable(np.array(3.0))
        y = np.array(2.0) * x2 + 4.0
        y.backward()
        self.assertEqual(10, y.data)
        self.assertEqual(2, x2.gradient)

    def test_neg(self):
        x0 = Variable(np.array(2.0))
        x1 = Variable(np.array(3.0))
        
        y = -x0
        y.backward()
        self.assertEqual(-2, y.data)
        self.assertEqual(-1, x0.gradient)

    def test_sub(self):
        x0 = Variable(np.array(2.0))
        x1 = Variable(np.array(3.0))
        y = x0 - x1
        y.backward()

        self.assertEqual(-1, y.data)
        self.assertEqual(1, x0.gradient)
        self.assertEqual(-1, x1.gradient)

        x2 = Variable(np.array(2.0))
        y = x2 - 5
        y.backward()
        
        self.assertEqual(-3, y.data)
        self.assertEqual(1, x2.gradient)

        x3 = Variable(np.array(2.0))
        y = np.array(5) - x3
        y.backward()
        
        self.assertEqual(3, y.data)
        self.assertEqual(-1, x3.gradient)

    def test_div(self):
        x0 = Variable(np.array(6.0))
        y = x0 / np.array(2.0)
        y.backward()

        self.assertEqual(3, y.data)
        self.assertEqual(0.5, x0.gradient)

    def test_pow(self):
        x0 = Variable(np.array(6.0))
        y = x0 ** 2
        y.backward()

        self.assertEqual(36, y.data)
        self.assertEqual(2 * 6, x0.gradient)