import unittest
from dezero import *
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
        '負数'
        x0 = Variable(np.array(2.0))
        x1 = Variable(np.array(3.0))
        
        y = -x0
        y.backward()
        self.assertEqual(-2, y.data)
        self.assertEqual(-1, x0.gradient)

    def test_sub(self):
        '引き算'
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
        '割り算'
        x0 = Variable(np.array(6.0))
        y = x0 / np.array(2.0)
        y.backward()

        self.assertEqual(3, y.data)
        self.assertEqual(0.5, x0.gradient)

    def test_pow(self):
        '累乗'
        x0 = Variable(np.array(6.0))
        y = x0 ** 2
        y.backward()

        self.assertEqual(36, y.data)
        self.assertEqual(2 * 6, x0.gradient)

    def test_sphere(self):
        x = Variable(np.array(2.0))
        y = Variable(np.array(5.0))
        z = self.sphere(x, y)
        z.backward()

        self.assertEqual(29, z.data)
        self.assertEqual(4, x.gradient)

    def test_matyas(self):
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = self.matyas(x, y)
        z.backward()

        self.assertEqual(0.040000000000000036, z.data)
        self.assertEqual(0.040000000000000036, x.gradient)

    def test_goldstein_price(self):
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = self.goldstein(x, y)
        z.backward()

        self.assertEqual(-5376, x.gradient)
        self.assertEqual(8064, y.gradient)

    def sphere(self, x, y):
        z = x ** 2 + y ** 2
        return z

    def matyas(self, x, y):
        z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
        return z

    def goldstein(self, x, y):
        return (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * \
            (30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y -36 * x * y + 27 * y ** 2))
