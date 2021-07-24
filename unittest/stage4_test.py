import unittest

from numpy.core.fromnumeric import reshape
from dezero import *
import numpy as np
import matplotlib.pyplot as plt
import dezero.functions as F

class util(unittest.TestCase):
    def test_shape(self):
        x = Variable(np.array([[1, 2], [4, 5], [7, 8]]))
        # y = F.reshape(x, [6, ])
        y = x.reshape(2, 3)

        y.backward(retain_gradient=True)
        self.assertEqual(x.data.shape, (3, 2))
        self.assertEqual(x.gradient.shape, (3, 2))
        self.assertEqual(y.data.shape, (2, 3))

        x.cleargradient()

        x = Variable(np.array([[1, 2], [4, 5], [7, 8]]))
        # y = F.reshape(x, [6, ])
        y = x.reshape((2, 3))

        y.backward(retain_gradient=True)
        self.assertEqual(x.data.shape, (3, 2))
        self.assertEqual(x.gradient.shape, (3, 2))
        self.assertEqual(y.data.shape, (2, 3))

    def test_transpose(self):
        x = Variable(np.array([[1, 2], [4, 5], [7, 8]]))
        y = x.T

        y.backward(retain_gradient=True)
        self.assertEqual(x.data.shape, (3, 2))
        self.assertEqual(x.gradient.shape, (3, 2))
        self.assertEqual(y.data.shape, (2, 3))