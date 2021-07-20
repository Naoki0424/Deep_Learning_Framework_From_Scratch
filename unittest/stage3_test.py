import unittest
from dezero import *
import numpy as np

class utils(unittest.TestCase):
    def test_get_dot_graph(self):
        'utilsのget_dot_graphをテストする'

        x0 = Variable(np.array(1.0))
        x1 = Variable(np.array(1.0))
        y = x0 + x1
        y.backward()
        self.assertEqual('digraph g {\n' + \
            str(id(y)) + '[label="", color=orange, style=filled]\n' + \
                str(id(y.creator)) + '[label="Add", color=lightblue, style=filled, shape=box]\n' + \
                    '{} -> {}\n'.format(id(x0), id(y.creator)) + \
                        '{} -> {}\n'.format(id(x1), id(y.creator)) + \
                            '{} -> {}\n'.format(id(y.creator), id(y)) +\
                                str(id(x0)) + '[label="", color=orange, style=filled]\n' + \
                                    str(id(x1)) + '[label="", color=orange, style=filled]\n' + \
                                        '}', get_dot_graph(y))

    def test_graph_output(self):
        x0 = Variable(np.array(1.0))
        x1 = Variable(np.array(1.0))
        y = x0 + x1
        y.backward()
        plot_dot_graph(y)

    def test_second_derivative(self):
        def f(x):
            y = x ** 4 - 2 * x ** 2
            return y
        
        x = Variable(np.array(2.0))
        y = f(x)
        y.backward(create_graph=True)
        print(x.gradient)

        gx = x.gradient
        x.cleargradient()
        gx.backward()
        print(x.gradient)

class calc(unittest.TestCase):
    def test_newtons_method(self):
        def f(x):
            y = x ** 4 - 2 * x ** 2
            return y
        x = Variable(np.array(2.0))
        iters = 10

        for i in range(iters):
            print(i, x)
            y = f(x)
            
            x.cleargradient()
            y.backward(create_graph=True)
            gx = x.gradient

            x.cleargradient()
            gx.backward()
            gx2 = x.gradient

            x.data -= gx.data / gx2.data




    # def test_sin(self):
    #     x = Variable(np.array(np.pi/4))
    #     y = sin(x)
    #     y.backward()
    #     print(y.data)
    #     print(x.gradient)
    #     self.assertEqual(1/np.sqrt(2), y.data)
    #     self.assertEqual(round(1/np.sqrt(2), 7), round(x.gradient, 7))

    # def test_my_sin(self):
    #     x = Variable(np.array(np.pi/4))
    #     y = my_sin(x)
    #     y.backward()
    #     print(y.data)
    #     print(x.gradient)
    #     plot_dot_graph(y)
    #     self.assertEqual(round(1/np.sqrt(2), 4), round(x.gradient, 4))

    # def test_rosenbrock(self):
    #     x0 = Variable(np.array(0))
    #     x1 = Variable(np.array(2))
    #     y = rosenbrock(x0, x1)
    #     y.backward()
    #     print(x0.gradient)
    #     print(x1.gradient)
    #     self.assertEqual(-2.0, x0.gradient)
    #     self.assertEqual(400, x1.gradient)

    # def test_rosenbrock_gradient_descent(self):
    #     x0 = Variable(np.array(0.0))
    #     x1 = Variable(np.array(2.0))
    #     lr = 0.001
    #     iters = 1000

    #     for i in range(iters):
    #         print(x0, x1)

    #         y = rosenbrock(x0, x1)

    #         x0.cleargradient()
    #         x1.cleargradient()
    #         y.backward()

    #         x0.data -= lr * x0.gradient
    #         x1.data -= lr * x1.gradient

    # def f(self, x):
    #     return x ** 4 - 2 * x ** 2

    # def gx2(self, x):
    #     return 12 * x ** 2 - 4

    # def test_Newtons_method(self):
    #     x = Variable(np.array(2.0))
    #     iters = 10

    #     for i in range(iters):
    #         print(x)

    #         y = self.f(x)

    #         x.cleargradient()
    #         y.backward()

    #         x.data -= x.gradient / self.gx2(x.data)
