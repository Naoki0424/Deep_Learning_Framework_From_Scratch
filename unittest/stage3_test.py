import unittest
from dezero import *
import numpy as np
import matplotlib.pyplot as plt

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

        self.assertEqual(x.data, 1)

    def test_sin_cos(self):
        x = Variable(np.linspace(-7, 7, 100))
        y = F.sin(x)
        y.backward(create_graph=True)

        logs = [y.data.flatten()]

        for i in range(3):
            logs.append(x.gradient.data.flatten())
            gx = x.gradient
            x.cleargradient()
            gx.backward(create_graph=True)

        labels = ["y=sin(x)", "y'", "y''", "y'''"]
        for i, v in enumerate(logs):
            plt.plot(x.data, logs[i], label=labels[i])
        plt.legend(loc = 'lower right')
        plt.show()

    def test_back_prop1(self):
        x = Variable(np.array(2))
        y = x ** 2
        y.backward(create_graph=True)
        gx = x.gradient
        x.cleargradient()

        z = gx ** 3 + y
        z.backward()
        self.assertEqual(100, x.gradient.data)