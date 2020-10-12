#!/usr/bin/env python3

from math import sin, sqrt

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

PRECISION = 0.001
RANDOM_SEED = 42
GRADIENT_DECREASING_COEFFICIENT = 0.001


def generate_data():
    alpha = np.random.uniform(0.0, 1.0)
    beta = np.random.uniform(0.0, 1.0)
    size = 101

    xs = np.zeros(size)
    ys = np.zeros(size)
    deltas = np.random.normal(scale=1, size=101)
    for k in range(0, size):
        xs[k] = k / 100
        ys[k] = alpha * xs[k] + beta + deltas[k]

    return xs, ys


def linear_approximation(args, x):
    return args[0] * x + args[1]


def rational_approximation(args, x):
    return args[0] / (1 + args[1] * x)


def arg_distance(a1, b1, a2, b2):
    return sqrt((a1 - a2) ** 2 + (b1 - b2) ** 2)


def squared_deviations_sum(approximation_function, xs, ys, args):
    res = 0.0

    for k in range(0, 101):
        orig_res = ys[k]
        approx_res = approximation_function(args, xs[k])
        res += (orig_res - approx_res) ** 2
    return res


def gradient_descent(func_to_optimize, x0, precision):
    result = x0
    i = 0
    function_evaluations = 0
    while True:
        i += 1
        function_value = func_to_optimize(result)
        gradient = opt.approx_fprime(result, func_to_optimize,
                                     [precision] * len(result))
        # approx_frime evaluates function twice
        function_evaluations += 3
        if arg_distance(gradient[0], gradient[1], 0, 0) < PRECISION:
            print()
            break
        result -= gradient * GRADIENT_DECREASING_COEFFICIENT

    print("Current function value:", function_value)
    print("Iterations:", i)
    print("Function evaluations:", function_evaluations)
    return result


def conjugate_gradient(func_to_optimize, x0, precision):
    return opt.minimize(func_to_optimize, x0, tol=precision, method='CG',
                        options={'disp': True}).x


def newton_method(func_to_optimize, x0, precision):
    def jacobian(params):
        return opt.approx_fprime(params, func_to_optimize, precision)

    return opt.minimize(func_to_optimize, x0, tol=precision,
                        method='Newton-CG',
                        jac=jacobian,
                        options={'disp': True}).x


def levenberg_marquardt(func_to_optimize, approx_func, x, y, x0, precision):
    def error_func(args):
        return y - approx_func(args, x)

    res = opt.least_squares(error_func, x0, method='lm', ftol=precision)
    function_value = func_to_optimize(res.x)
    print("Current function value:", function_value)
    print("Function evaluations:", res.nfev)
    return res.x



def main():
    np.random.seed(RANDOM_SEED)
    xs, ys = generate_data()

    for plot_title, approx_func in [
        ("Linear approximation", linear_approximation),
        ("Rational approximation", rational_approximation)
    ]:
        print("\n" + plot_title)

        def func_to_optimize(args):
            return squared_deviations_sum(approx_func, xs, ys, args)

        for method, color, name in [
            (gradient_descent, 'm', "Gradient descent"),
            (conjugate_gradient, 'r', "Conjugate gradient"),
            (newton_method, 'g', "Newton"),
            (levenberg_marquardt, 'y', "Levenberg-Marquardt")
        ]:
            print(method.__name__, " approximation:", sep="")

            if method == levenberg_marquardt:
                res = levenberg_marquardt(func_to_optimize, approx_func,
                                          xs, ys, [.25, 0], PRECISION)
            else:
                res = method(func_to_optimize, [.25, 0.], PRECISION)
            print("Result:", res)

            approx_ys = approx_func(res, xs)

            plt.plot(xs, approx_ys, color=color, label=name)
            print()

        plt.title(plot_title)
        plt.scatter(x=xs, y=ys, label='Generated data')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()
        plt.clf()


if __name__ == '__main__':
    main()
