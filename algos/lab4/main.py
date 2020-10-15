#!/usr/bin/env python3

from math import sin, sqrt

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

PRECISION = 0.001
RANDOM_SEED = 42
MAX_ITERATIONS = 1000


def generate_data():
    size = 1001

    xs = np.zeros(size)
    ys = np.zeros(size)
    deltas = np.random.normal(scale=1, size=size)

    def f(x):
        return 1 / (x ** 2 - 3 * x + 2)

    for k in range(0, size):
        xs[k] = 3 * k / 1000
        f_val = f(xs[k])
        if f_val < -100:
            ys[k] = -100 + deltas[k]
        elif f_val <= 100:
            ys[k] = f_val + deltas[k]
        else:
            ys[k] = 100 + deltas[k]
    return xs, ys


def rational_approximation(args, x):
    a, b, c, d = args
    return (a * x + b) / (x ** 2 + c * x + d)


def arg_distance(a1, b1, a2, b2):
    return sqrt((a1 - a2) ** 2 + (b1 - b2) ** 2)


def squared_deviations_sum(approximation_function, xs, ys, args):
    assert len(xs) == len(ys)
    res = 0.0

    for k in range(0, len(xs)):
        orig_res = ys[k]
        approx_res = approximation_function(args, xs[k])
        res += (orig_res - approx_res) ** 2
    return res


def nelder_mead(func_to_optimize, x0, precision):
    res = opt.minimize(func_to_optimize, x0, method='Nelder-Mead',
                        options={'xatol': precision, 'maxiter': MAX_ITERATIONS})

    print("Current function value:", res.fun)
    print("Function evaluations:", res.nfev)
    print("Iterations:", res.nit)
    return res.x


def levenberg_marquardt(func_to_optimize, approx_func, x, y, x0, precision):
    def error_func(args):
        return y - approx_func(args, x)

    res = opt.least_squares(error_func, x0, method='lm', xtol=precision, max_nfev=MAX_ITERATIONS)
    function_value = func_to_optimize(res.x)
    print("Current function value:", function_value)
    print("Function evaluations:", res.nfev)
    return res.x


def simulated_anneling(func_to_optimize, x0, precision):
    res = opt.basinhopping(func_to_optimize, x0, niter=MAX_ITERATIONS, disp=True)
    print("Function value:", res.fun)
    print("Function evaluations:", res.nfev)
    print("Iterations:", res.nit)
    return res.x


def differential_evolution(func_to_optimize, bounds, precision):
    res = opt.differential_evolution(func_to_optimize, bounds, maxiter=MAX_ITERATIONS,
                                     tol=precision, disp=True)
    print("Function value:", res.fun)
    print("Function evaluations:", res.nfev)
    print("Iterations:", res.nit)
    return res.x


def main():
    np.random.seed(RANDOM_SEED)
    xs, ys = generate_data()

    x0 = [1., 1., 1., 1.]
    bounds = [[0.001, 3.], [-4., -1.], [-5., 2.], [2., 5.]]

    def func_to_optimize(args):
        return squared_deviations_sum(rational_approximation, xs, ys, args)

    for method, color, name in [
        (nelder_mead, 'm', "Nelder-Mead"),
        (levenberg_marquardt, 'y', "Levenberg-Marquardt"),
        (simulated_anneling, 'g', 'Simulated anneling'),
        (differential_evolution, 'r', 'Differential evolution')
    ]:
        print(method.__name__, " approximation:", sep="")

        if method == levenberg_marquardt:
            res = levenberg_marquardt(func_to_optimize, rational_approximation,
                                      xs, ys, x0, PRECISION)
        elif method == differential_evolution:
            res = differential_evolution(func_to_optimize, bounds, PRECISION)
        else:
            res = method(func_to_optimize, x0, PRECISION)
        print("Result:", res)

        approx_ys = rational_approximation(res, xs)

        plt.plot(xs, approx_ys, color=color, label=name)
        print()

    plt.title("Rational approximation")
    plt.scatter(xs, ys, label='Generated data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    plt.clf()


if __name__ == '__main__':
    main()
