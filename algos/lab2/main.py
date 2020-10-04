#!/usr/bin/env python3

from math import sin, sqrt

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

PRECISION = 0.001
RANDOM_SEED = 42


def fun1(x):
    return x ** 3


def fun2(x):
    return abs(x - 0.2)


def fun3(x):
    return x * sin(1 / x)


def exhaustive_search(f, x_from, x_to, precision):
    best_x = x_from
    best_f_value = f(x_from)
    f_calculations = 1
    iterations = 1
    for x in np.arange(x_from + precision, x_to, precision):
        cur_f = f(x)
        f_calculations += 1
        if cur_f < best_f_value:
            best_f_value = cur_f
            best_x = x
        iterations += 1
    return best_x, f_calculations, iterations


def dichotomy_method(f, x_from, x_to, precision):
    f_calculations = 0
    iterations = 0
    delta = precision / 2

    while abs(x_to - x_from) > precision:
        x1 = (x_from + x_to - delta) / 2
        x2 = (x_from + x_to + delta) / 2
        f1_res = f(x1)
        f2_res = f(x2)
        f_calculations += 2

        if f1_res <= f2_res:
            x_to = x2
        else:
            x_from = x1

        iterations += 1

    best_x = (x_to + x_from) / 2
    return best_x, f_calculations, iterations


def golden_search_method(f, x_from, x_to, precision):
    f_calculations = 0
    iterations = 0

    x_for_next_iteration = None
    f_for_next_iteration = None
    not_count_x1 = None

    while abs(x_to - x_from) > precision:
        if not_count_x1 is True:
            x1 = x_for_next_iteration
            f1_res = f_for_next_iteration
        else:
            x1 = x_from + (3 - sqrt(5)) / 2 * (x_to - x_from)
            f1_res = f(x1)
            f_calculations += 1

        if not_count_x1 is False:
            x2 = x_for_next_iteration
            f2_res = f_for_next_iteration
        else:
            x2 = x_to + (sqrt(5) - 3) / 2 * (x_to - x_from)
            f2_res = f(x2)
            f_calculations += 1

        if f1_res <= f2_res:
            x_to = x2
            not_count_x1 = False
            x_for_next_iteration = x1
            f_for_next_iteration = f1_res
        else:
            x_from = x1
            not_count_x1 = True
            x_for_next_iteration = x2
            f_for_next_iteration = f2_res

        iterations += 1

    best_x = (x_to + x_from) / 2
    return best_x, f_calculations, iterations


def generate_data():
    alpha = np.random.uniform(0.0, 1.0)
    beta = np.random.uniform(0.0, 1.0)

    xs = []
    ys = []
    deltas = np.random.normal(scale=1, size=101)
    for k in range(0, 101):
        x = k / 100
        y = alpha * x + beta + deltas[k]
        xs.append(x)
        ys.append(y)

    return xs, ys


def linear_approximation(x, args):
    return args[0] * x + args[1]


def rational_approximation(x, args):
    return args[0] / (1 + args[1] * x)


def squared_deviations_sum(approximation_function, xs, ys, args):
    res = 0.0

    for k in range(0, 101):
        orig_res = ys[k]
        approx_res = approximation_function(xs[k], args)
        res += (orig_res - approx_res) ** 2
    return res


def exhaustive_search_two_arguments(f, a_from, a_to, b_from, b_to, precision):
    best_a = a_from
    best_b = b_from
    best_res = f([a_from, b_from])

    iterations = 0
    f_calculations = 1

    for a in np.arange(a_from + precision, a_to, precision):
        for b in np.arange(b_from + precision, b_to, precision):
            cur_res = f([a, b])
            f_calculations += 1
            if cur_res < best_res:
                best_a = a
                best_b = b
                best_res = cur_res

            iterations += 1

    print("\t\tCurrent function value:", best_res)
    print("\t\tIterations:", iterations)
    print("\t\tFunction evaluations:", f_calculations)
    return [best_a, best_b]


def gauss_descent(f, a_from, a_to, b_from, b_to, precision):
    best_args = [(a_from + a_to) / 2, (b_from + b_to) / 2]
    best_res = f(best_args)

    iterations = 0
    f_calculations = 1

    for a in np.arange(a_from + precision, a_to, precision):
        cur_res = f([a, best_args[1]])
        f_calculations += 1
        if cur_res < best_res:
            best_args[0] = a
            best_res = cur_res
        iterations += 1

    for b in np.arange(b_from + precision, b_to, precision):
        cur_res = f([best_args[0], b])
        f_calculations += 1
        if cur_res < best_res:
            best_args[1] = b
            best_res = cur_res
        iterations += 1

    print("\t\tCurrent function value:", best_res)
    print("\t\tIterations:", iterations)
    print("\t\tFunction evaluations:", f_calculations)
    return best_args


def nelder_mead(f, a_from, a_to, b_from, b_to, precision):
    result = opt.minimize(f, [(a_from + a_to) / 2, (b_from + b_to) / 2],
                          method='Nelder-Mead',
                          options={'xatol': precision, 'disp': True})
    return result.x


if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)

    for method in [exhaustive_search, dichotomy_method, golden_search_method]:
        res1 = method(fun1, 0.0, 1.0, PRECISION)
        res2 = method(fun2, 0.0, 1.0, PRECISION)
        res3 = method(fun3, 0.01, 1.0, PRECISION)

        print("\n", method.__name__, ":", sep="")
        print("f1:", res1)
        print("f2:", res2)
        print("f3:", res3)

    xs, ys = generate_data()

    def func_to_optimize_linear(args):
        return squared_deviations_sum(linear_approximation, xs, ys, args)

    def func_to_optimize_rational(args):
        return squared_deviations_sum(rational_approximation, xs, ys, args)

    approx_xs = np.linspace(start=min(xs), stop=max(xs), num=len(xs))

    for func_to_optimize, plot_title, approx, a_from, a_to, b_from, b_to in [
        (func_to_optimize_linear, "Linear approximation",
         linear_approximation, -0.5, 1.0, -1.5, 1.5),
        (func_to_optimize_rational, "Rational approximation",
         rational_approximation, -1, 1, -0.5, 1)
    ]:
        print()
        print(func_to_optimize.__name__, " optimization:", sep="")

        for method, color, name in [
            (exhaustive_search_two_arguments, 'm', "Exhaustive"),
            (gauss_descent, 'r', "Gauss"),
            (nelder_mead, 'g', "Nelder-Mead")
        ]:
            print(method.__name__, " approximation:", sep="")

            res = method(func_to_optimize, a_from, a_to, b_from, b_to, PRECISION)
            print("Result:", res)

            def approx_func(x): return approx(x, res)

            approx_ys = [approx_func(x) for x in approx_xs]

            plt.plot(approx_xs, approx_ys, color=color, label=name)
            print()

        plt.title(plot_title)
        plt.scatter(x=xs, y=ys, label='Generated data')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()
        plt.clf()

