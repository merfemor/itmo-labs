#!/usr/bin/env python3

import random
import time

import pyswarms as ps
from matplotlib import pyplot as plt
from scipy import optimize as opt
import numpy as np

from whale_optimixation import WhaleOptimization

RANDOM_SEED = 13
POPULATION_SIZE = 40

global_exec_time_sum = 0.0


def max_subarray_brute_force(arr):
    best_start = None
    best_len = None
    best_sum = None

    for i in range(len(arr)):
        cur_sum = 0
        for j in range(i, len(arr)):
            cur_sum += i
            if best_sum is None or cur_sum > best_sum:
                best_sum = cur_sum
                best_start = i
                best_len = j - i + 1

    return best_start, best_len, best_sum


def max_crossing_subarray(arr, start, mid, end):
    best_left_i = mid
    best_left_sum = 0
    cur_sum = 0
    for i in range(mid - 1, start - 1, -1):
        cur_sum += arr[i]
        if cur_sum > best_left_sum:
            best_left_sum = cur_sum
            best_left_i = i

    best_right_i = mid
    best_right_sum = 0
    cur_sum = 0
    for i in range(mid + 1, end):
        cur_sum += arr[i]
        if cur_sum > best_left_sum:
            best_left_sum = cur_sum
            best_left_i = i

    return best_left_i, best_right_i - best_left_i + 1, arr[mid] + best_left_sum + best_right_sum


def max_subarray_divide_conquer(arr, start=0, end=None):
    if end is None:
        end = len(arr)

    if start + 1 == end:
        return start, 1, arr[start]

    mid = (start + end) // 2

    left_start, left_len, left_sum = max_subarray_divide_conquer(arr, start, mid)
    right_start, right_len, right_sum = max_subarray_divide_conquer(arr, mid, end)
    cross_start, cross_len, cross_sum = max_crossing_subarray(arr, start, mid, end)

    if left_sum >= right_sum and left_sum >= cross_sum:
        return left_start, left_len, left_sum
    if right_sum >= cross_sum:
        return right_start, right_len, right_sum
    return cross_start, cross_len, cross_sum


def measure_time_exec(func):
    before = time.time_ns()
    ret = func()
    after = time.time_ns()
    return (after - before) / 1e6, ret


def measure_avg_time_exec(func, repeat_count=5):
    time_sum = 0
    for i in range(repeat_count):
        func_time, _ = measure_time_exec(func)
        time_sum += func_time
    return time_sum / repeat_count


def maximal_subarray_experiment():
    random.seed(RANDOM_SEED)

    ns = range(1, 10)
    results_bf = []
    results_dq = []

    for size in ns:
        print("On test no ", size)

        test = [random.randint(-1e6, 1e6) for _ in range(size)]
        bf_time = measure_avg_time_exec(lambda: max_subarray_brute_force(test))
        dc_time = measure_avg_time_exec(lambda: max_subarray_divide_conquer(test))
        results_bf.append(bf_time)
        results_dq.append(dc_time)

    plt.scatter(ns, results_bf, color='m', label="Brute-force")
    plt.scatter(ns, results_dq, color='b', label="Divide-and-conquer")
    plt.legend()
    plt.xlabel("Size of input, n")
    plt.ylabel("Execution time, ms")
    plt.show()


def differential_evolution(function, bounds, iter_num):
    return opt.differential_evolution(function, bounds, seed=RANDOM_SEED, maxiter=iter_num, tol=-np.inf)


def dual_annealing(function, bounds, iter_num):
    return opt.dual_annealing(function, bounds, maxiter=iter_num, seed=RANDOM_SEED)


def particle_swarm(function, bounds, iter_num):
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 2, 'p': 2}
    optimizer = ps.global_best.GlobalBestPSO(n_particles=10, dimensions=2,
                                             options=options, bounds=bounds)
    return optimizer.optimize(function, iter_num)


def whale_algorithm(function, bounds, iter_num):
    function_arr_arg = lambda x, y: function([x, y])
    algo = WhaleOptimization(function_arr_arg, bounds, nsols=50, b=0.5, a=2, a_step=0.5 / 50)
    solutions = []
    for _ in range(iter_num):
        algo.optimize()
        solutions = algo.get_solutions()

    return solutions


def whale_algo_experiment():
    bounds = [[-10, 10], [-10, 10]]

    def matyas_function(args):
        if len(args) != 2:
            x, y = args[:, 0], args[:, 1]
        else:
            x, y = args
        return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y

    iterations = range(1, 500)
    results = {}

    for iter in iterations:
        print("On iteration:", iter)

        for method in [
            differential_evolution,
            dual_annealing,
            particle_swarm,
            whale_algorithm
        ]:
            global global_exec_time_sum
            global_exec_time_sum = 0

            def func_wrapper(args):
                time_in_fun, res = measure_time_exec(lambda: matyas_function(args))
                global global_exec_time_sum
                global_exec_time_sum += time_in_fun
                return res

            time, res = measure_time_exec(lambda: method(func_wrapper, bounds, iter))
            if method not in (whale_algorithm, particle_swarm):
                print(method.__name__, " iterations ", res.nit, ", fcalls", res.nfev)
            algo_time = (time - global_exec_time_sum)

            arr = results.get(method.__name__)
            if arr is None:
                arr = []
            arr.append(algo_time)
            results[method.__name__] = arr

    plt.clf()
    plt.scatter(iterations, results[differential_evolution.__name__], label="Differential evolution", color='r')
    plt.scatter(iterations, results[dual_annealing.__name__], label="Dual annealing", color='m')
    plt.scatter(iterations, results[particle_swarm.__name__], label="Particle swarm", color='y')
    plt.scatter(iterations, results[whale_algorithm.__name__], label="Whale", color='b')
    plt.xlabel("Iterations")
    plt.ylabel("Execution time, ms")
    plt.legend()
    plt.show()


def main():
    maximal_subarray_experiment()
    whale_algo_experiment()


if __name__ == '__main__':
    main()
