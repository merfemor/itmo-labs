import matplotlib.pyplot as plt
import numpy as np


# Final Test, task #2
def task2():
    ys = [51, 33, 34, 49, 41, 48, 40, 47, 54, 60, 52, 42, 43, 53, 63, 57, 58, 50, 59, 70, 65, 55, 71, 69, 74]
    xs = range(len(ys))
    plt.scatter(xs, ys)

    n = len(ys)
    y_sum = np.sum(ys)
    x_sum = np.sum(xs)
    a = (n * np.sum([ys[i] * xs[i] for i in range(len(ys))]) - x_sum * y_sum) / \
        (n * ((np.array(xs) ** 2).sum()) - x_sum ** 2)
    b = (y_sum - a * x_sum) / n

    def approx_fun(x): return a * x + b

    approx_y = [approx_fun(x) for x in xs]
    plt.plot(xs, approx_y)
    plt.show()
    print(a, b)

    avg_y = y_sum / n

    r2 = 1 - np.sum([(ys[i] - approx_fun(xs[i])) ** 2 for i in range(n)]) / \
         np.sum([(ys[i] - avg_y) ** 2 for i in range(n)])
    print(r2)


def main():
    task2()


if __name__ == '__main__':
    main()
