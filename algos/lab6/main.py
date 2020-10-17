#!/usr/bin/env python3

import time
from queue import PriorityQueue

import numpy as np

RANDOM_SEED = 6669
VERTICLES_NUM = 100
EDGES_NUM = 500
MAX_WEIGTH = 1000
INF = 1e10
CELL_GRID_SIZE = 10


def get_two_different_random_values(to):
    v_from = np.random.randint(0, to)
    v_to = np.random.randint(0, to - 1)
    if v_to == v_from:
        v_to += 1
    return v_from, v_to


def generate_graph():
    adjacency_matrix = np.zeros((VERTICLES_NUM, VERTICLES_NUM), dtype=int)
    edges_created = 0
    edge_list = []

    while edges_created < EDGES_NUM:
        v_from, v_to = get_two_different_random_values(VERTICLES_NUM)
        if adjacency_matrix[v_from][v_to] != 0:
            continue

        edges_created += 1
        weight = np.random.randint(0, MAX_WEIGTH)
        adjacency_matrix[v_from][v_to] = weight
        adjacency_matrix[v_to][v_from] = weight
        edge_list.append((v_from, v_to, weight))

    return adjacency_matrix, edge_list


def generate_cell_grid():
    size = CELL_GRID_SIZE
    obstacle_count = 30

    matrix = [np.zeros(size + 2, dtype=int) for _ in range(size + 2)]
    for i in range(1, size + 1):
        for j in range(1, size + 1):
            matrix[i][j] = 1

    obstacle_count_yet = 0
    while obstacle_count_yet < obstacle_count:
        i, j = np.random.randint(1, size + 1, 2)
        if matrix[i][j] != 0:
            obstacle_count_yet += 1
            matrix[i][j] = 0

    return matrix


def choose_two_random_free_vertices(matrix):
    size = len(matrix)
    dots = []
    while len(dots) < 2:
        i, j = np.random.randint(0, size, 2)
        if matrix[i][j] == 1:
            dots.append((i, j))

    return dots[0], dots[1]


def adjacency_matrix_to_list(adjacency_matrix):
    verticles_num = len(adjacency_matrix)
    adjacency_list = [[] for _ in range(verticles_num)]

    for v_from in range(verticles_num):
        for v_to in range(v_from + 1, verticles_num):
            if adjacency_matrix[v_from][v_to] != 0:
                adjacency_list[v_from].append(v_to)
                adjacency_list[v_to].append(v_from)

    return adjacency_list


def dijkstra(adjacency_matrix, adjacency_list, v_from):
    verticles_num = len(adjacency_matrix)
    dist = np.empty(verticles_num, dtype=int)
    dist.fill(INF)
    dist[v_from] = 0

    visited = np.zeros(verticles_num, dtype=bool)
    not_visited = verticles_num

    while not_visited > 0:
        min_v = -1
        min_dist = INF + 1
        for v in range(0, verticles_num):
            if not visited[v] and dist[v] < min_dist:
                min_dist = dist[v]
                min_v = v

        for n in adjacency_list[min_v]:
            path_len = min_dist + adjacency_matrix[min_v][n]
            if dist[n] > path_len:
                dist[n] = path_len

        visited[min_v] = True
        not_visited -= 1

    return dist


def bellman_ford(verticles_num, edge_list, v_from):
    dist = np.empty(verticles_num, dtype=int)
    dist.fill(INF)
    dist[v_from] = 0

    smth_changed = True
    while smth_changed:
        smth_changed = False
        for v_from, v_to, weigth in edge_list:
            if dist[v_to] > dist[v_from] + weigth:
                dist[v_to] = dist[v_from] + weigth
                smth_changed = True
            elif dist[v_from] > dist[v_to] + weigth:
                dist[v_from] = dist[v_to] + weigth
                smth_changed = True

    return dist


def a_star(matrix, start, fin):
    size = len(matrix)
    queue = PriorityQueue()
    queue.put((0, start))

    costs = np.full((size, size), INF)
    costs[start[0]][start[1]] = 0
    came_from = np.full((size, size), None)

    di = [0, 0, -1, 1]
    dj = [-1, 1, 0, 0]

    def f(i, j):
        return abs(i - fin[0]) + abs(j - fin[1])

    while not queue.empty():
        _, el = queue.get()
        if el == fin:
            break

        for k in range(4):
            ni = el[0] + di[k]
            nj = el[1] + dj[k]
            if matrix[ni][nj] == 1:
                new_cost = costs[el[0], el[1]] + 1
                if costs[ni][nj] > new_cost:
                    costs[ni][nj] = new_cost
                    priority = new_cost + f(ni, nj)
                    queue.put((priority, (ni, nj)))
                    came_from[ni][nj] = el

    path = [fin]
    cur_v = fin
    while cur_v != start:
        cur_v = came_from[cur_v[0]][cur_v[1]]
        path.append(cur_v)
    return path[::-1]


def print_matrix_with_start_and_finish(matrix, start, fin):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            x = matrix[i][j]
            if (i, j) == start:
                s = 'S'
            elif (i, j) == fin:
                s = 'F'
            elif x == 1:
                s = '.'
            else:
                s = '#'
            print(s, end='')
        print()


def measure_avg_time_exec(func):
    sum = 0
    repeat_count = 10
    for i in range(repeat_count):
        before = time.time_ns()
        func()
        after = time.time_ns()
        sum += after - before
    return sum / repeat_count / 1e6


def main():
    np.random.seed(RANDOM_SEED)
    adjacency_matrix, edge_list = generate_graph()
    adjacency_list = adjacency_matrix_to_list(adjacency_matrix)
    v_from = np.random.randint(0, VERTICLES_NUM)

    dijkstra_time = measure_avg_time_exec(
        lambda: dijkstra(adjacency_matrix, adjacency_list, v_from))
    print("Dijkstra algorithm:", dijkstra_time, "ms")

    bf_time = measure_avg_time_exec(
        lambda: bellman_ford(VERTICLES_NUM, edge_list, v_from))
    print("Bellman-Ford algorithm:", bf_time, "ms")

    matrix = generate_cell_grid()
    for experiment_number in range(5):
        print("Random cell grid at exp", experiment_number + 1)
        start, fin = choose_two_random_free_vertices(matrix)
        print_matrix_with_start_and_finish(matrix, start, fin)
        path = a_star(matrix, start, fin)
        print("Path:", path)


if __name__ == '__main__':
    main()
