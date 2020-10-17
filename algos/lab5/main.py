#!/usr/bin/env python3

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

RANDOM_SEED = 666669
VERTICLES_NUM = 100
EDGES_NUM = 200


def get_two_different_random_verticles():
    v_from = np.random.randint(0, VERTICLES_NUM)
    v_to = np.random.randint(0, VERTICLES_NUM - 1)
    if v_to == v_from:
        v_to += 1
    return v_from, v_to


def generate_data():
    adjacency_matrix = np.zeros((VERTICLES_NUM, VERTICLES_NUM))
    edges_created = 0
    while edges_created < EDGES_NUM:
        v_from, v_to = get_two_different_random_verticles()
        if adjacency_matrix[v_from][v_to] == 1:
            continue

        edges_created += 1
        adjacency_matrix[v_from][v_to] = 1
        adjacency_matrix[v_to][v_from] = 1

    return adjacency_matrix


def plot_graph(adjacency_matrix):
    G = nx.Graph(adjacency_matrix)
    pos = nx.drawing.nx_agraph.graphviz_layout(G)
    nx.draw(G, pos)
    plt.show()


def _connected_components(adjacency_list, colors, color_value, v):
    colors[v] = color_value
    for n in adjacency_list[v]:
        if colors[n] == 0:
            _connected_components(adjacency_list, colors, color_value, n)


def _connected_components_colors(adjacency_list):
    vertex_count = len(adjacency_list)
    colors = np.zeros(vertex_count, dtype=int)
    color_count = 0
    for v in range(0, vertex_count):
        if colors[v] == 0:
            color_count += 1
            colors[v] = color_count
            _connected_components(adjacency_list, colors, color_count, v)

    return colors, color_count


def connected_components(adjacency_list):
    colors, count = _connected_components_colors(adjacency_list)
    components = [[] for _ in range(count)]
    for i in range(len(colors)):
        components[colors[i] - 1].append(i)

    return components


def find_path_between(adjacency_list, v_from, v_to):
    queue = [(v_from, [])]
    visited = np.zeros(len(adjacency_list), dtype=bool)
    visited[v_from] = True

    while len(queue) > 0:
        cur_v, old_path = queue.pop(0)
        path = old_path[:]
        path.append(cur_v)
        visited[cur_v] = True
        if cur_v == v_to:
            return path
        for n in adjacency_list[cur_v]:
            if not visited[n]:
                queue.append((n, path))
    return None


def adjacency_matrix_to_list(adjacency_matrix):
    verticles_num = len(adjacency_matrix)
    adjacency_list = [[] for _ in range(verticles_num)]

    for v_from in range(verticles_num):
        for v_to in range(v_from + 1, verticles_num):
            if adjacency_matrix[v_from][v_to] == 1:
                adjacency_list[v_from].append(v_to)
                adjacency_list[v_to].append(v_from)

    return adjacency_list


def main():
    np.random.seed(RANDOM_SEED)
    adjacency_matrix = generate_data()
    adjacency_list = adjacency_matrix_to_list(adjacency_matrix)

    print("3 first rows of adjacency matrix:")
    for r in range(3):
        print(adjacency_matrix[r])
    print("3 first rows of adjacency list:")
    for r in range(3):
        print(adjacency_list[r])

    plot_graph(adjacency_matrix)

    components = connected_components(adjacency_list)
    print("Components:")
    for c in components:
        print(c)

    v_from, v_to = get_two_different_random_verticles()
    path = find_path_between(adjacency_list, v_from, v_to)
    print("Path between", v_from, "and", v_to)
    print(path)


if __name__ == '__main__':
    main()
