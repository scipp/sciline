from typing import List, TypeVar

T = TypeVar("T")


def find_all_paths(graph, start: T, end: T) -> List[List[T]]:
    """Find all paths from start to end in a DAG."""
    if start == end:
        return [[start]]
    if start not in graph:
        return []
    paths = []
    for node in graph[start]:
        for path in find_all_paths(graph, node, end):
            paths.append([start] + path)
    return paths


def find_nodes_in_paths(graph, start: T, end: T) -> List[T]:
    # 0 is the provider, 1 is the args
    graph = {k: v[1] for k, v in graph.items()}
    paths = find_all_paths(graph, start, end)
    nodes = set()
    for path in paths:
        nodes.update(path)
    return list(nodes)
