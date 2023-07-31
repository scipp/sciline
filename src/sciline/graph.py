from typing import List, TypeVar

T = TypeVar("T")


def find_path(graph, start: T, end: T) -> List[T]:
    """Find a path from start to end in a DAG."""
    if start == end:
        return [start]
    for node in graph[start]:
        path = find_path(graph, node, end)
        if path:
            return [start] + path
    return []


def find_unique_path(graph, start: T, end: T) -> List[T]:
    """Find a path from start to end in a DAG.

    Like find_path, but raises if more than one path found
    """
    if start == end:
        return [start]
    if start not in graph:
        return []
    paths = []
    for node in graph[start]:
        path = find_unique_path(graph, node, end)
        if path:
            paths.append([start] + path)
    if len(paths) > 1:
        raise RuntimeError(f"Multiple paths found from {start} to {end}")
    return paths[0] if paths else []


def find_all_paths(graph, start: T, end: T) -> List[List[T]]:
    """Find all paths from start to end in a DAG."""
    if start == end:
        return [[start]]
    if start not in graph:
        return []
    paths = []
    # 0 is the provider, 1 is the args
    for node in graph[start][1]:
        for path in find_all_paths(graph, node, end):
            paths.append([start] + path)
    return paths


def find_nodes_in_paths(graph, start: T, end: T) -> List[T]:
    paths = find_all_paths(graph, start, end)
    nodes = set()
    for path in paths:
        nodes.update(path)
    return list(nodes)
