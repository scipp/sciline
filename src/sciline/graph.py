from typing import List

from sciline.pipeline import Key


def find_path(graph, start, end) -> List[Key]:
    """Find a path from start to end in a DAG."""
    if start == end:
        return [start]
    for node in graph[start]:
        path = find_path(graph, node, end)
        if path:
            return [start] + path
    return []


def find_unique_path(graph, start, end) -> List[Key]:
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


def find_all_paths(graph, start, end) -> List[Key]:
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
