from typing import Callable, Collection, List, Mapping, Tuple, TypeVar

T = TypeVar("T")


def find_all_paths(graph: Mapping[T, Collection[T]], start: T, end: T) -> List[List[T]]:
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


def find_nodes_in_paths(
    graph: Mapping[T, Tuple[Callable[..., T], Collection[T]]], start: T, end: T
) -> List[T]:
    # 0 is the provider, 1 is the args
    dependencies = {k: v[1] for k, v in graph.items()}
    paths = find_all_paths(dependencies, start, end)
    nodes = set()
    for path in paths:
        nodes.update(path)
    return list(nodes)
