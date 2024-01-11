from collections import defaultdict
from typing import Any, Callable, DefaultDict, Iterable, TypeVar

T = TypeVar('T')
G = TypeVar('G')


def groupby(f: Callable[[T], G], a: Iterable[T]) -> DefaultDict[G, list[T]]:
    g = defaultdict(lambda: [])
    for e in a:
        g[f(e)].append(e)
    return g


def qualname(obj: Any) -> str:
    return str(
        obj.__qualname__ if hasattr(obj, '__qualname__') else obj.__class__.__qualname__
    )
