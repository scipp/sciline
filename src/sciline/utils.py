from collections import defaultdict
from typing import Any, Callable, DefaultDict, Iterable, TypeVar, Union, get_args

from .typing import Item, Key

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


def keyname(key: Union[Key, TypeVar]) -> str:
    if isinstance(key, TypeVar):
        return str(key)
    if isinstance(key, Item):
        return f'{keyname(key.tp)}({keyname(key.label[0].tp)})'
    args = get_args(key)
    if len(args):
        parameters = ', '.join(map(keyname, args))
        return f'{qualname(key)}[{parameters}]'
    return qualname(key)
