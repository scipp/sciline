import inspect
from html import escape
from typing import Iterable, List, Tuple, TypeVar, Union

from .typing import Item, Key, Provider
from .utils import groupby, keyname, kind_of_provider


def _details(summary: str, body: str) -> str:
    return f'''
    <details>
      <summary>{summary}</summary>
      {body}
    </details>
    '''


def _provider_name(
    p: Tuple[Key, Tuple[Union[Key, TypeVar], ...], List[Provider]]
) -> str:
    key, args, _ = p
    if args:
        # This is always the case, but mypy complains
        if hasattr(key, '__getitem__'):
            return escape(keyname(key[args]))
    return escape(keyname(key))


def _provider_source(
    p: Tuple[Key, Tuple[Union[Key, TypeVar], ...], List[Provider]]
) -> str:
    key, _, (v, *rest) = p
    kind = kind_of_provider(v)
    if kind == 'table':
        # This is always the case, but mypy complains
        if isinstance(key, Item):
            return escape(
                f'ParamTable({keyname(key.label[0].tp)}, length={len((v, *rest))})'
            )
    if kind == 'function':
        module = getattr(inspect.getmodule(v), '__name__', '')
        return _details(
            escape(v.__name__),
            escape(f'{module}.{v.__name__}'),
        )
    return ''


def _provider_value(
    p: Tuple[Key, Tuple[Union[Key, TypeVar], ...], List[Provider]]
) -> str:
    _, _, (v, *_) = p
    kind = kind_of_provider(v)
    if kind == 'parameter':
        html = escape(str(v())).strip()
        return _details(f'{html[:30]}...', html) if len(html) > 30 else html
    return ''


def pipeline_html_repr(
    providers: Iterable[Tuple[Key, Tuple[Union[Key, TypeVar], ...], Provider]]
) -> str:
    def associate_table_values(
        p: Tuple[Key, Tuple[Union[Key, TypeVar], ...], Provider]
    ) -> Tuple[Key, Union[type, Tuple[Union[Key, TypeVar], ...]]]:
        key, args, v = p
        if isinstance(key, Item):
            return (key.label[0].tp, key.tp)
        return (key, args)

    providers_collected = (
        (key, args, [value, *(v for _, _, v in rest)])
        for ((key, args, value), *rest) in groupby(
            associate_table_values,
            providers,
        ).values()
    )
    provider_rows = '\n'.join(
        (
            f'''
        <tr>
          <td scope="row">{_provider_name(p)}</td>
          <td scope="row">{_provider_value(p)}</td>
          <td scope="row">{_provider_source(p)}</th>
        </tr>'''
            for p in sorted(
                providers_collected,
                key=_provider_name,
            )
        )
    )
    return f'''
    <div class="pipeline-html-repr">
      <table>
        <thead>
          <tr>
            <th scope="col">Name</th>
            <th scope="col">Value</th>
            <th scope="col">Source</th>
          </tr>
        </thead>
        <tbody>
          {provider_rows}
        </tbody>
      </table>
    </div>
    '''.strip()
