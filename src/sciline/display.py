# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from html import escape
from typing import Iterable, List, Tuple, TypeVar, Union

from ._provider import Provider
from ._utils import groupby, key_name
from .typing import Item, Key


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
            return escape(key_name(key[args]))
    return escape(key_name(key))


def _provider_source(
    p: Tuple[Key, Tuple[Union[Key, TypeVar], ...], List[Provider]]
) -> str:
    key, _, (v, *rest) = p
    if v.kind == 'table_cell':
        # This is always the case, but mypy complains
        if isinstance(key, Item):
            return escape(
                f'ParamTable({key_name(key.label[0].tp)}, length={len((v, *rest))})'
            )
    if v.kind == 'function':
        return _details(
            escape(v.location.name),
            escape(f'{v.location.module}.{v.location.name}'),
        )
    return ''


def _provider_value(
    p: Tuple[Key, Tuple[Union[Key, TypeVar], ...], List[Provider]]
) -> str:
    _, _, (v, *_) = p
    if v.kind == 'parameter':
        html = escape(str(v.call({}))).strip()
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
