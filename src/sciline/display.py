# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from html import escape
from typing import Any, Iterable, Tuple

from ._utils import key_name
from .typing import Key


def _details(summary: str, body: str) -> str:
    return f'''
    <details>
      <summary>{summary}</summary>
      {body}
    </details>
    '''


def _provider_name(p: Tuple[Key, dict[str, Any]]) -> str:
    return escape(key_name(p[0]))


def _provider_source(data: dict[str, Any]) -> str:
    if (v := data.get('provider', None)) is None:
        return ''
    if v.kind == 'function':
        return _details(
            escape(v.location.name),
            escape(f'{v.location.module}.{v.location.name}'),
        )
    return ''


def _provider_value(data: dict[str, Any]) -> str:
    return data.get('value', '')


def pipeline_html_repr(nodes: Iterable[Tuple[Key, dict[str, Any]]]) -> str:
    provider_rows = '\n'.join(
        (
            f'''
        <tr>
          <td scope="row">{_provider_name(item)}</td>
          <td scope="row">{_provider_value(item[1])}</td>
          <td scope="row">{_provider_source(item[1])}</th>
        </tr>'''
            for item in sorted(nodes, key=_provider_name)
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
