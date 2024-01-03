import inspect
from dataclasses import dataclass
from html import escape
from itertools import chain
from typing import Any, Literal, Mapping, Sequence, Tuple, TypeVar, Union

from .typing import Key
from .utils import groupby, qualname

ProviderKind = Literal['function', 'parameter', 'table']


@dataclass
class ProviderDisplayData:
    origin: Key
    args: Tuple[Union[Key, TypeVar], ...]
    kind: ProviderKind
    value: Any


def _details(summary: str, body: str) -> str:
    return f'''
    <details>
      <summary>{summary}</summary>
      {body}
    </details>
    '''


def _provider_name(
    p: Union[ProviderDisplayData, Tuple[Tuple[Any, Any], Sequence[ProviderDisplayData]]]
) -> str:
    if isinstance(p, tuple):
        (name, cname), values = p
        return escape(f'{qualname(cname)}({qualname(name)})')
    name = f'{qualname(p.origin)}'
    if p.args:
        args = ','.join(
            ('*' if isinstance(arg, TypeVar) else f'{qualname(arg)}' for arg in p.args)
        )
        name += f'[{args}]'
    return escape(f'{name}')


def _provider_source(
    p: Union[ProviderDisplayData, Tuple[Tuple[Any, Any], Sequence[ProviderDisplayData]]]
) -> str:
    if isinstance(p, tuple):
        (name, cname), values = p
        return escape(f'ParamTable({qualname(name)}, length={len(values)})')
    if p.kind == 'function':
        module = getattr(inspect.getmodule(p.value), '__name__', '')
        return _details(
            escape(p.value.__name__),
            escape(f'{module}.{p.value.__name__}'),
        )
    return ''


def _provider_value(
    p: Union[ProviderDisplayData, Tuple[Tuple[Any, Any], Sequence[ProviderDisplayData]]]
) -> str:
    if not isinstance(p, tuple) and p.kind == 'parameter':
        html = escape(str(p.value))
        return _details('', html) if len(html.strip()) > 40 else html
    return ''


def pipeline_html_repr(
    providers: Mapping[ProviderKind, Sequence[ProviderDisplayData]]
) -> str:
    param_table_columns_by_name_colname = groupby(
        lambda p: (p.origin.label[0].tp, p.origin.tp),
        providers['table'],
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
                chain(
                    providers['function'],
                    providers['parameter'],
                    param_table_columns_by_name_colname.items(),
                ),
                key=lambda p: _provider_name(p),
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
