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
    args: Tuple[Key | TypeVar, ...]
    kind: ProviderKind
    value: Any


def _details(summary: str, body: str) -> str:
    return f'''
    <details>
      <summary>{summary}</summary>
      {body}
    </details>
    '''


def _provider_name(p: Union[ProviderDisplayData, Tuple]) -> str:
    if isinstance(p, tuple):
        (name, cname), values = p
        return escape(f'{qualname(cname)}({qualname(name)})')
    name = f'{qualname(p.origin)}'
    if p.args:
        args = ','.join(
            ('*' if isinstance(arg, TypeVar) else f'{qualname(arg)}' for arg in p.args)
        )
        name += f'[{args}]'
    return escape(name)


def _provider_source(p: Union[ProviderDisplayData, Tuple]) -> str:
    if isinstance(p, tuple) or p.kind != 'function':
        return ''
    module = getattr(inspect.getmodule(p.value), '__name__', '')
    return _details(
        escape(p.value.__name__),
        escape(f'{module}.{p.value.__name__}'),
    )


def _provider_value(p: Union[ProviderDisplayData, Tuple]) -> str:
    if isinstance(p, tuple):
        (name, cname), values = p
        return escape(f'length: {len(values)}')
    if p.kind != 'parameter':
        return ''
    if hasattr(p.value, '_repr_html_'):
        return _details('', p.value._repr_html_())
    return escape(str(p.value))


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
          <td scope="row">{_provider_source(p)}</th>
          <td scope="row">{_provider_value(p)}</td>
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
            <th scope="col">Source</th>
            <th scope="col">Value</th>
          </tr>
        </thead>
        <tbody>
          {provider_rows}
        </tbody>
      </table>
    </div>
    '''.strip()
