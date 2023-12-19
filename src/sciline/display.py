import inspect
from dataclasses import dataclass
from html import escape
from itertools import chain
from typing import Any, Literal, Mapping, Sequence, Tuple, TypeVar

from .param_table import ParamTable
from .typing import Key
from .utils import qualname

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


def _provider_name(p: ProviderDisplayData) -> str:
    name = f'{qualname(p.origin)}'
    if p.args:
        args = ','.join(
            ('*' if isinstance(arg, TypeVar) else f'{qualname(arg)}' for arg in p.args)
        )
        name += f'[{args}]'
    return escape(name)


def _provider_source(p: ProviderDisplayData) -> str:
    if p.kind != 'function':
        return ''
    module = getattr(inspect.getmodule(p.value), '__name__', '')
    return _details(
        escape(p.value.__name__),
        escape(f'{module}.{p.value.__name__}'),
    )


def _provider_value(p: ProviderDisplayData) -> str:
    if p.kind != 'parameter':
        return ''
    if hasattr(p.value, '_repr_html_'):
        html = p.value._repr_html_()
        return f'{html}'
    return escape(str(p.value))


def provider_table(
    providers: Mapping[ProviderKind, Sequence[ProviderDisplayData]]
) -> str:
    provider_rows = '\n'.join(
        (
            f'''
        <tr>
          <td scope="row">{_provider_name(p)}</td>
          <td scope="row">{_provider_source(p)}</th>
          <td scope="row">{_provider_value(p)}</td>
        </tr>'''
            for p in sorted(
                chain(providers['function'], providers['parameter']),
                key=lambda p: _provider_name(p),
            )
        )
    )
    return f'''
    <div class="pipeline-component">
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
    '''


def _parameter_tables(parameter_tables: Sequence[ParamTable]) -> str:
    def parameter_table(p: ParamTable) -> str:
        html = p._repr_html_()
        return f'''
        <div>
          {html}
        </div>
        '''

    tables = '\n'.join(map(parameter_table, parameter_tables))
    return f'''
    <style>
      .parameter-tables-container {{
         display: flex;
         column-gap: 1em;
         align-items: flex-start;
      }}
    </style>
    <div class="pipeline-component">
      <h4>Parameter tables</h4>
      <div class="parameter-tables-container">
        {tables}
      </div>
    </div>
    '''


def pipeline_html_repr(
    providers: Mapping[ProviderKind, Sequence[ProviderDisplayData]],
    param_tables: Sequence[ParamTable],
) -> str:
    return f'''
     <style>
      .pipeline-component {{
          border: 1px solid;
          padding: 0.5em;
      }}
      .pipeline-container {{
          display: flex;
          column-gap: 1em;
          align-items: flex-start;
      }}
    </style>
    <div class="pipeline-container">
      {provider_table(providers) if len(providers) != 0 else ''}
      {_parameter_tables(param_tables) if len(param_tables) != 0 else ''}
    </div>
    '''.strip()
