# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import List, NewType, Optional, TypeVar

import pytest

import sciline as sl
from sciline.typing import Item, Label


def test_set_param_table_raises_if_param_names_are_duplicate() -> None:
    pl = sl.Pipeline()
    pl.set_param_table(sl.ParamTable(int, {float: [1.0, 2.0, 3.0]}))
    with pytest.raises(ValueError):
        pl.set_param_table(sl.ParamTable(str, {float: [4.0, 5.0, 6.0]}))
    assert pl.compute(Item((Label(int, 1),), float)) == 2.0
    with pytest.raises(sl.UnsatisfiedRequirement):
        pl.compute(Item((Label(str, 1),), float))


def test_set_param_table_removes_columns_of_replaced_table() -> None:
    pl = sl.Pipeline()
    pl.set_param_table(sl.ParamTable(int, {float: [1.0, 2.0, 3.0]}))
    # We could imagine that this would be allowed if the index
    # (here: automatic index as range(3)) is the same. For now we do not.
    pl.set_param_table(sl.ParamTable(int, {str: ['a', 'b', 'c']}))
    assert pl.compute(Item((Label(int, 1),), str)) == 'b'
    with pytest.raises(sl.UnsatisfiedRequirement):
        pl.compute(Item((Label(int, 1),), float))


def test_can_get_elements_of_param_table() -> None:
    pl = sl.Pipeline()
    pl.set_param_table(sl.ParamTable(int, {float: [1.0, 2.0, 3.0]}))
    assert pl.compute(Item((Label(int, 1),), float)) == 2.0


def test_can_get_elements_of_param_table_with_explicit_index() -> None:
    pl = sl.Pipeline()
    pl.set_param_table(sl.ParamTable(int, {float: [1.0, 2.0, 3.0]}, index=[11, 12, 13]))
    assert pl.compute(Item((Label(int, 12),), float)) == 2.0


def test_can_replace_param_table() -> None:
    pl = sl.Pipeline()
    table1 = sl.ParamTable(int, {float: [1.0, 2.0, 3.0]}, index=[11, 12, 13])
    pl.set_param_table(table1)
    assert pl.compute(Item((Label(int, 12),), float)) == 2.0
    table2 = sl.ParamTable(int, {float: [4.0, 5.0, 6.0]}, index=[21, 22, 23])
    pl.set_param_table(table2)
    assert pl.compute(Item((Label(int, 22),), float)) == 5.0


def test_can_replace_param_table_with_table_of_different_length() -> None:
    pl = sl.Pipeline()
    table1 = sl.ParamTable(int, {float: [1.0, 2.0, 3.0]}, index=[11, 12, 13])
    pl.set_param_table(table1)
    assert pl.compute(Item((Label(int, 13),), float)) == 3.0
    table2 = sl.ParamTable(int, {float: [4.0, 5.0]}, index=[21, 22])
    pl.set_param_table(table2)
    assert pl.compute(Item((Label(int, 22),), float)) == 5.0
    # Make sure rows beyond the new table are not accessible
    with pytest.raises(sl.UnsatisfiedRequirement):
        pl.compute(Item((Label(int, 13),), float))


def test_failed_replace_due_to_column_clash_in_other_table() -> None:
    Row1 = NewType("Row1", int)
    Row2 = NewType("Row2", int)
    table1 = sl.ParamTable(Row1, {float: [1.0, 2.0, 3.0]})
    table2 = sl.ParamTable(Row2, {str: ['a', 'b', 'c']})
    table1_replacement = sl.ParamTable(
        Row1, {float: [1.1, 2.2, 3.3], str: ['a', 'b', 'c']}
    )
    pl = sl.Pipeline()
    pl.set_param_table(table1)
    pl.set_param_table(table2)
    with pytest.raises(ValueError):
        pl.set_param_table(table1_replacement)
    # Make sure the original table is still accessible
    assert pl.compute(Item((Label(Row1, 1),), float)) == 2.0


def test_can_depend_on_elements_of_param_table() -> None:
    # This is not a valid type annotation, not sure why it works with get_type_hints
    def use_elem(x: Item((Label(int, 1),), float)) -> str:  # type: ignore[valid-type]
        return str(x)

    pl = sl.Pipeline([use_elem])
    pl.set_param_table(sl.ParamTable(int, {float: [1.0, 2.0, 3.0]}))
    assert pl.compute(str) == "2.0"


def test_can_depend_on_elements_of_param_table_kwarg() -> None:
    # This is not a valid type annotation, not sure why it works with get_type_hints
    def use_elem(
        *, x: Item((Label(int, 1),), float)  # type: ignore[valid-type]
    ) -> str:
        return str(x)

    pl = sl.Pipeline([use_elem])
    pl.set_param_table(sl.ParamTable(int, {float: [1.0, 2.0, 3.0]}))
    assert pl.compute(str) == "2.0"


def test_can_compute_series_of_param_values() -> None:
    pl = sl.Pipeline()
    pl.set_param_table(sl.ParamTable(int, {float: [1.0, 2.0, 3.0]}))
    assert pl.compute(sl.Series[int, float]) == sl.Series(int, {0: 1.0, 1: 2.0, 2: 3.0})


def test_cannot_compute_series_of_non_table_param() -> None:
    pl = sl.Pipeline()
    # Table for defining length
    pl.set_param_table(sl.ParamTable(int, {float: [1.0, 2.0, 3.0]}))
    pl[str] = 'abc'
    # The alternative option would be to expect to return
    #     sl.Series(int, {0: 'abc', 1: 'abc', 2: 'abc'})
    # For now, we are not supporting this since it is unclear if this would be
    # conceptually sound and risk free.
    with pytest.raises(sl.UnsatisfiedRequirement):
        pl.compute(sl.Series[int, str])


def test_can_compute_series_of_derived_values() -> None:
    def process(x: float) -> str:
        return str(x)

    pl = sl.Pipeline([process])
    pl.set_param_table(sl.ParamTable(int, {float: [1.0, 2.0, 3.0]}))
    assert pl.compute(sl.Series[int, str]) == sl.Series(
        int, {0: "1.0", 1: "2.0", 2: "3.0"}
    )


def test_can_compute_series_of_derived_values_kwarg() -> None:
    def process(*, x: float) -> str:
        return str(x)

    pl = sl.Pipeline([process])
    pl.set_param_table(sl.ParamTable(int, {float: [1.0, 2.0, 3.0]}))
    assert pl.compute(sl.Series[int, str]) == sl.Series(
        int, {0: "1.0", 1: "2.0", 2: "3.0"}
    )


def test_creating_pipeline_with_provider_of_series_raises() -> None:
    series = sl.Series(int, {0: 1.0, 1: 2.0, 2: 3.0})

    def make_series() -> sl.Series[int, float]:
        return series

    with pytest.raises(ValueError):
        sl.Pipeline([make_series])


def test_creating_pipeline_with_series_param_raises() -> None:
    series = sl.Series(int, {0: 1.0, 1: 2.0, 2: 3.0})

    with pytest.raises(ValueError):
        sl.Pipeline([], params={sl.Series[int, float]: series})


def test_explicit_index_of_param_table_is_forwarded_correctly() -> None:
    def process(x: float) -> int:
        return int(x)

    pl = sl.Pipeline([process])
    pl.set_param_table(
        sl.ParamTable(str, {float: [1.0, 2.0, 3.0]}, index=['a', 'b', 'c'])
    )
    assert pl.compute(sl.Series[str, int]) == sl.Series(str, {'a': 1, 'b': 2, 'c': 3})


def test_can_gather_index() -> None:
    Sum = NewType("Sum", float)
    Name = NewType("Name", str)

    def gather(x: sl.Series[Name, float]) -> Sum:
        return Sum(sum(x.values()))

    def make_float(x: str) -> float:
        return float(x)

    pl = sl.Pipeline([gather, make_float])
    pl.set_param_table(sl.ParamTable(Name, {str: ["1.0", "2.0", "3.0"]}))
    assert pl.compute(Sum) == 6.0


def test_can_zip() -> None:
    Sum = NewType("Sum", str)
    Str = NewType("Str", str)
    Run = NewType("Run", int)

    def gather_zip(x: sl.Series[Run, Str], y: sl.Series[Run, int]) -> Sum:
        z = [f'{x_}{y_}' for x_, y_ in zip(x.values(), y.values())]
        return Sum(str(z))

    def use_str(x: str) -> Str:
        return Str(x)

    pl = sl.Pipeline([gather_zip, use_str])
    pl.set_param_table(sl.ParamTable(Run, {str: ['a', 'a', 'ccc'], int: [1, 2, 3]}))

    assert pl.compute(Sum) == "['a1', 'a2', 'ccc3']"


def test_diamond_dependency_pulls_values_from_columns_in_same_param_table() -> None:
    Sum = NewType("Sum", float)
    Param1 = NewType("Param1", int)
    Param2 = NewType("Param2", int)
    Row = NewType("Row", int)

    def gather(x: sl.Series[Row, float]) -> Sum:
        return Sum(sum(x.values()))

    def join(x: Param1, y: Param2) -> float:
        return x / y

    pl = sl.Pipeline([gather, join])
    pl.set_param_table(sl.ParamTable(Row, {Param1: [1, 4, 9], Param2: [1, 2, 3]}))

    assert pl.compute(Sum) == Sum(6)


def test_diamond_dependency_on_same_column() -> None:
    Sum = NewType("Sum", float)
    Param = NewType("Param", int)
    Param1 = NewType("Param1", int)
    Param2 = NewType("Param2", int)
    Row = NewType("Row", int)

    def gather(x: sl.Series[Row, float]) -> Sum:
        return Sum(sum(x.values()))

    def to_param1(x: Param) -> Param1:
        return Param1(x)

    def to_param2(x: Param) -> Param2:
        return Param2(x)

    def join(x: Param1, y: Param2) -> float:
        return x / y

    pl = sl.Pipeline([gather, join, to_param1, to_param2])
    pl.set_param_table(sl.ParamTable(Row, {Param: [1, 2, 3]}))

    assert pl.compute(Sum) == Sum(3)


def test_dependencies_on_different_param_tables_broadcast() -> None:
    Row1 = NewType("Row1", int)
    Row2 = NewType("Row2", int)
    Param1 = NewType("Param1", int)
    Param2 = NewType("Param2", int)
    Product = NewType("Product", str)

    def gather_both(x: sl.Series[Row1, Param1], y: sl.Series[Row2, Param2]) -> Product:
        broadcast = [[x_, y_] for x_ in x.values() for y_ in y.values()]
        return Product(str(broadcast))

    pl = sl.Pipeline([gather_both])
    pl.set_param_table(sl.ParamTable(Row1, {Param1: [1, 2, 3]}))
    pl.set_param_table(sl.ParamTable(Row2, {Param2: [4, 5]}))
    assert pl.compute(Product) == "[[1, 4], [1, 5], [2, 4], [2, 5], [3, 4], [3, 5]]"


def test_dependency_on_other_param_table_in_parent_broadcasts_branch() -> None:
    Row1 = NewType("Row1", int)
    Row2 = NewType("Row2", int)
    Param1 = NewType("Param1", int)
    Param2 = NewType("Param2", int)
    Summed2 = NewType("Summed2", int)
    Product = NewType("Product", str)

    def gather2_and_combine(x: Param1, y: sl.Series[Row2, Param2]) -> Summed2:
        return Summed2(x * sum(y.values()))

    def gather1(x: sl.Series[Row1, Summed2]) -> Product:
        return Product(str(list(x.values())))

    pl = sl.Pipeline([gather1, gather2_and_combine])
    pl.set_param_table(sl.ParamTable(Row1, {Param1: [1, 2, 3]}))
    pl.set_param_table(sl.ParamTable(Row2, {Param2: [4, 5]}))
    assert pl.compute(Product) == "[9, 18, 27]"


def test_dependency_on_other_param_table_in_grandparent_broadcasts_branch() -> None:
    Row1 = NewType("Row1", int)
    Row2 = NewType("Row2", int)
    Param1 = NewType("Param1", int)
    Param2 = NewType("Param2", int)
    Summed2 = NewType("Summed2", int)
    Combined = NewType("Combined", int)
    Product = NewType("Product", str)

    def gather2(x: sl.Series[Row2, Param2]) -> Summed2:
        return Summed2(sum(x.values()))

    def combine(x: Param1, y: Summed2) -> Combined:
        return Combined(x * y)

    def gather1(x: sl.Series[Row1, Combined]) -> Product:
        return Product(str(list(x.values())))

    pl = sl.Pipeline([gather1, gather2, combine])
    pl.set_param_table(sl.ParamTable(Row1, {Param1: [1, 2, 3]}))
    pl.set_param_table(sl.ParamTable(Row2, {Param2: [4, 5]}))
    assert pl.compute(Product) == "[9, 18, 27]"


def test_nested_dependencies_on_different_param_tables() -> None:
    Row1 = NewType("Row1", int)
    Row2 = NewType("Row2", int)
    Param1 = NewType("Param1", int)
    Param2 = NewType("Param2", int)
    Combined = NewType("Combined", int)

    def combine(x: Param1, y: Param2) -> Combined:
        return Combined(x * y)

    pl = sl.Pipeline([combine])
    pl.set_param_table(sl.ParamTable(Row1, {Param1: [1, 2, 3]}))
    pl.set_param_table(sl.ParamTable(Row2, {Param2: [4, 5]}))
    assert pl.compute(sl.Series[Row1, sl.Series[Row2, Combined]]) == sl.Series(
        Row1,
        {
            0: sl.Series(Row2, {0: 4, 1: 5}),
            1: sl.Series(Row2, {0: 8, 1: 10}),
            2: sl.Series(Row2, {0: 12, 1: 15}),
        },
    )
    assert pl.compute(sl.Series[Row2, sl.Series[Row1, Combined]]) == sl.Series(
        Row2,
        {
            0: sl.Series(Row1, {0: 4, 1: 8, 2: 12}),
            1: sl.Series(Row1, {0: 5, 1: 10, 2: 15}),
        },
    )


def test_can_groupby_by_requesting_series_of_series() -> None:
    Row = NewType("Row", int)
    Param1 = NewType("Param1", int)
    Param2 = NewType("Param2", int)

    pl = sl.Pipeline()
    pl.set_param_table(sl.ParamTable(Row, {Param1: [1, 1, 3], Param2: [4, 5, 6]}))
    expected = sl.Series(
        Param1,
        {1: sl.Series(Row, {0: 4, 1: 5}), 3: sl.Series(Row, {2: 6})},
    )
    assert pl.compute(sl.Series[Param1, sl.Series[Row, Param2]]) == expected


def test_groupby_by_requesting_series_of_series_preserves_indices() -> None:
    Row = NewType("Row", int)
    Param1 = NewType("Param1", int)
    Param2 = NewType("Param2", int)

    pl = sl.Pipeline()
    pl.set_param_table(
        sl.ParamTable(Row, {Param1: [1, 1, 3], Param2: [4, 5, 6]}, index=[11, 12, 13])
    )
    assert pl.compute(sl.Series[Param1, sl.Series[Row, Param2]]) == sl.Series(
        Param1, {1: sl.Series(Row, {11: 4, 12: 5}), 3: sl.Series(Row, {13: 6})}
    )


def test_can_groupby_by_param_used_in_ancestor() -> None:
    Row = NewType("Row", int)
    Param = NewType("Param", str)

    pl = sl.Pipeline()
    pl.set_param_table(sl.ParamTable(Row, {Param: ['x', 'x', 'y']}))
    expected = sl.Series(
        Param,
        {"x": sl.Series(Row, {0: "x", 1: "x"}), "y": sl.Series(Row, {2: "y"})},
    )
    assert pl.compute(sl.Series[Param, sl.Series[Row, Param]]) == expected


def test_multi_level_groupby_raises_with_params_from_same_table() -> None:
    Row = NewType("Row", int)
    Param1 = NewType("Param1", int)
    Param2 = NewType("Param2", int)
    Param3 = NewType("Param3", int)

    pl = sl.Pipeline()
    pl.set_param_table(
        sl.ParamTable(
            Row, {Param1: [1, 1, 1, 3], Param2: [4, 5, 5, 6], Param3: [7, 8, 9, 10]}
        )
    )
    with pytest.raises(ValueError, match='Could not find unique grouping node'):
        pl.compute(sl.Series[Param1, sl.Series[Param2, sl.Series[Row, Param3]]])


def test_multi_level_groupby_with_params_from_different_table() -> None:
    Row = NewType("Row", int)
    Param1 = NewType("Param1", int)
    Param2 = NewType("Param2", int)
    Param3 = NewType("Param3", int)

    pl = sl.Pipeline()
    grouping1 = sl.ParamTable(Row, {Param2: [0, 1, 1, 2], Param3: [7, 8, 9, 10]})
    # We are not providing an explicit index here, so this only happens to work because
    # the values of Param2 match range(2).
    grouping2 = sl.ParamTable(Param2, {Param1: [1, 1, 3]})
    pl.set_param_table(grouping1)
    pl.set_param_table(grouping2)
    assert pl.compute(
        sl.Series[Param1, sl.Series[Param2, sl.Series[Row, Param3]]]
    ) == sl.Series(
        Param1,
        {
            1: sl.Series(
                Param2, {0: sl.Series(Row, {0: 7}), 1: sl.Series(Row, {1: 8, 2: 9})}
            ),
            3: sl.Series(Param2, {2: sl.Series(Row, {3: 10})}),
        },
    )


def test_multi_level_groupby_with_params_from_different_table_can_select() -> None:
    Row = NewType("Row", int)
    Param1 = NewType("Param1", int)
    Param2 = NewType("Param2", int)
    Param3 = NewType("Param3", int)

    pl = sl.Pipeline()
    grouping1 = sl.ParamTable(Row, {Param2: [4, 5, 5, 6], Param3: [7, 8, 9, 10]})
    # Note the missing index "6" here.
    grouping2 = sl.ParamTable(Param2, {Param1: [1, 1]}, index=[4, 5])
    pl.set_param_table(grouping1)
    pl.set_param_table(grouping2)
    assert pl.compute(
        sl.Series[Param1, sl.Series[Param2, sl.Series[Row, Param3]]]
    ) == sl.Series(
        Param1,
        {
            1: sl.Series(
                Param2, {4: sl.Series(Row, {0: 7}), 5: sl.Series(Row, {1: 8, 2: 9})}
            )
        },
    )


def test_multi_level_groupby_with_params_from_different_table_preserves_index() -> None:
    Row = NewType("Row", int)
    Param1 = NewType("Param1", int)
    Param2 = NewType("Param2", int)
    Param3 = NewType("Param3", int)

    pl = sl.Pipeline()
    grouping1 = sl.ParamTable(
        Row, {Param2: [4, 5, 5, 6], Param3: [7, 8, 9, 10]}, index=[100, 200, 300, 400]
    )
    grouping2 = sl.ParamTable(Param2, {Param1: [1, 1, 3]}, index=[4, 5, 6])
    pl.set_param_table(grouping1)
    pl.set_param_table(grouping2)
    assert pl.compute(
        sl.Series[Param1, sl.Series[Param2, sl.Series[Row, Param3]]]
    ) == sl.Series(
        Param1,
        {
            1: sl.Series(
                Param2,
                {4: sl.Series(Row, {100: 7}), 5: sl.Series(Row, {200: 8, 300: 9})},
            ),
            3: sl.Series(Param2, {6: sl.Series(Row, {400: 10})}),
        },
    )


def test_multi_level_groupby_with_params_from_different_table_can_reorder() -> None:
    Row = NewType("Row", int)
    Param1 = NewType("Param1", int)
    Param2 = NewType("Param2", int)
    Param3 = NewType("Param3", int)

    pl = sl.Pipeline()
    grouping1 = sl.ParamTable(
        Row, {Param2: [4, 5, 5, 6], Param3: [7, 8, 9, 10]}, index=[100, 200, 300, 400]
    )
    grouping2 = sl.ParamTable(Param2, {Param1: [1, 1, 3]}, index=[6, 5, 4])
    pl.set_param_table(grouping1)
    pl.set_param_table(grouping2)
    assert pl.compute(
        sl.Series[Param1, sl.Series[Param2, sl.Series[Row, Param3]]]
    ) == sl.Series(
        Param1,
        {
            1: sl.Series(
                Param2,
                {6: sl.Series(Row, {400: 10}), 5: sl.Series(Row, {200: 8, 300: 9})},
            ),
            3: sl.Series(Param2, {4: sl.Series(Row, {100: 7})}),
        },
    )


@pytest.mark.parametrize("index", [None, [4, 5, 7]])
def test_multi_level_groupby_raises_on_index_mismatch(
    index: Optional[List[int]],
) -> None:
    Row = NewType("Row", int)
    Param1 = NewType("Param1", int)
    Param2 = NewType("Param2", int)
    Param3 = NewType("Param3", int)

    pl = sl.Pipeline()
    grouping1 = sl.ParamTable(
        Row, {Param2: [4, 5, 5, 6], Param3: [7, 8, 9, 10]}, index=[100, 200, 300, 400]
    )
    grouping2 = sl.ParamTable(Param2, {Param1: [1, 1, 3]}, index=index)
    pl.set_param_table(grouping1)
    pl.set_param_table(grouping2)
    with pytest.raises(ValueError):
        pl.compute(sl.Series[Param1, sl.Series[Param2, sl.Series[Row, Param3]]])


@pytest.mark.parametrize("index", [None, [4, 5, 6]])
def test_groupby_over_param_table(index: Optional[List[int]]) -> None:
    Index = NewType("Index", int)
    Name = NewType("Name", str)
    Param = NewType("Param", int)
    ProcessedParam = NewType("ProcessedParam", int)
    SummedGroup = NewType("SummedGroup", int)
    ProcessedGroup = NewType("ProcessedGroup", int)

    def process_param(x: Param) -> ProcessedParam:
        return ProcessedParam(x + 1)

    def sum_group(group: sl.Series[Index, ProcessedParam]) -> SummedGroup:
        return SummedGroup(sum(group.values()))

    def process(x: SummedGroup) -> ProcessedGroup:
        return ProcessedGroup(2 * x)

    params = sl.ParamTable(
        Index, {Param: [1, 2, 3], Name: ['a', 'a', 'b']}, index=index
    )
    pl = sl.Pipeline([process_param, sum_group, process])
    pl.set_param_table(params)

    graph = pl.get(sl.Series[Name, ProcessedGroup])
    assert graph.compute() == sl.Series(Name, {'a': 10, 'b': 8})


def test_requesting_series_index_that_is_not_in_param_table_raises() -> None:
    Row = NewType("Row", int)
    Param1 = NewType("Param1", int)
    Param2 = NewType("Param2", int)

    pl = sl.Pipeline()
    pl.set_param_table(sl.ParamTable(Row, {Param1: [1, 1, 3], Param2: [4, 5, 6]}))

    with pytest.raises(KeyError):
        pl.compute(sl.Series[int, Param2])


def test_requesting_series_index_that_is_a_param_raises_if_not_grouping() -> None:
    Row = NewType("Row", int)
    Param1 = NewType("Param1", int)
    Param2 = NewType("Param2", int)

    pl = sl.Pipeline()
    pl.set_param_table(sl.ParamTable(Row, {Param1: [1, 1, 3], Param2: [4, 5, 6]}))
    with pytest.raises(ValueError, match='Could not find unique grouping node'):
        pl.compute(sl.Series[Param1, Param2])


def test_generic_providers_work_with_param_tables() -> None:
    Param = TypeVar('Param')
    Row = NewType("Row", int)

    class Str(sl.Scope[Param, str], str):
        ...

    def parametrized(x: Param) -> Str[Param]:
        return Str(f'{x}')

    def make_float() -> float:
        return 1.5

    pipeline = sl.Pipeline([make_float, parametrized])
    pipeline.set_param_table(sl.ParamTable(Row, {int: [1, 2, 3]}))

    assert pipeline.compute(Str[float]) == Str[float]('1.5')
    with pytest.raises(sl.UnsatisfiedRequirement):
        pipeline.compute(Str[int])
    assert pipeline.compute(sl.Series[Row, Str[int]]) == sl.Series(
        Row,
        {
            0: Str[int]('1'),
            1: Str[int]('2'),
            2: Str[int]('3'),
        },
    )


def test_generic_provider_can_depend_on_param_series() -> None:
    Param = TypeVar('Param')
    Row = NewType("Row", int)

    class Str(sl.Scope[Param, str], str):
        ...

    def parametrized_gather(x: sl.Series[Row, Param]) -> Str[Param]:
        return Str(f'{list(x.values())}')

    pipeline = sl.Pipeline([parametrized_gather])
    pipeline.set_param_table(
        sl.ParamTable(Row, {int: [1, 2, 3], float: [1.5, 2.5, 3.5]})
    )

    assert pipeline.compute(Str[int]) == Str[int]('[1, 2, 3]')
    assert pipeline.compute(Str[float]) == Str[float]('[1.5, 2.5, 3.5]')


def test_generic_provider_can_depend_on_derived_param_series() -> None:
    T = TypeVar('T')
    Row = NewType("Row", int)

    class Str(sl.Scope[T, str], str):
        ...

    def use_param(x: int) -> float:
        return x + 0.5

    def parametrized_gather(x: sl.Series[Row, T]) -> Str[T]:
        return Str(f'{list(x.values())}')

    pipeline = sl.Pipeline([parametrized_gather, use_param])
    pipeline.set_param_table(sl.ParamTable(Row, {int: [1, 2, 3]}))

    assert pipeline.compute(Str[float]) == Str[float]('[1.5, 2.5, 3.5]')


def test_params_in_table_can_be_generic() -> None:
    T = TypeVar('T')
    Row = NewType("Row", int)

    class Str(sl.Scope[T, str], str):
        ...

    class Param(sl.Scope[T, str], str):
        ...

    def parametrized_gather(x: sl.Series[Row, Param[T]]) -> Str[T]:
        return Str(','.join(x.values()))

    pipeline = sl.Pipeline([parametrized_gather])
    pipeline.set_param_table(
        sl.ParamTable(Row, {Param[int]: ["1", "2"], Param[float]: ["1.5", "2.5"]})
    )

    assert pipeline.compute(Str[int]) == Str[int]('1,2')
    assert pipeline.compute(Str[float]) == Str[float]('1.5,2.5')


def test_compute_time_handler_works_alongside_param_table() -> None:
    Missing = NewType("Missing", str)

    def process(x: float, missing: Missing) -> str:
        return str(x) + missing

    pl = sl.Pipeline([process])
    pl.set_param_table(sl.ParamTable(int, {float: [1.0, 2.0, 3.0]}))
    pl.get(sl.Series[int, str], handler=sl.HandleAsComputeTimeException())


def test_param_table_column_and_param_of_same_type_can_coexist() -> None:
    pl = sl.Pipeline()
    pl[float] = 1.0
    pl.set_param_table(sl.ParamTable(int, {float: [2.0, 3.0]}))
    assert pl.compute(float) == 1.0
    assert pl.compute(sl.Series[int, float]) == sl.Series(int, {0: 2.0, 1: 3.0})


def test_pipeline_copy_with_param_table() -> None:
    a = sl.Pipeline()
    a.set_param_table(sl.ParamTable(int, {float: [1.0, 2.0, 3.0]}))
    b = a.copy()
    assert b.compute(sl.Series[int, float]) == sl.Series(int, {0: 1.0, 1: 2.0, 2: 3.0})


def test_pipeline_set_param_table_on_original_does_not_affect_copy() -> None:
    a = sl.Pipeline()
    b = a.copy()
    a.set_param_table(sl.ParamTable(int, {float: [1.0, 2.0, 3.0]}))
    assert a.compute(sl.Series[int, float]) == sl.Series(int, {0: 1.0, 1: 2.0, 2: 3.0})
    with pytest.raises(sl.UnsatisfiedRequirement):
        b.compute(sl.Series[int, float])


def test_pipeline_set_param_table_on_copy_does_not_affect_original() -> None:
    a = sl.Pipeline()
    b = a.copy()
    b.set_param_table(sl.ParamTable(int, {float: [1.0, 2.0, 3.0]}))
    assert b.compute(sl.Series[int, float]) == sl.Series(int, {0: 1.0, 1: 2.0, 2: 3.0})
    with pytest.raises(sl.UnsatisfiedRequirement):
        a.compute(sl.Series[int, float])


def test_can_make_html_repr_with_param_table() -> None:
    pl = sl.Pipeline()
    pl.set_param_table(sl.ParamTable(int, {float: [1.0, 2.0, 3.0]}))
    assert pl._repr_html_()


def test_set_param_series_sets_up_pipeline_so_derived_series_can_be_computed() -> None:
    ints = [1, 2, 3]

    def to_str(x: int) -> str:
        return str(x)

    pl = sl.Pipeline((to_str,))
    pl.set_param_series(int, ints)
    assert pl.compute(sl.Series[int, str]) == sl.Series(int, {1: '1', 2: '2', 3: '3'})


def test_multiple_param_series_can_be_broadcast() -> None:
    ints = [1, 2, 3]
    floats = [1.0, 2.0]

    def to_str(x: int, y: float) -> str:
        return str(x) + str(y)

    pl = sl.Pipeline((to_str,))
    pl.set_param_series(int, ints)
    pl.set_param_series(float, floats)
    assert pl.compute(sl.Series[int, sl.Series[float, str]]) == sl.Series(
        int,
        {
            1: sl.Series(float, {1.0: '11.0', 2.0: '12.0'}),
            2: sl.Series(float, {1.0: '21.0', 2.0: '22.0'}),
            3: sl.Series(float, {1.0: '31.0', 2.0: '32.0'}),
        },
    )
