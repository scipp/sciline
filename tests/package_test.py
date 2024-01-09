# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import sciline as pkg


def test_has_version() -> None:
    assert hasattr(pkg, '__version__')
