#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lazy imports."""

from collections.abc import Sequence
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .dataset import DataSet
    from .measure import MeasureLike
else:
    from typing import Any

    # Since the typing symbols are exported in __all__, they need to be defined at runtime.
    DataSet = Any


def dataset[X: "MeasureLike[float]"](data: Sequence[X], /) -> "DataSet[X]":
    from rberga06.phylab.dataset import DataSet

    return DataSet(data)


__all__ = ["DataSet", "dataset"]
