#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A concrete data set."""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Sequence, final, override

from .bins import ADataSet, BinSet
from .measure import MeasureLike


@final
@dataclass(slots=True, frozen=True)
class DataSet[X: MeasureLike[float]](ADataSet[X]):
    data: Sequence[X]  # pyright: ignore[reportIncompatibleMethodOverride]

    if TYPE_CHECKING:
        # We have to re-define these methods because we don't have HKTs.

        @override
        def intbins[T: MeasureLike[int]](self: "DataSet[T]", /) -> BinSet[T, "DataSet[T]"]:
            return super().intbins()  # type: ignore

        @override
        def map[A: MeasureLike[float], B: MeasureLike[float]](self: "DataSet[A]", f: Callable[[A], B], /) -> "DataSet[B]":
            return super().map(f)  # type: ignore


__all__ = ["DataSet"]
