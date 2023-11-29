#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pyright: reportIncompatibleMethodOverride=false
# pyright: reportIncompatibleVariableOverride=false
"""Data statistics"""
from dataclasses import dataclass
from math import sqrt
from typing import TYPE_CHECKING, Callable, Iterator, Protocol, Self, Sequence, final, overload, override
from .measure import Measure, MeasureLike, best


def square[X: MeasureLike[float]](x: X, /) -> X:
    return x ** 2  # type: ignore


class DataSequence[T](Protocol):
    """Data sequence wrapper."""

    if TYPE_CHECKING:
        @property
        def data(self, /) -> Sequence[T]: ...
    else:
        data: Sequence[T]

    def map[A, B](self: "DataSequence[A]", f: Callable[[A], B], /) -> "DataSequence[B]":
        return type(self)(tuple(map(f, self.data)))  # type: ignore

    def __len__(self, /) -> int:
        return len(self.data)

    def __iter__(self, /) -> Iterator[T]:
        return iter(self.data)

    @overload
    def __getitem__(self, key: int, /) -> T: ...
    @overload
    def __getitem__(self, key: slice, /) -> Self: ...
    def __getitem__(self, key: int | slice, /) -> T | Self:
        if isinstance(key, slice):
            return type(self)(self.data[key])
        return self.data[key]


class AbstractStats(Protocol):
    """Statistics."""
    n: int

    @property
    def sum(self, /) -> float: ...

    @property
    def average(self, /) -> float:
        return self.sum/self.n

    @property
    def variance(self, /) -> float: ...

    @property
    def sigma(self, /) -> float:
        return sqrt(self.variance)


class DataStats[X: MeasureLike[float]](AbstractStats, Protocol):
    """Statistics on data."""
    if TYPE_CHECKING:
        @property
        def data(self, /) -> Sequence[X]: ...
    else:
        data: Sequence[X]

    @property
    @override
    def n(self, /) -> int:
        return len(self.data)

    @property
    @override
    def sum(self, /) -> float:
        return sum(map(best, self.data))

    @property
    @override
    def variance(self, /) -> float:
        return sum([best(x)**2 for x in self.data])/self.n - self.average**2


class ADataSet[X: MeasureLike[float]](DataSequence[X], DataStats[X], Measure[float], Protocol):
    """Abstract DataSet."""

    @property
    @override
    def best(self, /) -> float:
        return self.average

    @property
    @override
    def delta(self, /) -> float:
        return self.sigma/sqrt(self.n)

    # We have to re-define this because we don't have HKTs.
    @override
    def map[A: MeasureLike[float], B: MeasureLike[float]](self: "ADataSet[A]", f: Callable[[A], B], /) -> "ADataSet[B]":
        return super().map(f)  # type: ignore


@final
@dataclass(slots=True, frozen=True)
class DataSet[X: MeasureLike[float]](ADataSet[X]):
    data: Sequence[X]

    # We have to re-define this because we don't have HKTs.
    @override
    def map[A: MeasureLike[float], B: MeasureLike[float]](self: "DataSet[A]", f: Callable[[A], B], /) -> "DataSet[B]":
        return super().map(f)  # type: ignore


__all__ = ["DataSequence", "DataStats", "ADataSet", "DataSet"]
