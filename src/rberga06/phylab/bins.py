#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split data into bins."""
from dataclasses import dataclass
from itertools import chain
from math import floor
from typing import Any, Callable, Protocol, Self, Sequence, final, override

from .measure import MeasureLike, best
from .data import ADataSet as _ADataSet
from ._lazy import DataSet, dataset


@final
@dataclass(slots=True, frozen=True)
class Bin[X: MeasureLike[float]](_ADataSet[X]):
    data: Sequence[X]  # pyright: ignore[reportIncompatibleMethodOverride]
    left: float
    center: float
    right: float

    @property
    @override
    def best(self, /) -> float:
        return self.center

    @override
    def map[A: MeasureLike[float], B: MeasureLike[float]](self: "Bin[A]", f: Callable[[A], B], /) -> "DataSet[B]":
        return dataset(self.data).map(f)

    @classmethod
    def ranged(cls, data: Sequence[X], left: float, right: float, /) -> Self:
        return cls(data, left, (left + right)/2, right)

    @classmethod
    def centered(cls, data: Sequence[X], center: float, delta: float, /) -> Self:
        return cls(data, center - delta, center, center + delta)


@final
@dataclass(frozen=True, slots=True)
class BinSet[X: MeasureLike[float], D: _ADataSet[MeasureLike[float]]](_ADataSet[X]):
    """A `DataSet` of `Bin`s"""
    orig: D
    """Original data."""
    bins: Sequence[Bin[X]]

    @property
    @override
    def data(self, /) -> tuple[X, ...]:
        return tuple(chain.from_iterable([[b.best]*b.n for b in self.bins]))  # type: ignore

    @property
    @override
    def n(self, /) -> int:
        return len(self.data)

    @override
    def map[A: MeasureLike[float], B: MeasureLike[float]](self: "BinSet[A, Any]", f: Callable[[A], B], /) -> "DataSet[B]":
        return DataSet(self.data).map(f)


class ADataSet[X: MeasureLike[float]](_ADataSet[X], Protocol):
    def bins(self, nbins: int, /) -> BinSet[X, Self]:
        """Split `self` into `nbins` bins."""
        # Create bins
        min = best(self.min)
        dx = (best(self.max) - min)/nbins
        bins: list[list[X]] = [[] for _ in range(nbins)]
        # Populate bins
        data: list[X] = sorted(self.data, key=best)
        for x in data:
            i = int(floor((best(x) - min)/dx))
            if i == nbins:  # [...)[...)[...)[...] <- self.max is included in rightmost bin
                i -= 1
            bins[i].append(x)
        return BinSet(self, tuple([Bin.ranged(bin, min+i*dx, min+(i+1)*dx) for i, bin in enumerate(bins)]))


__all__ = ["ADataSet"]
