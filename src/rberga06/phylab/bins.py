#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split data into bins."""
from dataclasses import dataclass
from itertools import chain
from math import floor, sqrt
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

    @override
    def __repr__(self, /) -> str:
        return f"<Bin: [{self.left}, {self.right}], n={self.n}>"


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


type AnyBinSet[X: MeasureLike[float]] = BinSet[X, _ADataSet[X]]


class ADataSet[X: MeasureLike[float]](_ADataSet[X], Protocol):
    def bins(
        self, nbins: int | None = None, /, *,
        left:  float | None = None,
        right: float | None = None,
    ) -> BinSet[X, Self]:
        """Split `self` into `nbins` bins."""
        if nbins is None:
            nbins = int(floor(sqrt(self.n)))
        if nbins <= 0:
            return BinSet(self, ())
        if left is None:
            left = best(self.min)
        if right is None:
            right = best(self.max)
        # Create bins
        dx = (right - left)/nbins
        bins: list[list[X]] = [[] for _ in range(nbins)]
        # Populate bins
        for x in sorted(self.data, key=best):
            if x == right:
                i = nbins-1
            else:
                i = int(floor((best(x) - left)/dx))
            if i < 0 or i >= nbins:
                continue
            bins[i].append(x)
        # Return the results
        return BinSet(self, tuple([Bin.ranged(bin, left+i*dx, left+(i+1)*dx) for i, bin in enumerate(bins)]))

    def intbins[T: MeasureLike[int]](self: "ADataSet[T]", /) -> BinSet[T, "ADataSet[T]"]:
        """Split `self` (integer data set) into bins."""
        max, min = best(self.max), best(self.min)
        return self.bins(max+1-min, left=min-.5, right=max+.5)


__all__ = ["ADataSet"]
