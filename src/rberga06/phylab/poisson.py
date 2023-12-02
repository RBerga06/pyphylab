#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Poisson distribution."""
from math import factorial, exp, sqrt
from dataclasses import dataclass
from typing import Self, Sequence, override
from typing_extensions import deprecated

from .range import Range
from .measure import MeasureLike
from .distribution import OldDiscreteDist, DiscreteDistribution, ADataSet, DistributionFit


@dataclass(slots=True, frozen=True)
class Poisson(DiscreteDistribution):
    n: int
    average: float  # pyright: ignore[reportIncompatibleMethodOverride]

    @property
    @override
    def variance(self, /) -> float:
        return self.average

    @override
    def pdf(self, x: int) -> float:
        return (pow(self.average, x) * exp(-self.average))/factorial(x)

    @override
    def p_worse(self, x: float, /) -> float:
        raise NotImplementedError  # TODO: Implement this

    @classmethod
    @override
    def fit[S: ADataSet[MeasureLike[int]]](cls, data: S, /) -> DistributionFit[Self, S]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return DistributionFit(cls(data.n, data.average), data)


@deprecated("Use `Poisson` instead.")
@dataclass(slots=True, frozen=True)
class OldPoisson[M: MeasureLike[int]](OldDiscreteDist[M]):
    data: Sequence[M]
    custom_bins_start: float | None = None
    custom_bins_stop:  float | None = None

    @property
    @override
    def bins_ranges(self, /) -> tuple[Range, ...]:
        if not self.data and None in (self.custom_bins_start, self.custom_bins_stop):
            return ()
        return tuple(map(Range.mk, range(int(self.bins_start), int(self.bins_stop) + 1)))

    @property
    def sigma(self, /) -> float:
        return sqrt(self.average)

    @property
    @override
    def best(self, /) -> float:  # pyright: ignore[reportIncompatibleVariableOverride]
        return self.average

    @property
    @override
    def delta(self, /) -> float:  # pyright: ignore[reportIncompatibleVariableOverride]
        return self.sigma / sqrt(len(self.data))

    @override
    def discrete_probability(self, x: int, /) -> float:
        µ = self.average
        return (pow(µ, x) * exp(-µ))/factorial(x)


__all__ = ["Poisson"]
