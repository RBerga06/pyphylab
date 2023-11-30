#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Normal distribution."""
from math import erf, exp, floor, sqrt, pi
from dataclasses import dataclass
from typing import Self, Sequence, final, override
from typing_extensions import deprecated

from .data import ADataSet
from .range import Range
from .measure import AnyMeasure, MeasureLike
from .distribution import Dist, Distribution, DistFit


@final
@dataclass(slots=True, frozen=True)
class Gaussian(Distribution[float]):
    average: float
    sigma:   float

    @override
    def pdf(self, x: float) -> float:
        return exp(-(((x-self.average)/self.sigma)**2)/2)/(sqrt(2*pi)*self.sigma)

    @override
    def p(self, x1: float, x2: float) -> float:
        return (erf((x2-self.average)/self.sigma) - erf((x1-self.average)/self.sigma))/2

    @override
    def p_worse(self, x: float) -> float:
        return 1 - erf(abs(x - self.average)/self.sigma)

    @classmethod
    @override
    def fit[S: ADataSet[MeasureLike[float]]](cls, data: S, /, *, bins: int | None = None) -> DistFit[Self, S, MeasureLike[float]]:
        return DistFit(cls(data.average, data.sigma), data, data.bins(bins))


@dataclass(slots=True, frozen=True)
@deprecated("Use `Gaussian` instead.")
class Normal[M: AnyMeasure](Dist[M]):
    data: Sequence[M]

    @property
    def sigma(self, /) -> float:
        µ = self.average
        N = len(self.data)
        return sqrt(sum([len(b)*(b.center - µ)**2 for b in self.bins])/(N-1))

    @property
    def nbins(self, /) -> int:
        return int(floor(sqrt(len(self.data))))

    @property
    @override
    def best(self, /) -> float:  # pyright: ignore[reportIncompatibleVariableOverride]
        return self.average

    @property
    @override
    def delta(self, /) -> float:  # pyright: ignore[reportIncompatibleVariableOverride]
        return self.sigma / sqrt(len(self.data))

    @override
    def probability(self, r: Range, /) -> float:
        µ, s = self.average, self.sigma
        return (erf((r.right-µ)/s) - erf((r.left-µ)/s))/2

    @property
    @override
    def bins_ranges(self, /) -> tuple[Range, ...]:
        left  = min(self.data, key=lambda m: m.best)
        right = max(self.data, key=lambda m: m.best)
        return Range("[", left.best - left.delta, right.best + right.delta, "]").split(self.nbins)


__all__ = ["Gaussian"]
