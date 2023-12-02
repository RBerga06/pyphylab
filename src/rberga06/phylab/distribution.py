#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dataclasses import dataclass
from math import ceil, floor
from typing import Any, Protocol, Self, final, override

from .measure import Measure, MeasureLike
from .data import ADataSet, AbstractStats


@final
@dataclass(frozen=True, slots=True)
class DistributionFit[D: "Distribution[Any]", S: ADataSet[MeasureLike[float]]]:
    """A container for distribution fit results."""
    dist: D
    data: S


class Distribution[T: float](AbstractStats, Measure[T], Protocol):
    """An (abstract) distribution."""
    n: int

    @property
    @override
    def average(self, /) -> float:
        ...

    @property
    @override
    def sum(self, /) -> int:
        return self.n

    @property
    @override
    def best(self, /) -> float:
        return self.average

    @property
    @override
    def delta(self, /) -> float:
        return self.sigma_avg

    def pdf(self, x: T, /) -> float:
        """The probability density function (PDF), evaluated at x."""
        ...

    def p(self, x1: float, x2: float, /) -> float:
        """The probability of getting something between x1 and x2."""
        ...

    def p_worse(self, x: float, /) -> float:
        """The probability of getting x, or worse."""
        ...

    def chauvenet(self, x: T, /) -> bool:
        """Check if we can apply Chauvenet to x."""
        return (self.n * self.p_worse(x)) < .5

    def expected(self, x1: float, x2: float, /) -> float:
        """The expected value between `x1` and `x2`."""
        return self.p(x1, x2) * self.n

    def bins(self, nbins: int, left: float, right: float, /) -> tuple[float, ...]:
        step = (right - left)/nbins
        return tuple([self.expected(left+i*step, left+(i+1)*step) for i in range(nbins+1)])

    def intbins(self, left: int, right: int, /) -> tuple[float, ...]:
        return self.bins(right+1-left, left, right+1)

    def sample(self, n: int, /) -> tuple[float, ...]:
        """Return pseudo-random data with this distribution."""
        raise NotImplementedError  # TODO: Implement this!

    @classmethod
    def fit[S: ADataSet[MeasureLike[float]]](cls, data: S, /) -> DistributionFit[Self, S]:
        """Find the distribution that best fits `data`."""
        ...


class DiscreteDistribution(Distribution[int], Protocol):
    @override
    def p(self, x1: float, x2: float, /) -> float:
        return sum([*map(self.pdf, range(int(ceil(x1)), int(floor(x2))+1))])


__all__ = ["DistributionFit", "Distribution", "DiscreteDistribution"]
