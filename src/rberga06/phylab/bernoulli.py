#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Poisson distribution."""
from math import factorial, sqrt
from dataclasses import dataclass
from typing import Self, Sequence, override

from .range import Range
from .measure import MeasureLike
from .distribution import OldDiscreteDist, DiscreteDistribution, DistributionFit, ADataSet


def binomial(n: int, k: int, /) -> int:
    return factorial(n)//(factorial(k)*factorial(n-k))


@dataclass(slots=True, frozen=True)
class Bernoulli(DiscreteDistribution):
    n: int
    n_trials:  int
    p_success: float

    @property
    @override
    def variance(self, /) -> float:
        return self.n_trials * self.p_success * (1 - self.p_success)

    @property
    @override
    def average(self, /) -> float:
        return self.p_success * self.n_trials

    @override
    def pdf(self, x: int) -> float:
        n, p = self.n_trials, self.p_success
        return binomial(n, x) * pow(p, x) * pow(1-p, n-x)

    @override
    def p_worse(self, x: float, /) -> float:
        raise NotImplementedError  # TODO: Implement this

    @classmethod
    @override
    def fit[S: ADataSet[MeasureLike[int]]](  # pyright: ignore[reportIncompatibleMethodOverride]
        cls, data: S, /, *,
        n_trials: int,
        p_success: float
    ) -> DistributionFit[Self, S]:
        return DistributionFit(cls(data.n, n_trials, p_success), data)


@dataclass(slots=True, frozen=True)
class OldBernoulli[M: MeasureLike[int]](OldDiscreteDist[M]):
    data: Sequence[M]
    p: float
    n: int
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
        return sqrt(self.n * self.p * (1 - self.p))

    @property
    @override
    def expected_avg(self, /) -> float:
        return self.p * self.n

    @property
    @override
    def best(self, /) -> float:  # pyright: ignore[reportIncompatibleVariableOverride]
        return self.expected_avg

    @property
    @override
    def delta(self, /) -> float:  # pyright: ignore[reportIncompatibleVariableOverride]
        return self.sigma / sqrt(len(self.data))

    @override
    def discrete_probability(self, x: int, /) -> float:
        return binomial(self.n, x) * pow(self.p, x) * pow(1-self.p, self.n-x)


__all__ = ["Bernoulli"]
