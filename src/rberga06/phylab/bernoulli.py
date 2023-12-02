#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Poisson distribution."""
from math import factorial
from dataclasses import dataclass
from typing import Self, override

from .measure import MeasureLike
from .distribution import DiscreteDistribution, DistributionFit, ADataSet


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


__all__ = ["Bernoulli"]
