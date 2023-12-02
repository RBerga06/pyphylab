#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Poisson distribution."""
from math import factorial, exp
from dataclasses import dataclass
from typing import Self, override

from .measure import MeasureLike
from .distribution import DiscreteDistribution, ADataSet, DistributionFit


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


__all__ = ["Poisson"]
