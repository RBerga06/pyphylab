#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Normal distribution."""
from math import erf, exp, sqrt, pi
from dataclasses import dataclass
from typing import Self, final, override

from .data import ADataSet
from .measure import MeasureLike
from .distribution import Distribution, DistributionFit


@final
@dataclass(slots=True, frozen=True)
class Gaussian(Distribution[float]):
    n: int
    µ: float
    s: float

    @property
    @override
    def average(self, /) -> float:
        return self.µ

    @property
    @override
    def variance(self, /) -> float:
        return self.s ** 2

    @override
    def pdf(self, x: float) -> float:
        return exp(-(((x-self.µ)/self.s)**2)/2)/(sqrt(2*pi)*self.s)

    @override
    def p(self, x1: float, x2: float) -> float:
        return (erf((x2-self.µ)/self.s) - erf((x1-self.µ)/self.s))/2

    @override
    def p_worse(self, x: float) -> float:
        return 1 - erf(abs(x - self.µ)/self.s)

    @classmethod
    @override
    def fit[S: ADataSet[MeasureLike[float]]](cls, data: S, /) -> DistributionFit[Self, S]:
        return DistributionFit(cls(data.n, data.average, data.sigma), data)


__all__ = ["Gaussian"]
