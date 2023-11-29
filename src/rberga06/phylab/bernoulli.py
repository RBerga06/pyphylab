#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Poisson distribution."""
from math import factorial, sqrt
from dataclasses import dataclass
from typing import Sequence, override

from .range import Range
from .measure import MeasureLike
from .distribution import DiscreteDist


def binomial(n: int, k: int, /) -> int:
    return factorial(n)//(factorial(k)*factorial(n-k))


@dataclass(slots=True, frozen=True)
class Bernoulli[M: MeasureLike[int]](DiscreteDist[M]):
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
