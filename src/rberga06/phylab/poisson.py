#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Poisson distribution."""
from math import factorial, exp, sqrt
from dataclasses import dataclass
from typing import Sequence, override

from .range import Range
from .measure import Measure
from .distribution import DiscreteDist


@dataclass(slots=True, frozen=True)
class Poisson[M: Measure](DiscreteDist[M]):
    data: Sequence[M]

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

    @property
    @override
    def binsr(self, /) -> tuple[Range, ...]:
        left  = min(self.data, key=lambda m: m.best)
        right = max(self.data, key=lambda m: m.best)
        return tuple(map(Range.mk, range(int(left.best), int(right.best) + 1)))


__all__ = ["Poisson"]
